# %%
import pandas as pd
import numpy as np
import xarray as xr
from bisect import bisect_left
import matplotlib.pyplot as plt
from typing import Callable
from datetime import datetime, timedelta
import os

from datasets import DWDSTationData
from plots import plot_geopandas

import geopandas as gpd


def saveplot(name):
    target_dir = "./outputs/figures/era5_investigation/"
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(f"{target_dir}/{name}.pdf", bbox_inches="tight")


def plot_geopandas(gdf, column="TEMP", fig_ax=None, legend=True, vmin=None, vmax=None):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    country = gpd.read_file("data/shapefiles/DEU_adm0.shp")
    country.crs = "epsg:4326"
    gdf.plot(column=column, ax=ax, legend=legend, vmin=vmin, vmax=vmax)
    country.plot(edgecolor="black", ax=ax, alpha=1, facecolor="none")
    return fig, ax


def index(a, x):
    "Locate the leftmost value exactly equal to x"
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


class Gridder:
    def __init__(self, lats, lons):
        self.lat_grid = pd.unique(lats - lats.astype(int))

        self.lon_grid = pd.unique(lons - lons.astype(int))

    def closest(self, grid, val):
        int_val = int(val)
        remainder = val - int_val
        return round(int_val + min(grid, key=lambda x: abs(x - remainder)), 3)

    def closest_lat(self, lat):
        return self.closest(self.lat_grid, lat)

    def closest_lon(self, lon):
        return self.closest(self.lon_grid, lon)

    def closest_latlon(self, lat, lon):
        return self.closest_lat(lat), self.closest_lon(lon)

    def grid_latlons(self, df):
        df[["GRID_LAT", "GRID_LON"]] = df.apply(
            lambda row: self.closest_latlon(row.LAT, row.LON),
            result_type="expand",
            axis=1,
        )

        grid_geom = gpd.points_from_xy(df["GRID_LON"], df["GRID_LAT"], crs="epsg:4326")
        grid_geom = gpd.GeoSeries(grid_geom)

        grid_geom = grid_geom.to_crs(crs=3310)
        df2 = df.to_crs(crs=3310)
        df["GRID_DIST"] = grid_geom.distance(df2["geometry"])
        return df


# %%
dwd_sd = DWDSTationData("data/raw/dwd/airtemp2m/unzipped", "2000-01-01", "today")
era5 = xr.open_dataset("data/raw/ERA_5_Germany/1.grib", engine="cfgrib")
era5 = era5["t2m"] - 273.15


# %%
class Comparer:
    def __init__(self, dwd_sd, era5, start, end, freq="1H") -> None:
        gridder = Gridder(era5["latitude"].values, era5["longitude"].values)
        dwd_df = dwd_sd.between_datetimes(start, end, "7H")
        dwd_df = gridder.grid_latlons(dwd_df)

        era5_df = era5[(era5["time"] > start) & (era5["time"] < end)].to_dataframe()

        df = dwd_df.merge(
            era5_df,
            left_on=["GRID_LAT", "GRID_LON", "DATETIME"],
            right_on=["latitude", "longitude", "time"],
        )

        df = df.drop(
            [
                "LAT",
                "LON",
                "number",
                "step",
                "surface",
                "valid_time",
            ],
            axis=1,
        )

        df = df.rename({"TEMP": "TEMP_REAL", "t2m": "TEMP_ERA5"}, axis=1)
        df["TEMP_DIFF"] = df["TEMP_ERA5"] - df["TEMP_REAL"]
        self.df = df

    def remove_outliers(self, df):
        # This station (UFS TW Ems) is often very far off.
        df = df[df["STATION_ID"] != 1228]
        df = df[df["HEIGHT"] < 800]
        return df

    def rmse_map(self, remove_outliers: bool = False, **kwargs):
        rmse = lambda x: np.sqrt(np.mean(x**2))
        self.agg_map(rmse, remove_outliers, **kwargs)

    def abs_mean_map(self, remove_outliers: bool = False):
        agg = lambda x: np.abs(x).mean()
        self.agg_map(agg, remove_outliers)

    def agg_df(self, agg: Callable, remove_outliers: bool = False):
        agg_df = self.df.groupby(["STATION_ID", "geometry"], as_index=False).agg(agg)
        agg_df = agg_df.set_geometry("geometry")
        if remove_outliers:
            agg_df = self.remove_outliers(agg_df)
        return agg_df

    def agg_map(self, agg: Callable, remove_outliers: bool = False, **kwargs):
        agg_df = self.agg_df(agg, remove_outliers)
        plot_geopandas(agg_df, "TEMP_DIFF", **kwargs)

    def rmse_height(self, remove_outliers: bool = False):
        rmse = lambda x: np.sqrt(np.mean(x**2))
        agg_df = self.agg_df(rmse, remove_outliers)
        plt.scatter(agg_df["HEIGHT"], agg_df["TEMP_DIFF"])
        plt.xlabel("Height [m]")
        plt.ylabel("$\sqrt{MSE(T)}$ [°C]")

    def rmse_grid_dist(self, remove_outliers: bool = False):
        rmse = lambda x: np.sqrt(np.mean(x**2))
        agg_df = self.agg_df(rmse, remove_outliers)
        plt.scatter(agg_df["GRID_DIST"], agg_df["TEMP_DIFF"])
        plt.xlabel("Station Distance to Grid [m]")
        plt.ylabel("$\sqrt{MSE(T)}$ [°C]")

    def rmse_hist(self, remove_outliers: bool = False, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        rmse = lambda x: np.sqrt(np.mean(x**2))
        agg_df = self.agg_df(rmse, remove_outliers)
        ax.hist(agg_df["TEMP_DIFF"], bins=30)
        ax.set_xlabel("$\sqrt{MSE(T)}$ [°C]")
        ax.set_ylabel("Count")

    def rmse_boxplot(self, remove_outliers: bool = False, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))

        df = self.df.copy()
        if remove_outliers:
            df = self.remove_outliers(df)

        ax.boxplot(df["TEMP_DIFF"])


# %%

fig, axs = plt.subplots(4, 3, figsize=(10, 20))
for ax, month in zip(axs.flatten(), range(1, 13)):
    start = pd.to_datetime(datetime(year=2022, month=month, day=1))
    end = pd.to_datetime(start + timedelta(30))
    c = Comparer(dwd_sd, era5, start, end, "18H")
    c.rmse_map(remove_outliers=False, fig_ax=(fig, ax), vmin=-2, vmax=8, legend=False)
    ax.set_title(f"Month = {month}")
saveplot("rmse_map_by_month")
plt.show()
# %%
fig, axs = plt.subplots(4, 3, figsize=(10, 15))
for ax, month in zip(axs.flatten(), range(1, 13)):
    start = pd.to_datetime(datetime(year=2022, month=month, day=1))
    end = pd.to_datetime(start + timedelta(30))
    c = Comparer(dwd_sd, era5, start, end, "18H")
    c.rmse_hist(remove_outliers=False, ax=ax)
    ax.set_title(f"Month = {month}")
plt.tight_layout()
saveplot("rmse_hist_by_month")
plt.show()
# %%
fig, axs = plt.subplots(4, 3, figsize=(10, 15))
for ax, month in zip(axs.flatten(), range(1, 13)):
    start = pd.to_datetime(datetime(year=2022, month=month, day=1))
    end = pd.to_datetime(start + timedelta(30))
    c = Comparer(dwd_sd, era5, start, end, "18H")
    c.rmse_boxplot(remove_outliers=True, ax=ax)
    ax.set_title(f"Month = {month}")
plt.tight_layout()
saveplot("rmse_boxplot_by_month_no_outliers")
plt.show()

# %%

# TODO: By distance to grid point.

start = pd.to_datetime("2022-03-03 12:00:00")
end = pd.to_datetime("2022-04-03 12:00:00")
c = Comparer(dwd_sd, era5, start, end, "5H")
c.rmse_height(False)
saveplot("rmse_vs_height")
plt.show()
c.rmse_grid_dist()
saveplot("rmse_vs_grid_dist")
plt.show()

# %%
plt.hist(rmse_df["TEMP_DIFF"], bins=30)
plt.xlabel("$\sqrt{MSE(T)}$ [°C]")
# plt.scatter(grouped["HEIGHT"], grouped["TEMP_DIFF"])

# grouped[grouped["TEMP_DIFF"] == grouped["TEMP_DIFF"].min()]
plot_geopandas(rmse_df, "TEMP_DIFF")

# %%
no_df = rmse_df[rmse_df["TEMP_DIFF"] < 5]
plot_geopandas(no_df, "TEMP_DIFF")
# %%
plt.scatter(rmse_df["HEIGHT"], rmse_df["TEMP_DIFF"])
plt.xlabel("Height [m]")
plt.ylabel("$\sqrt{MSE(T)}$ [°C]")
# %%
high_err = rmse_df.sort_values("TEMP_DIFF", ascending=False).iloc[:10, :]
plot_geopandas(high_err[high_err["HEIGHT"] < 500], "TEMP_DIFF")
sid = int(high_err[high_err["HEIGHT"] == 0]["STATION_ID"])
# %%
df[df["STATION_ID"] == sid]
# %%
td = df["TEMP_DIFF"]
plt.boxplot(td)
plt.ylabel("$T_{ERA5} - T_{real}$")
# %%
fig, ax = plot_geopandas(df, "TEMP_DIFF")
ax.set_title("$T_{ERA5} - T_{real}$")
# %%
df[df["TEMP_DIFF"] == df["TEMP_DIFF"].min()]

# %%
era5.where((era5["time"] > start) & (era5["time"] < end))
# %%
# %%
dwd_df
