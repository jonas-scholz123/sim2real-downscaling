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
from config import paths
from gridder import Gridder

# %%
dwd_sd = DWDSTationData(paths)
era5 = xr.open_dataset(paths.era5, engine="cfgrib")
era5 = era5["t2m"] - 273.15
# %%


def saveplot(name):
    target_dir = "./outputs/figures/era5_investigation/"
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(f"{target_dir}/{name}.pdf", bbox_inches="tight")


def index(a, x):
    "Locate the leftmost value exactly equal to x"
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


class Comparer:
    def __init__(
        self, dwd_sd: DWDSTationData, era5: xr.Dataset, start, end, freq="1H"
    ) -> None:
        gridder = Gridder(era5["latitude"].values, era5["longitude"].values)
        dwd_sd.apply_grid(gridder)
        dwd_df = dwd_sd.between_datetimes(start, end, freq)
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
        agg_df = self.df.groupby("STATION_ID").agg(
            {
                "geometry": "first",
                "TEMP_DIFF": agg,
            }
        )
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

    def agg_rmse_hist(self, remove_outliers: bool = False, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        rmse = lambda x: np.sqrt(np.mean(x**2))
        agg_df = self.agg_df(rmse, remove_outliers)
        ax.hist(agg_df["TEMP_DIFF"], bins=30)
        ax.set_xlabel("$\Delta T$ [°C]")
        ax.set_ylabel("Count")

    def err_hist(self, remove_outliers: bool = False, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))

        df = self.df
        if remove_outliers:
            df = self.remove_outliers(self.df)

        ax.hist(df["TEMP_DIFF"], bins=30)
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
    c.err_hist(remove_outliers=False, ax=ax)
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
    c.agg_rmse_hist(remove_outliers=False, ax=ax)
    ax.set_title(f"Month = {month}")
plt.tight_layout()
saveplot("agg_rmse_hist_by_month")
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
dwd_sd = DWDSTationData(paths)
df = dwd_sd.between_datetimes("2022-01-01", "2022-01-05")
# %%

gridder = Gridder(era5["latitude"].values, era5["longitude"].values)

gridder.grid_latlons(df)
