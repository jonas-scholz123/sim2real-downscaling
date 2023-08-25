# %%
import pandas as pd
import numpy as np
import xarray as xr
from bisect import bisect_left
import matplotlib.pyplot as plt
from typing import Callable
from datetime import datetime, timedelta
import os

from sim2real.datasets import DWDStationData
from sim2real.plots import plot_geopandas
from sim2real.config import paths, names
from sim2real.gridder import Gridder
from sim2real.plots import save_plot

# %%
dwd_sd = DWDStationData(paths)
era5 = xr.open_dataset(paths.era5)
# %%


def index(a, x):
    "Locate the leftmost value exactly equal to x"
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


class Comparer:
    def __init__(
        self, dwd_sd: DWDStationData, era5: xr.Dataset, start, end, freq="1H"
    ) -> None:
        gridder = Gridder(era5[names.lat].values, era5[names.lon].values)
        dwd_sd.apply_grid(gridder)
        dwd_df = dwd_sd.between_datetimes(start, end, freq)

        era5_df = era5.where(
            (era5[names.time] > start) & (era5[names.time] < end), drop=True
        ).to_dataframe()

        df = (
            dwd_df.reset_index()
            .merge(
                era5_df,
                left_on=["GRID_LAT", "GRID_LON", names.time],
                right_on=[names.lat, names.lon, names.time],
                suffixes=("_REAL", "_ERA5"),
            )
            .set_index([names.time, names.station_id])
        )

        df = df.drop(
            [
                names.lat,
                names.lon,
                "number",
                "step",
                "surface",
                "valid_time",
            ],
            axis=1,
        )
        df["TEMP_DIFF"] = df[f"{names.temp}_ERA5"] - df[f"{names.temp}_REAL"]
        self.df = df

    def remove_outliers(self, df):
        # Station 1228 (UFS TW Ems) is often very far off.
        df = df.query(f"{names.station_id} != 1228 & {names.height} < 800")
        return df

    def rmse_map(self, remove_outliers: bool = False, **kwargs):
        rmse = lambda x: np.sqrt(np.mean(x**2))
        self.agg_map(rmse, remove_outliers, **kwargs)

    def abs_mean_map(self, remove_outliers: bool = False):
        agg = lambda x: np.abs(x).mean()
        self.agg_map(agg, remove_outliers)

    def agg_df(self, agg: Callable, remove_outliers: bool = False):
        agg_df = self.df.groupby(names.station_id).agg(
            {
                "geometry": "first",
                names.height: "first",
                "GRID_DIST": "first",
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

    def rmse_height(self, remove_outliers: bool = False, **kwargs):
        rmse = lambda x: np.sqrt(np.mean(x**2))
        agg_df = self.agg_df(rmse, remove_outliers)
        plt.scatter(agg_df[names.height], agg_df["TEMP_DIFF"], **kwargs)
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
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
start = pd.to_datetime("2021-01-01")
end = pd.to_datetime("2022-12-12")
c = Comparer(dwd_sd, era5, start, end, "18H")
# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# c.rmse_map(remove_outliers=True, fig_ax=(fig, ax), vmin=-2, vmax=8, legend=False)
df = c.agg_df(agg=lambda x: np.abs(x).mean())
# ax.set_title(f"Month = {month}")
# saveplot("rmse_map_by_month")
plot_geopandas(df, "TEMP_DIFF", vmin=0, vmax=3, s=4**2, fig_ax=(fig, ax))
# %%

fig, axs = plt.subplots(4, 3, figsize=(10, 20))
for ax, month in zip(axs.flatten(), range(1, 13)):
    start = pd.to_datetime(datetime(year=2022, month=month, day=1))
    end = pd.to_datetime(start + timedelta(30))
    c = Comparer(dwd_sd, era5, start, end, "18H")
    c.rmse_map(remove_outliers=False, fig_ax=(fig, ax), vmin=-2, vmax=8, legend=False)
    ax.set_title(f"Month = {month}")
save_plot(None, "rmse_map_by_month", fig)
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
save_plot(None, "rmse_hist_by_month")
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
save_plot(None, "agg_rmse_hist_by_month")
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
save_plot(None, "rmse_boxplot_by_month_no_outliers")
plt.show()

# %%
start = pd.to_datetime("2022-03-03 12:00:00")
end = pd.to_datetime("2022-04-03 12:00:00")
c = Comparer(dwd_sd, era5, start, end, "5H")
c.rmse_height(False, s=2**3)
save_plot(None, "rmse_vs_height")
plt.show()
c.rmse_grid_dist()
save_plot(None, "rmse_vs_grid_dist")
plt.show()

# %%

fig, axs = plt.subplots(1, 2, figsize=(9, 1.5), sharey=True)
agg = lambda x: np.abs(x).mean()
agg_df = c.agg_df(agg, False)
size = 4**2

mean = agg_df["TEMP_DIFF"].mean()


axs[0].scatter(
    agg_df["GRID_DIST"],
    agg_df["TEMP_DIFF"],
    s=size,
    marker="x",
    linewidth=0.35,
    label="DWD Stations",
)
axs[0].set_ylabel("$MAE(T)$ [°C]")
axs[0].set_xlabel("Station Distance to Grid [m]")

axs[1].scatter(
    agg_df[names.height],
    agg_df["TEMP_DIFF"],
    s=size,
    marker="x",
    linewidth=0.35,
    label="DWD Stations",
)
axs[1].set_xlabel("Height [m]")

for ax in axs:
    ax.axhline(mean, color="C1", label=f"Mean = {mean:.2f}°C")
    ax.legend(loc="upper left")

save_plot(None, "mae_grid_and_height", fig)
