# %%
from sim2real.datasets import (
    load_era5,
    load_elevation,
    DWDStationData,
    load_station_splits,
)
from sim2real.config import paths
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from sim2real.config import paths, names
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize

from sim2real.plots import add_germany_lines, save_plot


def lons_and_lats(df):
    lats = df.index.get_level_values(names.lat)
    lons = df.index.get_level_values(names.lon)
    return lons, lats


era5 = load_era5()
dwd = DWDStationData(paths)
elevation = load_elevation().coarsen({"LAT": 25, "LON": 25}, boundary="trim").mean()

# %%
cmap = "coolwarm"

date = pd.Timestamp("2022-04-03 12:00:00")

transform = ccrs.PlateCarree()
proj = ccrs.TransverseMercator(central_longitude=10, approx=False)
fig, axs = plt.subplots(
    subplot_kw={"projection": proj},
    nrows=1,
    ncols=6,
    figsize=(10, 3),
    gridspec_kw={"width_ratios": [1, 1, 0.2, 0.3, 1, 0.2], "wspace": 0.05},
)
for i, ax in enumerate(axs):
    if i in [2, 3, 5]:
        continue
    add_germany_lines(ax)

era5_date = era5.sel({"TIME": date})["T2M"]
truth = dwd.at_datetime(date).loc[date].set_index(["LAT", "LON"])

vmin, vmax = era5_date.min(), era5_date.max()
# vmin = min(vmin, truth.T2M.min())
# vmax = min(vmax, truth.T2M.max())

era5_date.plot(
    ax=axs[0],
    transform=transform,
    add_colorbar=False,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)


axs[0].set_title("ERA5 Data [°C]")

axs[1].set_title("DWD Station Data [°C]")

im = axs[1].scatter(
    *lons_and_lats(truth),
    s=3**2,
    c=truth[names.temp],
    transform=transform,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), ax=axs[2], fraction=1.0
)
axs[3].remove()

elevation.HEIGHT.plot(
    ax=axs[4],
    transform=transform,
    add_colorbar=False,
    cmap="viridis",
)

vmin, vmax = float(elevation.HEIGHT.min()), float(elevation.HEIGHT.max())

axs[4].set_title("Elevation Data [m]")

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap="viridis"),
    ax=axs[5],
    fraction=1.0,
)
axs[2].remove()
axs[5].remove()
save_plot(None, "data samples", ext="png", dpi=500)
# %%
from sim2real.plots import init_fig


def plot_train_val(n_stat, ax=None, val_frac=0.2, **kwargs):
    if ax is None:
        fig, axs = init_fig()
        ax = axs[0]
    splits = load_station_splits()

    n_train = int((1 - val_frac) * n_stat)
    n_val = n_stat - n_train
    train_stations = (
        splits[splits["SET"] == "TRAIN"].sort_values("ORDER").index[:n_train]
    )
    val_stations = splits[splits["SET"] == "VAL"].sort_values("ORDER").index[:n_val]
    test_stations = splits[splits["SET"] == "TEST"].index

    dwd.plot_stations(train_stations, "o", "C0", ax=ax, label="Train", **kwargs)
    dwd.plot_stations(val_stations, "s", "C1", ax=ax, label="Val", **kwargs)
    dwd.plot_stations(test_stations, "^", "C2", ax=ax, label="Test", **kwargs)


fig, axs = init_fig(1, 3, (8, 3))
markersizes = [5**2, 4**2, 2.5**2]

for i, (n_stat, msize) in enumerate(zip([20, 100, 500], markersizes)):
    plot_train_val(n_stat, markersize=msize, ax=axs[i])
    axs[i].set_title(f"$N_{{stations}} = {n_stat}$")
axs[0].legend(loc="upper left")
save_plot(None, "train_val_stations")


# %%
points = dwd.meta_df["geometry"]
# cdist(points, points)

df = dwd.meta_df.groupby("STATION_ID").last()
df.crs = "epsg:4326"
df = df.to_crs(crs=3310)
dists = df.geometry.apply(lambda g: df.distance(g)).values
# %%
dists[dists == 0.0] += 1000000

min_dists = dists.min(axis=1) / 1000
ymin = 0
ymax = 70
smallest = min_dists.min()
mean_sep = min_dists.mean()

from sim2real.plots import add_germany_lines, save_plot

import cartopy.crs as ccrs

proj = ccrs.TransverseMercator(central_longitude=10, approx=False)

fig = plt.figure(figsize=(6, 1.5))
ax = fig.add_axes([0, 0, 0.7, 1])
gax = fig.add_axes([0.8, 0, 0.2, 1], projection=proj)

ax.set_ylim(ymin, ymax)
ax.vlines(
    mean_sep, 0, 100, linestyles="--", color="C1", label=f"Mean: {mean_sep:.2f} km"
)
ax.vlines(
    smallest, 0, 100, linestyles="--", color="C2", label=f"Min: {smallest:.2f} km"
)
ax.hist(min_dists, bins=30)
ax.legend()
ax.set_xlabel("Closest station distance [km]")
ax.set_ylabel("Station Count")

splits = load_station_splits()
value_stations = splits[splits["SET"] == "TEST"].index
dwd.plot_stations(value_stations, ax=gax, markersize=2**3)
add_germany_lines(gax)
gax.set_xlabel("asd")
ax.text(
    0.5,
    -0.2,
    "VALUE Stations",
    va="bottom",
    ha="center",
    rotation="horizontal",
    rotation_mode="anchor",
    transform=gax.transAxes,
)
save_plot(None, "station distance and value")
# %%
from sim2real.plots import save_plot, adjust_plot

# Forgetting
forg_dir = "../data/processed/wandb/forgetting"

nums_stations = [20, 100, 500]
num_stations = nums_stations[0]
i = 0
real_col = "C0"
e5_col = "C1"

val_linestyle = "-"
test_linestyle = "--"

fig, axs = plt.subplots(1, 3, figsize=(8, 3))
for i, num_stations in enumerate(nums_stations):
    fpath = f"{forg_dir}/{num_stations}_stations.csv"
    df = pd.read_csv(fpath)

    e5_test = [col for col in df.columns if "0.05" in col and "test" in col][0]
    e5_val = [col for col in df.columns if "0.05" in col and "val" in col][0]
    real_test = [
        col for col in df.columns if "TunerType" in col and "test_loss" in col
    ][0]
    real_val = [col for col in df.columns if "TunerType" in col and "val_loss" in col][
        0
    ]

    e5_test = df[e5_test]
    e5_val = df[e5_val]
    real_test = df[real_test]
    real_val = df[real_val]

    ymin = min(real_val) - 0.3
    ymax = max(e5_test[1:]) * 1.1

    axs[i].plot(
        e5_test, label="Test $\\omega = 0.05$", color=e5_col, linestyle=test_linestyle
    )
    axs[i].plot(
        e5_val, label="Val $\\omega = 0.05$", color=e5_col, linestyle=val_linestyle
    )

    axs[i].plot(
        real_test,
        label="Test $\\omega = 0.05$",
        color=real_col,
        linestyle=test_linestyle,
    )
    axs[i].plot(
        real_val, label="Val $\\omega = 0.05$", color=real_col, linestyle=val_linestyle
    )

    axs[i].set_ylim(ymin, ymax)
    axs[i].set_title(f"$N_{{stations}} = {num_stations}$")
    axs[i].set_xlabel("Epoch")

axs[0].legend(
    ncol=4, bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure, loc="center"
)
axs[0].set_ylabel("Negative Log-Likelihood, $\mathcal{L}$")
adjust_plot(fig, axs)
save_plot(None, "catastrophic_forgetting", fig=fig)
# %%
