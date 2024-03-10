# %%
import logging

logging.captureWarnings(True)
# %%
import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev

from deepsensor.train.train import train_epoch, set_gpu_default_device
from deepsensor.data.utils import construct_x1x2_ds

import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from sim2real.datasets import DWDStationData
from sim2real.config import paths

# %%
crs = ccrs.PlateCarree()

use_gpu = True
if use_gpu:
    set_gpu_default_device()

model_folder = "models/stationinterp/"
fig_folder = "figures/stationinterp/"
# %%
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)
# %%
hires_aux_raw_ds = (
    xr.open_mfdataset(
        "/home/jonas/Documents/code/sim2real-downscaling/data/processed/srtm_dem/srtm_germany_dtm.nc"
    )
    .rename({"LAT": "lat", "LON": "lon"})
    .coarsen(lat=5, lon=5, boundary="trim")
    .mean()
    .load()
)
hires_aux_raw_ds = hires_aux_raw_ds
print(hires_aux_raw_ds)
# %%
# Compute Topographic Position Index from elevation data
import scipy

# Resolutions in coordinate values along the spatial row and column dimensions
#   Here we assume the elevation is on a regular grid, so the first difference
#   is equal to all others.
coord_names = list(hires_aux_raw_ds.dims)
resolutions = np.array(
    [np.abs(np.diff(hires_aux_raw_ds.coords[coord].values)[0]) for coord in coord_names]
)

for window_size in [0.1, 0.05, 0.025]:
    smoothed_elev_da = hires_aux_raw_ds["HEIGHT"].copy(deep=True)

    # Compute gaussian filter scale in terms of grid cells
    scales = window_size / resolutions

    smoothed_elev_da.data = scipy.ndimage.gaussian_filter(
        smoothed_elev_da.data, sigma=scales, mode="nearest"
    )

    TPI_da = hires_aux_raw_ds["HEIGHT"] - smoothed_elev_da
    hires_aux_raw_ds[f"TPI_{window_size}"] = TPI_da

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
    TPI_da.plot(ax=ax)
    ax.add_feature(cf.BORDERS)
    ax.coastlines()
# %%
print(hires_aux_raw_ds)
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
hires_aux_raw_ds["HEIGHT"].plot(ax=ax)
ax.add_feature(cf.BORDERS)
ax.coastlines()
fname = "hires_elevation.png"
fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")
# %%
aux_raw_ds = hires_aux_raw_ds.coarsen(lat=20, lon=20, boundary="trim").mean()["HEIGHT"]
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
aux_raw_ds.plot(ax=ax)
ax.add_feature(cf.BORDERS)
ax.coastlines()
fname = "lowres_elevation.png"
fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")
print(aux_raw_ds)
# %%
# Print resolution of lowres and hires elevation data
print(
    f"Lowres lat resolution: {np.abs(np.diff(aux_raw_ds.coords['lat'].values)[0]):.4f} degrees"
)
print(
    f"Lowres lon resolution: {np.abs(np.diff(aux_raw_ds.coords['lon'].values)[0]):.4f} degrees"
)
print(
    f"Hires lat resolution: {np.abs(np.diff(hires_aux_raw_ds.coords['lat'].values)[0]):.4f} degrees"
)
print(
    f"Hires lon resolution: {np.abs(np.diff(hires_aux_raw_ds.coords['lon'].values)[0]):.4f} degrees"
)
# %%
dwd = DWDStationData(paths)
# %%
raw_df = dwd.to_deepsensor_df()
raw_df.index.names = ["time", "lat", "lon"]
# times = raw_df.index.get_level_values("time").unique()
# selected_times = pd.Series(times).sample(frac=0.1)
# raw_df = raw_df.loc[selected_times]
# %%
data_processor = DataProcessor(
    x1_name="lat",
    x1_map=(aux_raw_ds["lat"].min(), aux_raw_ds["lat"].max()),
    x2_name="lon",
    x2_map=(aux_raw_ds["lon"].min(), aux_raw_ds["lon"].max()),
)
station_df = data_processor([raw_df])[0]
aux_ds, hires_aux_ds = data_processor([aux_raw_ds, hires_aux_raw_ds], method="min_max")
print(data_processor)
# %%
task_loader = TaskLoader(
    context=[station_df, aux_ds],
    target=station_df,
    aux_at_targets=hires_aux_ds,
    links=[(0, 0)],
)
print(task_loader)
# %%
model = ConvNP(
    data_processor,
    task_loader,
    unet_channels=(64,) * 4,
    likelihood="cnp",
)


# %%

times = station_df.index.get_level_values("time")
cutoff = times[int(len(times) * 0.8)]
train_df = station_df.loc[times < cutoff]
val_df = station_df.loc[times >= cutoff]
# %%
print("val frac = ", len(val_df) / len(station_df))


# %%
def generate_tasks(station_df):
    train_times = station_df.index.get_level_values("time").unique()
    for time in tqdm(train_times):
        split_frac = np.random.uniform(0.0, 1.0)
        yield task_loader(
            time,
            context_sampling=["split", "all"],
            target_sampling="split",
            split_frac=split_frac,
        )


train_tasks = list(generate_tasks(train_df))
val_tasks = list(generate_tasks(val_df))
print("train tasks: ", len(train_tasks))
print("val tasks: ", len(val_tasks))
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=crs))
ax.coastlines()
ax.add_feature(cf.BORDERS)
deepsensor.plot.offgrid_context(
    ax,
    train_tasks[0],
    data_processor,
    task_loader,
    plot_target=True,
    add_legend=True,
    linewidths=0.5,
)
plt.show()
fname = "train_stations.png"
fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")


# %%
def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        loss = float(model.loss_fn(task, normalise=True))
        val_losses.append(loss)
    return np.mean(val_losses)


n_epochs = 80
train_losses = []
val_losses = []

val_loss_best = np.inf

# %%
for epoch in tqdm(range(n_epochs)):
    batch_losses = train_epoch(model, train_tasks)
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    val_loss = compute_val_loss(model, val_tasks)
    val_losses.append(val_loss)
    print(f"{train_loss=}, {val_loss=}")

    if val_loss < val_loss_best:
        import torch
        import os

        val_loss_best = val_loss
        torch.save(model.model.state_dict(), model_folder + f"model.pt")

    # print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

data_processor.config
# %%
model.predict(val_tasks[0], X_t=hires_aux_ds)
# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
fname = "loss.png"
fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")
# %%
fontsize = 14

params = {
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "font.size": fontsize,
    "figure.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "sans-serif",
    "figure.facecolor": "w",
}

import matplotlib as mpl

mpl.rcParams.update(params)


# %%
def gen_test_fig(
    era5_raw_ds=None,
    mean_ds=None,
    std_ds=None,
    samples_ds=None,
    task=None,
    extent=None,
    add_colorbar=False,
    var_cmap="jet",
    var_clim=None,
    std_cmap="Greys",
    std_clim=None,
    var_cbar_label=None,
    std_cbar_label=None,
    fontsize=None,
    figsize=(15, 5),
):
    if var_clim is None and era5_raw_ds is not None and mean_ds is not None:
        vmin = np.array(min(era5_raw_ds.min(), mean_ds.min()))
        vmax = np.array(max(era5_raw_ds.max(), mean_ds.max()))
    elif var_clim is not None:
        vmin, vmax = var_clim
    else:
        vmin = None
        vmax = None

    if std_clim is None and std_ds is not None:
        std_vmin = np.array(std_ds.min())
        std_vmax = np.array(std_ds.max())
    elif std_clim is not None:
        std_vmin, std_vmax = std_clim
    else:
        std_vmin = None
        std_vmax = None

    ncols = 0
    if era5_raw_ds is not None:
        ncols += 1
    if mean_ds is not None:
        ncols += 1
    if std_ds is not None:
        ncols += 1
    if samples_ds is not None:
        ncols += samples_ds.shape[0]

    fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=crs), figsize=figsize)

    axis_i = 0
    if era5_raw_ds is not None:
        ax = axes[axis_i]
        # era5_raw_ds.sel(lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())).plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=False)
        era5_raw_ds.plot(
            ax=ax,
            cmap=var_cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=add_colorbar,
            cbar_kwargs=dict(label=var_cbar_label),
        )
        ax.set_title("ERA5", fontsize=fontsize)
        axis_i += 1

    if mean_ds is not None:
        ax = axes[axis_i]
        mean_ds.plot(
            ax=ax,
            cmap=var_cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=add_colorbar,
            cbar_kwargs=dict(label=var_cbar_label),
        )
        ax.set_title("ConvNP mean", fontsize=fontsize)
        axis_i += 1

    if samples_ds is not None:
        for i in range(samples_ds.shape[0]):
            ax = axes[axis_i]
            samples_ds.isel(sample=i).plot(
                ax=ax,
                cmap=var_cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=add_colorbar,
                cbar_kwargs=dict(label=var_cbar_label),
            )
            ax.set_title(f"ConvNP sample {i+1}", fontsize=fontsize)
            axis_i += 1

    if std_ds is not None:
        ax = axes[axis_i]
        std_ds.plot(
            ax=ax,
            cmap=std_cmap,
            add_colorbar=add_colorbar,
            vmin=std_vmin,
            vmax=std_vmax,
            cbar_kwargs=dict(label=std_cbar_label),
        )
        ax.set_title("ConvNP std dev", fontsize=fontsize)
        axis_i += 1

    for ax in axes:
        ax.add_feature(cf.BORDERS)
        ax.coastlines()
        if extent is not None:
            ax.set_extent(extent)
    if task is not None:
        deepsensor.plot.offgrid_context(axes, task, data_processor, task_loader)
    print("Done")
    return fig, axes


time = val_df.index.get_level_values("time").unique()[500]
# %%
for context_sampling in [20, 100, "all"]:
    test_task = task_loader(time, [context_sampling, "all"])
    prediction = model.predict(
        test_task, X_t=hires_aux_raw_ds.sel(lat=slice(55, 47.5), lon=slice(6, 15))
    )["T2M"]

    mean_ds, std_ds = prediction["mean"], prediction["std"]

    fig, axes = gen_test_fig(
        # era5_raw_ds.sel(time=test_task['time'], lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())),
        None,
        mean_ds.coarsen(lat=5, lon=5, boundary="trim").mean(),
        std_ds.coarsen(lat=5, lon=5, boundary="trim").mean(),
        task=test_task,
        add_colorbar=True,
        var_cbar_label="2m temperature [°C]",
        std_cbar_label="std dev [°C]",
        std_clim=(1, 3),
        var_clim=(-4.0, 10.0),
        extent=(6, 15, 47.5, 55),
        figsize=(20, 20 / 3),
    )
    fname = f"downscale_{context_sampling}"
    fig.savefig(os.path.join(fig_folder, fname + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_folder, fname + ".pdf"), bbox_inches="tight", dpi=300)
