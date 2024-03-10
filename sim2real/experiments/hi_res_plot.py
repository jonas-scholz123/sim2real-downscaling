# %%
import copy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import replace
from itertools import product
import torch
from torch.utils.data import DataLoader

import deepsensor.torch
from deepsensor.model.convnp import ConvNP
from deepsensor.model.nps import convert_task_to_nps_args
from deepsensor.plot import offgrid_context

import lab as B

from neuralprocesses.numdata import num_data
from neuralprocesses.dist import MultiOutputNormal

from tqdm import tqdm
from sim2real.datasets import load_elevation, load_station_splits, load_time_splits
from sim2real.plots import init_fig, save_plot
from sim2real.test import Evaluator
from sim2real.train.tune import Sim2RealTrainer
from sim2real.config import (
    DataSpec,
    ModelSpec,
    OptimSpec,
    OutputSpec,
    Paths,
    TuneSpec,
    TunerType,
    opt,
    out,
    model,
    data,
    names,
    paths,
    tune,
)
from sim2real.utils import (
    ensure_dir_exists,
    exp_dir_sim2real,
    exp_dir_sim,
    load_weights,
    sample_dates,
    sample_stations,
    weight_dir,
)

from sim2real.plots import adjust_plot

import shapely.vectorized
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import xarray as xr
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev, MeanStddev
import cartopy.feature as cf
import geopandas as gpd

# %%
opt = replace(opt, start_from="best")
trainer = Sim2RealTrainer(paths, opt, out, data, model, tune)
# %%
task = trainer.train_set[3]
task
# %%
trainer.plot_example_task(task)
# %%
# %%
aux = load_elevation()
aux = aux.coarsen(LAT=5, LON=5, boundary="trim").mean()
prediction = trainer.model.predict(task, aux)["T2M"]
prediction["mean"].plot()
# %%
aux["HEIGHT"].plot()
# %%
# %%
trainer.plot_prediction(task)
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

    fig, axes = plt.subplots(
        1, ncols, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=figsize
    )

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
        deepsensor.plot.offgrid_context(
            axes, task, trainer.data_processor, trainer.task_loader
        )
    print("Done")
    return fig, axes


# hires_aux_raw_ds = load_elevation()

test_date = trainer.val_set.times[0]
for context_sampling in [20, 100, "all"]:
    task = trainer.task_loader(test_date, [context_sampling, "all"])
    prediction = trainer.model.predict(
        task, X_t=aux.sel(LAT=slice(55, 47.5), LON=slice(6, 15))
    )["T2M"]
    mean_ds, std_ds = prediction["mean"], prediction["std"]

    fig, axes = gen_test_fig(
        # era5_raw_ds.sel(time=test_task['time'], lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())),
        None,
        mean_ds.coarsen(LAT=5, LON=5, boundary="trim").mean(),
        std_ds.coarsen(LAT=5, LON=5, boundary="trim").mean(),
        task=task,
        add_colorbar=True,
        var_cbar_label="2m temperature [°C]",
        std_cbar_label="std dev [°C]",
        std_clim=(1, 3),
        var_clim=(-4.0, 10.0),
        extent=(6, 15, 47.5, 55),
        figsize=(20, 20 / 3),
    )
    # fname = f"downscale_{context_sampling}"
    # fig.savefig(os.path.join(fig_folder, fname + ".png"), bbox_inches="tight")
    # fig.savefig(os.path.join(fig_folder, fname + ".pdf"), bbox_inches="tight", dpi=300)
# %%
test_date
# %%
trainer.task_loader
# %%
