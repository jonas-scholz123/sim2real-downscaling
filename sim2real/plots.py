# %%
from deepsensor import offgrid_context
from sim2real.utils import ensure_dir_exists
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as feature
from sim2real.config import paths, names, data

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def save_plot(exp_path, name, fig=None, ext="pdf", **kwargs):
    if exp_path is None:
        path = f"{paths.out}/figures/{name}.{ext}"
    else:
        path = f"{exp_path}/plots/{name}.{ext}"
    ensure_dir_exists(path)
    if fig is None:
        plt.savefig(path, bbox_inches="tight", **kwargs)
    else:
        fig.savefig(path, bbox_inches="tight", **kwargs)


def plot_geopandas(
    gdf,
    column=names.temp,
    fig_ax=None,
    legend=True,
    vmin=None,
    vmax=None,
    **kwargs,
):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    country = gpd.read_file("../data/shapefiles/DEU_adm0.shp")
    country.crs = "epsg:4326"
    gdf.plot(column=column, ax=ax, legend=legend, vmin=vmin, vmax=vmax, **kwargs)
    country.plot(edgecolor="black", ax=ax, alpha=1, facecolor="none")
    return fig, ax


def timeline_plot(dts, label=None, fig=None, ax=None, xlim=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 1))
    ax.scatter(dts, [1 for _ in dts], s=100, marker="|", label=label, **kwargs)

    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(0.5, 2)
    ax.legend(ncol=3)
    ax.set_yticks([])
    fig.autofmt_xdate()
    return fig, ax


def init_fig(nrows=1, ncols=1, figsize=(4, 4), ret_transform=False):
    """
    Generate a figure configured with a basic map of germany.
    """
    proj = ccrs.TransverseMercator(central_longitude=10, approx=False)
    fig, axs = plt.subplots(
        subplot_kw={"projection": proj}, nrows=nrows, ncols=ncols, figsize=figsize
    )

    axs = np.array(axs).flatten()

    for ax in axs:
        add_germany_lines(ax)

    if ret_transform:
        return fig, axs, ccrs.PlateCarree()
    return fig, axs


def add_germany_lines(ax):
    bounds = [*data.bounds.lon, *data.bounds.lat]
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    ax.add_feature(feature.BORDERS, linewidth=0.25)
    ax.coastlines(linewidth=0.25)


def adjust_plot(fig=None, axs=None):
    if fig is None:
        fig = plt.gcf()
    if axs is None:
        axs = [plt.gca()]
    for ax in axs:
        ax.spines[["right", "top"]].set_visible(False)


def plot_era5_prediction(
    era5_data, mean_data, std_data, error_data, data_processor, task
):
    proj = ccrs.TransverseMercator(central_longitude=10, approx=False)
    fig, axs = plt.subplots(
        subplot_kw={"projection": proj}, nrows=1, ncols=4, figsize=(10, 2.5)
    )

    era5_plot = era5_data.plot(cmap="seismic", ax=axs[0], transform=ccrs.PlateCarree())
    cbar = era5_plot.colorbar
    vmin, vmax = cbar.vmin, cbar.vmax

    axs[0].set_title("ERA5")

    mean_data.plot(
        cmap="seismic",
        ax=axs[1],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_title("ConvNP mean")
    std_data.plot(cmap="Greys", ax=axs[2], transform=ccrs.PlateCarree())
    axs[2].set_title("ConvNP std dev")
    error_data.plot(cmap="seismic", ax=axs[3], transform=ccrs.PlateCarree())
    axs[3].set_title("ConvNP error")

    context_axs = [ax for i, ax in enumerate(axs) if i != 1]
    offgrid_context(
        context_axs,
        task,
        data_processor,
        s=3**2,
        linewidths=0.5,
        add_legend=False,
        transform=ccrs.PlateCarree(),
    )

    bounds = [*data.bounds.lon, *data.bounds.lat]
    for ax in axs:
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        ax.add_feature(feature.BORDERS, linewidth=0.25)
        ax.coastlines(linewidth=0.25)

    return fig
