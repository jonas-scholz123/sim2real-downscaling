# %%
from sim2real.utils import ensure_dir_exists
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as feature
from sim2real.config import paths, names, data


def save_plot(exp_path, name, fig=None):
    if exp_path is None:
        path = f"{paths.out}/figures/{name}.pdf"
    else:
        path = f"{exp_path}/plots/{name}.pdf"
    ensure_dir_exists(path)
    if fig is None:
        plt.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")


def plot_geopandas(
    gdf, column=names.temp, fig_ax=None, legend=True, vmin=None, vmax=None
):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    country = gpd.read_file("../data/shapefiles/DEU_adm0.shp")
    country.crs = "epsg:4326"
    gdf.plot(column=column, ax=ax, legend=legend, vmin=vmin, vmax=vmax)
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
    bounds = [*data.bounds.lon, *data.bounds.lat]

    axs = np.array(axs).flatten()

    for ax in axs:
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        ax.add_feature(feature.BORDERS, linewidth=0.25)
        ax.coastlines(linewidth=0.25)

    if ret_transform:
        return fig, axs, ccrs.PlateCarree()
    return fig, axs
