# %%
from sim2real.datasets import DWDSTationData, ECADStationData, load_era5
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


class CountryPlot:
    def __init__(
        self,
        shapefile_path: str = None,
        era5: xr.Dataset = None,
        dwd_data: DWDSTationData = None,
        ecad_data: ECADStationData = None,
    ) -> None:
        self.vmin = -5
        self.vmax = 30
        self.vmin, self.vmax = None, None
        self.cm = "coolwarm"

        self.dwd = None
        self.era5 = None

        if shapefile_path is not None:
            self.country = gpd.read_file(shapefile_path)
            self.country.crs = "epsg:4326"

        if era5 is not None:
            self.era5 = era5

        if dwd_data is not None:
            self.dwd = dwd_data
            self.dwd.crs = "epsg:4326"

        if ecad_data is not None:
            # TODO
            raise NotImplementedError

    def plot(self, datetime):
        datetime = pd.to_datetime(datetime)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        if self.era5 is not None:
            era5_t = self.era5["t2m"].sel(time=datetime) - 273.15
            era5_t.plot(ax=ax, vmin=self.vmin, vmax=self.vmax, cmap=self.cm)

        if self.dwd is not None:
            gdf = self.dwd.at_datetime(datetime)
            gdf.plot(
                ax=ax,
                column=names.temp,
                edgecolors="gray",
                cmap=self.cm,
                vmin=self.vmin,
                vmax=self.vmax,
            )

        if self.country is not None:
            self.country.plot(edgecolor="black", ax=ax, alpha=1, facecolor="none")

        self.fig = fig
        self.ax = ax

        return fig, ax


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


def init_fig(nrows=1, ncols=1, figsize=(4, 4)):
    """
    Generate a figure configured with a basic map of germany.
    """
    proj = ccrs.TransverseMercator(central_longitude=10, approx=False)
    fig, axs = plt.subplots(
        subplot_kw={"projection": proj}, nrows=nrows, ncols=ncols, figsize=figsize
    )
    bounds = [*data.bounds.lon, *data.bounds.lat]

    for ax in np.array(axs).flat:
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        ax.add_feature(feature.BORDERS, linewidth=0.25)
        ax.coastlines(linewidth=0.25)

    return fig, axs


if __name__ == "__main__":
    init_fig()
    from sim2real.config import paths

    dwd_sd = DWDSTationData(paths)
    era5 = load_era5()

    datetime = "2022-04-25 14:00:00"
    gdf = dwd_sd.at_datetime(datetime)

    vmin = 4
    vmax = 18

    proj = ccrs.TransverseMercator(central_longitude=10, approx=False)
    fig, axs = plt.subplots(
        subplot_kw={"projection": proj}, nrows=2, ncols=1, figsize=(5, 10)
    )
    bounds = [*data.bounds.lon, *data.bounds.lat]

    gdf = gdf.sample(frac=1, random_state=4)

    gdf.crs = proj

    gdf.plot(
        column=names.temp,
        ax=axs[0],
        cmap="seismic",
        transform=ccrs.PlateCarree(),
        edgecolors="black",
        linewidth=0.25,
        legend=False,
        vmin=vmin,
        vmax=vmax,
    )

    sel = {names.time: pd.to_datetime(datetime)}

    era5[names.temp].sel(sel).plot(
        ax=axs[1], transform=ccrs.PlateCarree(), cmap="seismic", vmin=vmin, vmax=vmax
    )

    for ax in axs.flat():
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        ax.add_feature(feature.BORDERS, linewidth=0.25)
        ax.coastlines(linewidth=0.25)

    axs[1].set_title("")
    save_plot(None, "real_and_era_5")
