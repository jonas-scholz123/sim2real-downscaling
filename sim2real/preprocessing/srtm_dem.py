# %%
from pathlib import Path
import sys
from sim2real.config import paths, names
from sim2real.utils import ensure_dir_exists
import xarray as xr
import pandas as pd


def process_srtm():
    print("Loading raw SRTM dataset...")
    elevation = xr.open_dataset(paths.raw_srtm, engine="rasterio")
    elevation = elevation.rename(
        {
            "x": names.lon,
            "y": names.lat,
        }
    )
    elevation = elevation.sel(band=1).drop("band")
    elevation = elevation.rename({"band_data": names.height})
    ensure_dir_exists(paths.srtm)
    elevation.to_netcdf(paths.srtm)


if __name__ == "__main__":
    process_srtm()
