from dataclasses import dataclass
from typing import Tuple
from pathlib import Path


@dataclass
class Paths:
    root: str
    raw_dwd: str
    dwd: str
    dwd_meta: str
    raw_era5: str
    era5: str
    raw_srtm: str
    srtm: str


@dataclass
class Bounds:
    lat: Tuple[float, float]
    lon: Tuple[float, float]


@dataclass
class DataSpec:
    bounds: Bounds
    crs_str: str
    epsg: int


@dataclass
class Names:
    """
    Consistent names in e.g. dataframes.
    """

    temp: str
    lat: str
    lon: str
    height: str
    station_name: str
    station_id: str
    time: str


@dataclass
class OutputSpec:
    wandb: bool


names = Names(
    temp="T2M",
    lat="LAT",
    lon="LON",
    height="HEIGHT",
    station_name="STATION_NAME",
    station_id="STATION_ID",
    time="TIME",
)

root = str(Path(__file__).resolve().parent.parent)

paths = Paths(
    root=root,
    raw_dwd=f"{root}/data/raw/dwd/airtemp2m/unzipped",
    dwd=f"{root}/data/processed/dwd/airtemp2m/dwd.feather",
    dwd_meta=f"{root}/data/processed/dwd/airtemp2m/dwd_meta.feather",
    raw_era5=f"{root}/data/raw/ERA_5_Germany/1.grib",
    era5=f"{root}/data/processed/era5/era5_small.nc",
    raw_srtm=f"{root}/data/raw/srtm_dem/srtm_germany_dtm.tif",
    srtm=f"{root}/data/processed/srtm_dem/srtm_germany_dtm.nc",
)

data = DataSpec(
    bounds=Bounds(lat=(47.2, 54.95), lon=(5.8, 15.05)),
    crs_str="epsg:4326",
    epsg=4326,
)

out = OutputSpec(wandb=True)
