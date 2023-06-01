from dataclasses import dataclass
from typing import Tuple


@dataclass
class Paths:
    raw_dwd: str
    dwd: str
    dwd_meta: str
    raw_era5: str
    era5: str


@dataclass
class Bounds:
    low: Tuple[float, float]
    high: Tuple[float, float]


@dataclass
class DataSpec:
    bounds: Bounds
    crs: str


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


names = Names(
    temp="T2M",
    lat="LAT",
    lon="LON",
    height="HEIGHT",
    station_name="STATION_NAME",
    station_id="STATION_ID",
    time="TIME",
)

paths = Paths(
    raw_dwd="./data/raw/dwd/airtemp2m/unzipped",
    dwd="./data/processed/dwd/airtemp2m/dwd.feather",
    dwd_meta="./data/processed/dwd/airtemp2m/dwd_meta.feather",
    raw_era5="./data/raw/ERA_5_Germany/1.grib",
    era5="./data/processed/era5/era5_small.nc",
)

data = DataSpec(
    bounds=Bounds(
        low=(47.2, 5.8),
        high=(54.95, 15.05),
    ),
    crs="epsg:4326",
)
