from dataclasses import dataclass


@dataclass
class Paths:
    raw_dwd: str
    dwd: str
    dwd_meta: str
    era5: str


paths = Paths(
    raw_dwd="./data/raw/dwd/airtemp2m/unzipped",
    dwd="./data/processed/dwd/airtemp2m/dwd.feather",
    dwd_meta="./data/processed/dwd/airtemp2m/dwd_meta.feather",
    # era5="./data/raw/ERA_5_Germany/1.grib",
    era5="./data/processed/era5/era5_small.nc",
)
