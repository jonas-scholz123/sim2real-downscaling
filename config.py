from dataclasses import dataclass


@dataclass
class Paths:
    raw_dwd: str
    dwd: str
    dwd_meta: str


paths = Paths(
    raw_dwd="./data/raw/dwd/airtemp2m/unzipped",
    dwd="./data/processed/dwd/airtemp2m/dwd.feather",
    dwd_meta="./data/processed/dwd/airtemp2m/dwd_meta.feather",
)
