from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
from pathlib import Path
import cartopy.crs as ccrs


@dataclass
class Paths:
    root: str
    raw_dwd: str
    dwd: str
    dwd_meta: str
    dwd_test_stations: str
    raw_era5: str
    era5: str
    raw_srtm: str
    srtm: str
    station_split: str
    time_split: str
    out: str
    test_results: str


@dataclass
class Bounds:
    lat: Tuple[float, float]
    lon: Tuple[float, float]


@dataclass
class DataSpec:
    bounds: Bounds
    crs_str: str
    epsg: int
    train_dates: Tuple[str, str]
    cv_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    val_freq: str
    era5_context: Tuple[int, int]
    era5_target: int
    dwd_context: Tuple[int, int]
    dwd_target: int
    norm_params: Dict


@dataclass
class ModelSpec:
    unet_channels: Tuple
    film: bool
    freeze_film: bool
    likelihood: str
    ppu: int
    dim_yt: int
    dim_yc: Tuple[int]
    encoder_scales: List[float]
    decoder_scale: float
    encoder_scales_learnable: bool
    decoder_scale_learnable: bool


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
    train: str
    val: str
    test: str
    set: str
    order: str
    epoch: str
    train_loss: str
    val_loss: str
    val_temporal_loss: str
    val_spatial_loss: str


@dataclass
class OutputSpec:
    wandb: bool
    plots: bool
    fig_crs: ccrs._CylindricalProjection
    data_crs: ccrs._CylindricalProjection
    sample_dates: list
    wandb_name: str = None
    spatiotemp_vals: bool


@dataclass
class OptimSpec:
    seed: int
    device: str
    batch_size: int
    batch_size_val: int
    batches_per_epoch: int
    num_epochs: int
    lr: float
    start_from: str
    early_stop_patience: int
    scheduler_patience: int
    scheduler_factor: float


class TunerType(Enum):
    naive = 0
    film = 1
    long_range = 2
    none = 3


@dataclass
class TuneSpec:
    tuner: TunerType
    num_stations: int
    num_tasks: int
    val_frac_stations: float
    val_frac_times: float
    split: bool
    frac_power: int
    frequency_level: int
    no_pretraining: bool


# Inferred from ERA5 data and pasted here.
norm_params = {
    "coords": {
        "time": {"name": "TIME"},
        "x1": {"name": "LAT", "map": (47.2, 54.95)},
        "x2": {"name": "LON", "map": (5.8, 15.05)},
    },
    "T2M": {"mean": 9.695045471191406, "std": 7.660834312438965},
    "HEIGHT": {"mean": 263.4660561588103, "std": 345.67930603408615},
}


names = Names(
    temp="T2M",
    lat="LAT",
    lon="LON",
    height="HEIGHT",
    station_name="STATION_NAME",
    station_id="STATION_ID",
    time="TIME",
    train="TRAIN",
    val="VAL",
    test="TEST",
    set="SET",
    order="ORDER",
    val_loss="val_loss",
    val_spatial_loss="val_spatial_loss",
    val_temporal_loss="val_temporal_loss",
    train_loss="train_loss",
    epoch="epoch",
)

root = str(Path(__file__).parent.parent.resolve())

paths = Paths(
    root=root,
    raw_dwd=f"{root}/data/raw/dwd/airtemp2m/unzipped",
    dwd=f"{root}/data/processed/dwd/airtemp2m/dwd.feather",
    dwd_meta=f"{root}/data/processed/dwd/airtemp2m/dwd_meta.feather",
    dwd_test_stations=f"{root}/data/processed/dwd/value_stations.feather",
    raw_era5=f"{root}/data/raw/ERA_5_Germany/1.grib",
    era5=f"{root}/data/processed/era5/era5_small.nc",
    raw_srtm=f"{root}/data/raw/srtm_dem/srtm_germany_dtm.tif",
    srtm=f"{root}/data/processed/srtm_dem/srtm_germany_dtm.nc",
    station_split=f"{root}/data/processed/splits/stations.feather",
    time_split=f"{root}/data/processed/splits/times.feather",
    out=f"{root}/_outputs",
    test_results=f"{root}/_outputs/test_results.csv",
)

data = DataSpec(
    bounds=Bounds(lat=(47.2, 54.95), lon=(5.8, 15.05)),
    crs_str="epsg:4326",
    epsg=4326,
    train_dates=("2012-01-01", "2020-12-31"),
    cv_dates=("2021-01-01", "2021-12-31"),
    test_dates=("2022-01-01", "2022-12-31"),
    # This should be set in a way that ensures all times
    # of day are covered.
    val_freq="39H",
    era5_context=(0, 500),
    era5_target="all",
    dwd_context=(0.0, 1.0),
    dwd_target="all",
    norm_params=norm_params,
)

pretrain_opt = OptimSpec(
    seed=42,
    device="cuda",
    batch_size=16,
    batch_size_val=512,
    batches_per_epoch=200,
    num_epochs=200,
    lr=1e-4,
    start_from=None,  # None, "best", "latest"
    scheduler_patience=8,
    early_stop_patience=20,
    scheduler_factor=1 / 3,
)

tune_opt = OptimSpec(
    seed=42,
    device="cuda",
    batch_size=16,
    batch_size_val=512,
    batches_per_epoch=25,
    num_epochs=100,
    lr=3e-5,
    start_from=None,  # None, "best", "latest"
    scheduler_patience=3,
    early_stop_patience=5,
    scheduler_factor=1 / 3,
)

opt = pretrain_opt

ppu = 200  # Found from dwd.compute_ppu()
model = ModelSpec(
    unet_channels=(96,) * 6,
    dim_yt=1,
    dim_yc=(1, 7),
    ppu=ppu,
    film=True,
    freeze_film=True,
    likelihood="het",
    encoder_scales=[1 / ppu, 1 / ppu],
    decoder_scale=1 / ppu,
    encoder_scales_learnable=False,
    decoder_scale_learnable=False,
)

out = OutputSpec(
    wandb=True,
    plots=True,
    wandb_name=None,
    fig_crs=ccrs.TransverseMercator(central_longitude=10, approx=False),
    data_crs=ccrs.PlateCarree(),
    # Must be part of test dates.
    sample_dates=["2022-03-01 08:00:00", "2022-01-02 04:00:00"],
    spatiotemp_vals=False,
)

tune = TuneSpec(
    tuner=TunerType.naive,
    num_stations=500,
    num_tasks=400,
    val_frac_stations=0.2,
    val_frac_times=0.2,
    split=True,
    # How much should sparse tasks be preferred? Larger => more sparse tasks.
    frac_power=2,
    frequency_level=4,
    no_pretraining=True,
)
