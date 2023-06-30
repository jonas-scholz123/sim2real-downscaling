from dataclasses import dataclass
from enum import Enum
from typing import Tuple
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
    aux_coarsen_factor: float


@dataclass
class ModelSpec:
    unet_channels: Tuple
    film: bool
    freeze_film: bool
    likelihood: str
    ppu: int
    dim_yt: int
    dim_yc: Tuple[int]
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


@dataclass
class OutputSpec:
    wandb: bool
    plots: bool
    fig_crs: ccrs._CylindricalProjection
    data_crs: ccrs._CylindricalProjection
    wandb_name: str = None


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


@dataclass
class TuneSpec:
    tuner: TunerType
    num_stations: int


names = Names(
    temp="T2M",
    lat="LAT",
    lon="LON",
    height="HEIGHT",
    station_name="STATION_NAME",
    station_id="STATION_ID",
    time="TIME",
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
    era5_context=(5, 30),
    era5_target=100,
    dwd_context=(5, 50),
    dwd_target="all",
    # How much more dense should the elevation data
    # be compared to the era5 data? Smaller => more dense
    # aux data.
    aux_coarsen_factor=1,
)

opt = OptimSpec(
    seed=42,
    device="cpu",
    batch_size=16,
    batch_size_val=512,
    batches_per_epoch=10,
    num_epochs=200,
    lr=1e-4,
    start_from=None,  # None, "best", "latest"
    scheduler_patience=5,
    early_stop_patience=15,
    scheduler_factor=1 / 3,
)

model = ModelSpec(
    unet_channels=(128,) * 3,
    dim_yt=1,
    dim_yc=(1, 7),
    ppu=40,  # Found from dwd.compute_ppu()
    film=True,
    freeze_film=True,
    likelihood="het",
    encoder_scales_learnable=False,
    decoder_scale_learnable=False,
)

out = OutputSpec(
    wandb=False,
    plots=True,
    wandb_name=None,
    fig_crs=ccrs.TransverseMercator(central_longitude=10, approx=False),
    data_crs=ccrs.PlateCarree(),
)

tune = TuneSpec(tuner=TunerType.naive, num_stations=300)
