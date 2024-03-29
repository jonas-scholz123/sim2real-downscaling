# %%
from typing import Tuple
import pandas as pd
import os
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm

from enum import Enum

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import xarray as xr
import numpy as np
from scipy.spatial.distance import cdist

from sim2real.config import Paths, paths, names, data, out
from sim2real.gridder import Gridder
from sim2real.utils import get_default_data_processor
import sim2real.plots as plots


class QualityCode(Enum):
    valid = 0
    suspect = 1
    missing = 9


def convert_latlon(latlon):
    multiplier = 1 if latlon[0] == "+" else -1
    hr, minute, sec = latlon.split(":")

    result = int(hr) + int(minute) / 60 + int(sec) / 3600
    return multiplier * result


class ECADStationData:
    """
    Wraps the station data of the ECA&D Dataset.
    """

    def __init__(
        self, raw_path, start_date, end_date, filter_suspect=True, country_filter="DE"
    ) -> None:
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.filter_missing = True
        self.filter_suspect = filter_suspect
        self.country_filter = country_filter
        self.meta_df = self._load_metadata(raw_path)
        self.df = self._load_data(raw_path)

    def _load_data(self, raw_path):
        dfs = []
        print(f"Reading data from {raw_path}")

        valid_stations = set(self.meta_df["STAID"])
        for fname in tqdm(os.listdir(raw_path)):
            if not fname.startswith("TG"):
                # Don't want meta data.
                continue

            # Get the station ID from the filename and check if we want to consider it.
            if int(fname.split(".")[0].split("STAID")[-1]) not in valid_stations:
                continue

            fpath = f"{raw_path}/{fname}"
            try:
                # UTF-8 fails on some stations. Use latin-1.
                df = pd.read_csv(fpath, header=15, encoding="latin-1")
            except Exception as e:
                print(f"WARNING: Failed to read {fpath} due to {e}")
                continue
            df = self._sanitise_df(df)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _load_metadata(self, raw_path):
        meta = pd.read_csv(f"{raw_path}/stations.txt", header=13, encoding="latin-1")
        meta.columns = ["STAID", "STANAME", "CN", "LAT", "LON", "HGHT"]
        meta["STANAME"] = meta["STANAME"].str.strip()
        meta["CN"] = meta["CN"].str.strip()
        meta["LAT"] = meta["LAT"].apply(convert_latlon)
        meta["LON"] = meta["LON"].apply(convert_latlon)

        if self.country_filter is not None:
            meta = meta[meta["CN"] == self.country_filter]
        return meta

    def _sanitise_df(self, df: pd.DataFrame):
        # Sanitise column headers.
        df.columns = ["STAID", "SOUID", "DATE", "TEMP", "QUALITY"]

        # Sanitise quality code.
        df["QUALITY"] = df["QUALITY"].apply(lambda x: QualityCode(x))

        # Filter date.
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
        df = df[(df["DATE"] >= self.start_date) & (df["DATE"] <= self.end_date)].copy()

        if self.filter_missing:
            df = df[df["QUALITY"] != QualityCode.missing]
        if self.filter_suspect:
            df = df[df["QUALITY"] != QualityCode.suspect]

        # Convert temperature to deg C.
        df["TEMP"] /= 10
        return df

    def at_datetime(self, date):
        date = pd.to_datetime(date)
        entries = self.df[self.df["DATE"] == date]
        entries = entries.merge(self.meta_df, on="STAID")
        return entries


class DWDStationData:
    def __init__(self, paths: Paths, df=None, meta_df=None) -> None:
        if df is None or meta_df is None:
            self.df, self.meta_df = self._load_data(paths)
        else:
            self.df, self.meta_df = df, meta_df

    def _load_data(self, paths: Paths):
        try:
            # If cached, load and return.
            meta_df = gpd.read_feather(paths.dwd_meta)
            df = pd.read_feather(paths.dwd)
            df = df.set_index([names.time, names.station_id])

            assert meta_df.crs == data.crs_str

            return df, meta_df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"""DWD Station data is not found at {paths.dwd}
                or {paths.dwd_meta}. Make sure to run preprocessing/dwd.py first."""
            )

    def _to_gdf(self, df):
        gdf = gpd.GeoDataFrame(df)
        gdf.crs = data.crs_str
        return gdf

    def train_val_test_split(self, val_frac=0.2, seed=42) -> Tuple:
        # TODO: Redo.
        v = gpd.read_feather(paths.dwd_test_stations)

        # Choose nearest stations to VALUE test stations.
        eval_station_ids = set(v.sjoin_nearest(self.meta_df)[names.station_id])
        query = f"{names.station_id} in @eval_station_ids"
        test_df = self.df.query(query)
        test_meta_df = self.meta_df.query(query)

        # remainder:
        np.random.seed(seed)
        meta_remainder = self.meta_df.query(
            f"{names.station_id} not in @eval_station_ids"
        )

        station_ids = meta_remainder["STATION_ID"].unique()
        val_station_ids = np.random.choice(
            station_ids, int(val_frac * len(station_ids))
        )

        val_df = self.df.query(f"{names.station_id} in @val_station_ids")
        val_meta_df = meta_remainder.query(f"{names.station_id} in @val_station_ids")

        train_station_ids = set(station_ids) - set(val_station_ids)
        train_df = self.df.query(f"{names.station_id} in @train_station_ids")

        train_meta_df = meta_remainder.query(
            f"{names.station_id} in @train_station_ids"
        )

        train = DWDStationData(None, train_df, train_meta_df)
        val = DWDStationData(None, val_df, val_meta_df)
        test = DWDStationData(None, test_df, test_meta_df)

        return train, val, test

    def at_datetime(self, dt):
        dt = pd.to_datetime(dt)
        df = self.df.query(f"{names.time} == @dt")

        # Select relevant metadata.
        meta = self.meta_df.query("FROM_DATE < @dt & TO_DATE >= @dt")

        # df = df.merge(meta, on=names.station_id)
        df = meta.merge(df, on=names.station_id)
        df = df.drop(["FROM_DATE", "TO_DATE"], axis=1)
        df = df.reset_index(drop=True)
        # df = self._to_gdf(df)
        df[names.time] = dt
        df = df.set_index([names.time, names.station_id])
        return df

    def at_datetimes(self, dts):
        dts = set(dts)
        df = self.df.query(f"{names.time} in @dts")
        df = df.reset_index()
        df = self.meta_df.merge(df, on=names.station_id)
        df = df.set_index([names.time, names.station_id])
        df = df.query(f"FROM_DATE < {names.time} & TO_DATE >= {names.time}")

        # Drop all entries where the meta data info doesn't match the
        # real data info.
        # df = self._to_gdf(df)
        df = df.drop(["FROM_DATE", "TO_DATE"], axis=1)
        return df

    def between_datetimes(self, start, end, freq="H"):
        dts = pd.date_range(start, end, freq=freq)
        return self.at_datetimes(dts)

    def apply_grid(self, gridder: Gridder):
        self.meta_df = gridder.grid_latlons(self.meta_df)

    def full(self):
        """
        Return a dataframe with all of the data.
        """
        dts = self.df.index.get_level_values(0)
        min_dt, max_dt = dts.min(), dts.max()
        return self.between_datetimes(min_dt, max_dt)

    def to_deepsensor_df(self):
        """
        Returns a dataframe indexed by [time, lat, lon] with only the temperature column T2M.
        """
        df = self.full()
        return df.reset_index().set_index([names.time, names.lat, names.lon])[
            [names.temp]
        ]

    def compute_ppu(self):
        """
        Compute the appropriate PPU to capture the shortest length-scale
        interactions.
        """

        def smallest_distance(group):
            if len(group) < 5:
                return 1
            coords = group.reset_index().drop(["time", "T2M"], axis=1)
            distances = cdist(coords, coords, metric="euclid")
            np.fill_diagonal(distances, np.inf)
            smallest_distance = np.quantile(distances, 0.00)
            return smallest_distance

        df = self.to_deepsensor_df()
        dp = get_default_data_processor()

        # normalise lats/lons to [0, 1] range.
        df = dp(df)

        # select only a subset of available times. Don't need to check
        # all, because in those 100 there will be one time that features
        # the maximum number of stations.
        dts = np.random.choice(df.reset_index()["time"].unique(), 100)
        df = df.query("time in @dts")
        grouped = df.groupby("time")
        dists = grouped.apply(smallest_distance)
        ppu = 1 / dists.min()
        return ppu

    def plot_stations(self, station_ids, marker="o", color="C0", ax=None, **kwargs):
        if ax is None:
            _, axs = plots.init_fig()
            ax = axs[0]

        gdf = self.meta_df.query(f"{names.station_id} in @station_ids")
        gdf = gdf.groupby(names.station_id).last()
        gdf.plot(
            ax=ax,
            transform=out.data_crs,
            marker=marker,
            facecolor="none",
            color=color,
            **kwargs,
        )


def load_era5():
    if paths.era5.endswith("nc"):
        return xr.load_dataset(paths.era5)
    else:
        return xr.load_dataset(paths.era5, engine="cfgrib")


def load_elevation():
    return xr.load_dataset(paths.srtm)


def load_dwd_eval():
    return pd.read_feather(paths.dwd_test_stations)


def load_station_splits():
    df = pd.read_feather(paths.station_split).set_index(names.station_id, drop=True)
    return df


def load_time_splits():
    df = pd.read_feather(paths.time_split).set_index(names.time, drop=True)
    return df


if __name__ == "__main__":
    dwd = DWDStationData(paths)
