# %%
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


# %%
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


# %%
class DWDSTationData:
    def __init__(self, raw_path, start_date, end_date) -> None:
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.df, self.meta_df = self._load_data(raw_path)

    def _load_data(self, root):
        dfs = []
        meta_dfs = []
        for subdir in tqdm(os.listdir(root)):
            dir_path = f"{root}/{subdir}"
            if not os.path.isdir(dir_path):
                continue

            # Data:
            fname = [n for n in os.listdir(dir_path) if n.startswith("produkt")][0]
            fpath = f"{root}/{subdir}/{fname}"
            dfs.append(self._load_station_df(fpath))

            # Metadata:
            fname = [n for n in os.listdir(dir_path) if n.startswith("Metadaten_Geo")][
                0
            ]
            fpath = f"{root}/{subdir}/{fname}"
            meta_dfs.append(self._load_station_metadata(fpath))

        return pd.concat(dfs, ignore_index=True), pd.concat(meta_dfs, ignore_index=True)

    def _load_station_df(self, fpath):
        df = pd.read_csv(fpath, delimiter=";", encoding="latin-1")
        df.columns = ["STATION_ID", "DATETIME", "QN_9", "TEMP", "RF_TU", "eor"]
        # drop relative humidity, "end of record" column.
        df = df.drop(["RF_TU", "eor"], axis=1)
        # Filter date.
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%Y%m%d%H")
        df = df[
            (df["DATETIME"] >= self.start_date) & (df["DATETIME"] <= self.end_date)
        ].copy()

        # Filter invalid values.
        df = df[df["TEMP"] != -999.0]
        return df

    def _load_station_metadata(self, fpath):
        meta_columns = [
            "STATION_ID",
            "HEIGHT",
            "LAT",
            "LON",
            "FROM_DATE",
            "TO_DATE",
            "STATION_NAME",
        ]
        df = pd.read_csv(fpath, delimiter=";", encoding="latin-1")
        df.columns = meta_columns

        df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"], format="%Y%m%d")

        # Last "TO_DATE" is empty because it represents current location.
        df["TO_DATE"] = pd.to_datetime(df["TO_DATE"], format="%Y%m%d", errors="coerce")
        df.loc[df.index[-1], "TO_DATE"] = pd.to_datetime("today").normalize()

        df = df[(df["TO_DATE"] >= self.start_date) & (df["FROM_DATE"] <= self.end_date)]
        return df

    def at_datetime(self, dt):
        dt = pd.to_datetime(dt)
        entries = self.df[self.df["DATETIME"] == dt]

        # Select relevant metadata.
        meta = self.meta_df[
            (self.meta_df["FROM_DATE"] < dt) & (self.meta_df["TO_DATE"] >= dt)
        ]
        entries = entries.merge(meta, on="STATION_ID")

        geometry = gpd.points_from_xy(entries["LON"], entries["LAT"])
        gdf = GeoDataFrame(entries, geometry=geometry)
        return gdf