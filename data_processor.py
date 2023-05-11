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


class StationData:
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
        return pd.concat(dfs)

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

    # TODO: return cube here?
    def at_date(self, date):
        date = pd.to_datetime(date)
        entries = self.df[self.df["DATE"] == date]
        entries = entries.merge(self.meta_df, on="STAID")
        return entries


sd = StationData(
    "data/raw/ECA_blend_tg",
    "2018-01-01",
    "2019-01-01",
    country_filter="DE",
    filter_suspect=True,
)
# %%
germany = gpd.read_file("data/shapefiles/DEU_adm0.shp")
df = sd.at_date(f"2018-06-12")
geometry = gpd.points_from_xy(df["LON"], df["LAT"])
gdf = GeoDataFrame(df, geometry=geometry)
gdf.crs = "epsg:4326"
germany.crs = "epsg:4326"
# %%
# %%
import xarray as xr

ds = xr.open_dataset("data/raw/ERA_5_Germany/1.grib", engine="cfgrib")

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ds["t2m"].sel(time=pd.to_datetime("2018-06-12")).plot(ax=ax)
# this is a simple map that goes with geopandas
ax = germany.plot(edgecolor="red", ax=ax, alpha=1, facecolor="none")
gdf.plot(ax=ax, column="TEMP", legend=True)

# %%
