#%%
import pandas as pd
import xarray as xr
from bisect import bisect_left

import matplotlib.pyplot as plt

from datasets import DWDSTationData
from plots import plot_geopandas


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

class Gridder:
    def __init__(self, lats, lons):
        
        self.lat_grid = pd.unique(lats - lats.astype(int))

        
        self.lon_grid = pd.unique(lons - lons.astype(int))
    
    def closest(self, grid, val):
        int_val = int(val)
        remainder = val - int_val
        return round(int_val + min(grid, key=lambda x: abs(x - remainder)), 3)

    def closest_lat(self, lat):
        return self.closest(self.lat_grid, lat)

    def closest_lon(self, lon):
        return self.closest(self.lon_grid, lon)
    
    def closest_latlon(self, lat, lon):
        return self.closest_lat(lat), self.closest_lon(lon)
    
    def grid_latlons(self, df):
        df[["GRID_LAT", "GRID_LON"]] = df.apply(
            lambda row: self.closest_latlon(row.LAT, row.LON),
            result_type="expand",
            axis=1)
        return df
    
#%%
dwd_sd = DWDSTationData(
    "data/raw/dwd/airtemp2m/unzipped",
    "2000-01-01",
    "today"
    )
#%%
era5 = xr.open_dataset("data/raw/ERA_5_Germany/1.grib", engine="cfgrib")
era5 = era5["t2m"] - 273.15
#%%
gridder = Gridder(era5["latitude"].values, era5["longitude"].values)
dt = pd.to_datetime("2022-01-01 12:00:00")
dt = pd.to_datetime("2022-05-01")
#%%
start = pd.to_datetime("2022-05-01")
end = pd.to_datetime("2022-06-01")
dwd_df = dwd_sd.between_datetimes(start, end)
dwd_df = gridder.grid_latlons(dwd_df)

#%%
dt = pd.to_datetime("2022-05-20 16:00:00")
#era5_df = era5.sel(time=dt).to_dataframe()
era5_df = era5[(era5["time"] > start) & (era5["time"] < end)].to_dataframe()

#%%
#%%

df = dwd_df.merge(
    era5_df,
    left_on=["GRID_LAT", "GRID_LON", "DATETIME"],
    right_on=["latitude", "longitude", "time"])

df = df.drop([
    "LAT",
    "LON",
    "number",
    "step",
    "surface",
    "valid_time",
], axis=1)

df = df.rename({
    "TEMP": "TEMP_REAL",
    "t2m": "TEMP_ERA5"
}, axis=1)

df["TEMP_DIFF"] = df["TEMP_ERA5"] - df["TEMP_REAL"]

#%%
#%%
grouped = df.groupby(["STATION_ID"]).mean()
#plt.scatter(grouped["HEIGHT"], grouped["TEMP_DIFF"])

#grouped[grouped["TEMP_DIFF"] == grouped["TEMP_DIFF"].min()]
grouped
#%%
plt.scatter(df["HEIGHT"], df["TEMP_DIFF"])
plt.xlabel("Height [m]")
plt.ylabel("$T_{ERA5} - T_{real}$")
# %%
td = df["TEMP_DIFF"]
plt.boxplot(td)
plt.ylabel("$T_{ERA5} - T_{real}$")
#%%
fig, ax = plot_geopandas(df, "TEMP_DIFF")
ax.set_title("$T_{ERA5} - T_{real}$")
# %%
df[df["TEMP_DIFF"] == df["TEMP_DIFF"].min()]

#%%
era5.where((era5["time"] > start) & (era5["time"] < end))
#%%
#%%
dwd_df