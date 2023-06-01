import pandas as pd
import geopandas as gpd
from config import names, data


class Gridder:
    def __init__(self, lats, lons):
        self.lat_grid = sorted(lats)
        self.lon_grid = sorted(lons)

    def closest_df(self, grid, series):
        diff = grid[1] - grid[0]
        bins = sorted(grid - diff / 2)
        a = pd.cut(series, bins)
        a = a.apply(lambda x: x.mid)
        return a

    def grid_latlons(self, df):
        df["GRID_LAT"] = self.closest_df(self.lat_grid, df[names.lat])
        df["GRID_LON"] = self.closest_df(self.lon_grid, df[names.lon])
        grid_geom = gpd.points_from_xy(df["GRID_LON"], df["GRID_LAT"], crs=data.crs)
        grid_geom = gpd.GeoSeries(grid_geom)

        grid_geom = grid_geom.to_crs(crs=3310)
        df2 = df.to_crs(crs=3310)
        df["GRID_DIST"] = grid_geom.distance(df2["geometry"])
        return df
