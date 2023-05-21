from datasets import DWDSTationData, ECADStationData
import xarray as xr

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

class CountryPlot:
    def __init__(self,
                 shapefile_path: str = None,
                 era5: xr.Dataset = None,
                 dwd_data: DWDSTationData = None,
                 ecad_data: ECADStationData = None) -> None:
        
        self.vmin = -5
        self.vmax = 30
        self.cm = "coolwarm"
        
        if shapefile_path is not None:
            self.country = gpd.read_file(shapefile_path)
            self.country.crs = "epsg:4326"
        
        if era5 is not None:
            self.era5 = era5

        if dwd_data is not None:
            self.dwd = dwd_data
            self.dwd.crs = "epsg:4326"
        
        if ecad_data is not None:
            #TODO
            raise NotImplementedError

        
    
    def show(self):
        self.fig.show()
    
    def plot(self, datetime):
        datetime = pd.to_datetime(datetime)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        if self.era5 is not None:
            era5_t = era5["t2m"].sel(time=datetime) - 273.15
            era5_t.plot(ax=ax, vmin=self.vmin, vmax=self.vmax, cmap=self.cm)

        if self.dwd is not None:
            gdf = self.dwd.at_datetime(datetime)
            gdf.plot(
                ax=ax,
                column="TEMP",
                edgecolors="gray",
                cmap=self.cm,
                vmin=self.vmin,
                vmax=self.vmax,
            )

        if self.country is not None:
            self.country.plot(edgecolor="black", ax=ax, alpha=1, facecolor="none")
        
        self.fig = fig
        self.ax = ax
        
        return fig, ax
    
if __name__ == "__main__":
    dwd_sd = DWDSTationData(
        "data/raw/dwd/airtemp2m/unzipped",
        "2000-01-01",
        "today"
        )
    era5 = xr.open_dataset("data/raw/ERA_5_Germany/1.grib", engine="cfgrib")
    cp = CountryPlot(
        shapefile_path="data/shapefiles/DEU_adm0.shp",
        era5=era5,
        dwd_data=dwd_sd)
    datetime = "2022-02-22 14:00:00"
    cp.plot(datetime)