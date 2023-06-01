# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import deepsensor.torch as elevation
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot.utils import plot_context_encoding, plot_offgrid_context

from config import paths, names
from utils import ensure_exists
from datasets import load_era5

# %%

era5 = load_era5()

path = "data/raw/SRTM_DEM/srtm_germany_dtm.tif"
elevation = xr.open_rasterio(path)
elevation = elevation.rename(
    {
        "x": names.lat,
        "y": names.lon,
    }
)
coarsen = {
    names.lat: len(elevation[names.lat]) // len(era5[names.lat]),
    names.lon: len(elevation[names.lon]) // len(era5[names.lon]),
}
elevation = elevation.coarsen(coarsen, boundary="trim").mean()
# %%
x1_min = min(elevation[names.lat].min(), era5[names.lat].min())
x1_max = max(elevation[names.lat].max(), era5[names.lat].max())
x2_min = min(elevation[names.lon].min(), era5[names.lon].min())
x2_max = max(elevation[names.lon].max(), era5[names.lon].max())

data_processor = DataProcessor(
    x1_name=names.lat,
    x2_name=names.lon,
    x1_map=(x1_min, x1_max),
    x2_map=(x2_min, x2_max),
)

era5, elevation = data_processor([era5, elevation])
# %%

task_loader = TaskLoader(context=[era5[names.temp], elevation], target=era5[names.temp])
# %%
model = ConvNP(data_processor, task_loader, verbose=True)
# %%

date = pd.to_datetime("2022-01-01")
task = task_loader(date, [0.1, 0.1], [0.1])
# %%
fig = plot_context_encoding(model, task, task_loader)
plt.show()
