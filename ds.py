# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import deepsensor.torch as ds
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot.utils import plot_context_encoding, plot_offgrid_context

from config import paths
from utils import ensure_exists
from datasets import load_era5

# %%
era5 = load_era5()
# %%

era5 = era5.rename(
    {
        "latitude": "lat",
        "longitude": "lon",
    }
)
# %%

# data_processor = DataProcessor()

x1_min = era5["lat"].min()
x1_max = era5["lat"].max()
x2_min = era5["lon"].min()
x2_max = era5["lon"].max()

data_processor = DataProcessor(
    x1_name="lat", x2_name="lon", x1_map=(x1_min, x1_max), x2_map=(x2_min, x2_max)
)

era5 = data_processor(era5)

task_loader = TaskLoader(context=[era5["t2m"]], target=era5["t2m"])
# %%
# %%
model = ConvNP(data_processor, task_loader, verbose=True)
# %%

date = pd.to_datetime("2022-01-01")
task = task_loader(date, 0.1, 0.1)
fig = plot_context_encoding(model, task, task_loader)
plt.show()
