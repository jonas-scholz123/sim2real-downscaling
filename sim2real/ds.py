# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import lab as B
from tqdm import tqdm
import numpy as np
from deepsensor.data.utils import concat_tasks

import deepsensor.torch as elevation
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot.utils import plot_context_encoding, plot_offgrid_context

from sim2real.config import paths, names, data, out, Paths
from sim2real.utils import ensure_exists
from sim2real.datasets import load_era5
import cartopy.crs as ccrs
import cartopy.feature as feature


class SimTrainer:
    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        pass

    def add_era5(self):
        era5 = load_era5()


# %%

era5 = load_era5()

path = "data/raw/SRTM_DEM/srtm_germany_dtm.tif"
elevation = xr.open_rasterio(path)
elevation = elevation.rename(
    {
        "x": names.lon,
        "y": names.lat,
    }
)
coarsen = {
    names.lat: len(elevation[names.lat]) // len(era5[names.lat]),
    names.lon: len(elevation[names.lon]) // len(era5[names.lon]),
}
elevation = elevation.coarsen(coarsen, boundary="trim").mean()
elevation.name = names.height

era5 = era5.rename({names.time: "time"})
# %%
x1_min = float(min(elevation[names.lat].min(), era5[names.lat].min()))
x1_max = float(max(elevation[names.lat].max(), era5[names.lat].max()))
x2_min = float(min(elevation[names.lon].min(), era5[names.lon].min()))
x2_max = float(max(elevation[names.lon].max(), era5[names.lon].max()))

print((x1_min, x1_max), (x2_min, x2_max))

# %%

data_processor = DataProcessor(
    x1_name=names.lat,
    x2_name=names.lon,
    x1_map=(x1_min, x1_max),
    x2_map=(x2_min, x2_max),
)

era5, elevation = data_processor([era5, elevation])
task_loader = TaskLoader(context=[era5[names.temp], elevation], target=era5[names.temp])
model = ConvNP(data_processor, task_loader, verbose=True)


# %%
def sample_plot(task):
    fig = plot_context_encoding(model, task, task_loader)
    plt.show()


def context_target_plot(task):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.TransverseMercator(central_longitude=10))
    bounds = [*data.bounds.lon, *data.bounds.lat]
    ax.set_extent(bounds, crs=ccrs.PlateCarree())

    ax.add_feature(feature.BORDERS, linewidth=0.25)
    ax.add_feature(feature.LAKES, linewidth=0.25)
    ax.add_feature(feature.RIVERS, linewidth=0.25)
    ax.add_feature(feature.OCEAN)
    ax.add_feature(feature.LAND)
    ax.coastlines(linewidth=0.25)

    plot_offgrid_context(
        ax,
        task,
        data_processor,
        task_loader,
        plot_target=True,
        add_legend=True,
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
    )
    plt.show()


date = pd.to_datetime("2022-01-01")
task = task_loader(date, [0.1, "all"], [0.1])
sample_plot(task)
context_target_plot(task)
# %%

import wandb

if out.wandb:
    wandb.init(project="climate-sim2real")


opt = optim.Adam(model.model.parameters(), lr=5e-4)


def train_step(tasks):
    if not isinstance(tasks, list):
        tasks = [tasks]
    opt.zero_grad()
    task_losses = []
    for task in tasks:
        task_losses.append(model.loss_fn(task, normalise=True))
    mean_batch_loss = B.mean(torch.stack(task_losses))
    mean_batch_loss.backward()
    opt.step()
    return mean_batch_loss.detach().cpu().numpy()


batch_size = 4
n_batches = 10
n_epochs = 100
epoch_losses = []
start, end = era5.time.values.min(), era5.time.values.max()
dates = pd.date_range(start, end)
for epoch in tqdm(range(n_epochs), position=0):
    batch_losses = []
    for batch_i in range(n_batches):
        batch_tasks = []
        for i in range(batch_size):
            date = np.random.choice(dates, 1)[0]
            n_obs = np.random.randint(5, 50)  # Random number of context observations
            # n_obs = np.random.uniform(0.001, 0.9)  # Random fraction of context observations
            # n_t = 5000  # Number of target points
            n_t = 1.0  # All targets
            task = task_loader(
                date, (n_obs, "all"), n_t
            )  # Generate task with all auxiliary context
            batch_tasks.append(task)
        task = concat_tasks(batch_tasks)
        task["Y_c"][1].mask = task["Y_c"][1].mask[:, 0:1, :]
        batch_loss = train_step(task)
        batch_losses.append(batch_loss)

        wandb.log({"epoch": epoch, "train_loss": batch_loss})

    epoch_loss = np.mean(batch_losses)
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch} loss: {epoch_loss:.2f}")
# %%
plt.plot(epoch_losses)
# %%
task
