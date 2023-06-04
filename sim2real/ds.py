# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import lab as B
from tqdm import tqdm
import numpy as np

import deepsensor.torch
from deepsensor.data.utils import concat_tasks
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot.utils import plot_context_encoding, plot_offgrid_context

from sim2real.config import paths, names, data, out, Paths
from sim2real.utils import ensure_exists
from sim2real.datasets import load_elevation, load_era5
import cartopy.crs as ccrs
import cartopy.feature as feature


def sample_plot(model, task, task_loader):
    fig = plot_context_encoding(model, task, task_loader)
    plt.show()


def context_target_plot(task, data_processor, task_loader):
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


class SimTrainer:
    def __init__(self, paths: Paths) -> None:
        self.paths = paths

        # TODO: Make data config.
        self.era5_context_points = (5, 30)
        self.era5_target_points = 100

        # TODO: Make opt config.
        batch_size = 4

        self.raw = []
        self.target = []
        self.context_points = []
        self.target_points = []

        self.era5 = self.add_era5()
        self.elevation = self.add_elevation()

        self.data_processor = self._init_data_processor()
        self.processed = self.data_processor(self.raw)
        self.target = self.data_processor(self.target)

        self.task_loader = TaskLoader(
            context=self.processed, target=self.target, time_freq="H"
        )

        # TODO: validation, evaluation
        start, end = (
            self.era5[names.time].values.min(),
            self.era5[names.time].values.max(),
        )

        train_set = Taskset(
            start, end, self.task_loader, self.context_points, self.target_points
        )

        # Don't turn into pytorch tensors. We just want the sampling functionality.
        self.train_loader = DataLoader(
            train_set, shuffle=True, batch_size=batch_size, collate_fn=concat_tasks
        )

        self.model = ConvNP(self.data_processor, self.task_loader)
        self.opt = optim.Adam(self.model.model.parameters(), lr=5e-4)

        pass

    def _init_data_processor(self):
        x1_min = float(min(data[names.lat].min() for data in self.raw))
        x2_min = float(min(data[names.lon].min() for data in self.raw))
        x1_max = float(max(data[names.lat].max() for data in self.raw))
        x2_max = float(max(data[names.lon].max() for data in self.raw))
        return DataProcessor(
            time_name=names.time,
            x1_name=names.lat,
            x2_name=names.lon,
            x1_map=(x1_min, x1_max),
            x2_map=(x2_min, x2_max),
        )

    def add_era5(self):
        era5 = load_era5()[names.temp]
        self.raw.append(era5)
        self.target.append(era5)
        self.context_points.append(self.era5_context_points)
        self.target_points.append(self.era5_target_points)
        return era5

    def add_elevation(self):
        elevation = load_elevation()
        coarsen = {
            names.lat: len(elevation[names.lat]) // len(self.era5[names.lat]),
            names.lon: len(elevation[names.lon]) // len(self.era5[names.lon]),
        }
        elevation = elevation.coarsen(coarsen, boundary="trim").mean()
        self.raw.append(elevation)
        self.context_points.append("all")

    def train_on_batch(self, tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        self.opt.zero_grad()
        task_losses = []
        for task in tasks:
            task_losses.append(self.model.loss_fn(task, normalise=True))
        mean_batch_loss = B.mean(torch.stack(task_losses))
        mean_batch_loss.backward()
        self.opt.step()
        return mean_batch_loss.detach().cpu().numpy()

    def train(self):
        n_batches = 10
        n_epochs = 20
        epoch_losses = []

        train_iter = iter(self.train_loader)

        pbar = tqdm(range(1, n_epochs + 1))

        for epoch in pbar:
            batch_losses = []
            for _ in range(n_batches):
                task = next(train_iter)
                task["Y_c"][1].mask = task["Y_c"][1].mask[:, 0:1, :]
                batch_loss = self.train_on_batch(task)
                batch_losses.append(batch_loss)
                # self.wandb.log({"epoch": epoch, "train_loss": batch_loss})

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            pbar.set_postfix({"loss": epoch_loss})

    def plot_prediction(self):
        # TODO: generalise.
        test_date = pd.Timestamp("2022-01-01")
        task = self.task_loader(test_date, context_sampling=(30, "all"))

        mean_ds, std_ds = self.model.predict(task, X_t=self.era5)

        coord_map = {
            names.lat: self.era5[names.lat],
            names.lon: self.era5[names.lon],
        }

        # Fix rounding errors along dimensions.
        mean_ds = mean_ds.assign_coords(coord_map)
        std_ds = std_ds.assign_coords(coord_map)
        err_da = mean_ds[names.temp] - self.era5

        # proj = ccrs.TransverseMercator(central_longitude=10)
        proj = ccrs.PlateCarree()

        fig, axs = plt.subplots(
            subplot_kw={"projection": proj}, nrows=1, ncols=4, figsize=(20, 5)
        )

        sel = {names.time: task["time"]}

        self.era5.sel(sel).plot(cmap="seismic", ax=axs[0])
        axs[0].set_title("ERA5")
        mean_ds[names.temp].sel(sel).plot(cmap="seismic", ax=axs[1])
        axs[1].set_title("ConvNP mean")
        std_ds[names.temp].sel(sel).plot(cmap="Greys", ax=axs[2])
        axs[2].set_title("ConvNP std dev")
        err_da.sel(sel).plot(cmap="seismic", ax=axs[3])
        axs[3].set_title("ConvNP error")
        plot_offgrid_context(axs, task, self.data_processor, s=3**2, linewidths=0.5)

        bounds = [*data.bounds.lon, *data.bounds.lat]
        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.add_feature(feature.BORDERS, linewidth=0.25)
            ax.add_feature(feature.LAKES, linewidth=0.25)
            ax.add_feature(feature.RIVERS, linewidth=0.25)
            ax.add_feature(feature.OCEAN)
            ax.add_feature(feature.LAND)
            ax.coastlines(linewidth=0.25)
        plt.show()


class Taskset(Dataset):
    def __init__(
        self, start, end, taskloader: TaskLoader, num_context, num_target
    ) -> None:
        self.dates = pd.date_range(start, end)
        self.num_context, self.num_target = num_context, num_target
        self.task_loader = taskloader

    def _map_num_context(self, num_context):
        """
        Map num_context specs to something understandable by TaskLoader.
        """
        if isinstance(num_context, list):
            return [self._map_num_context(el) for el in num_context]
        elif isinstance(num_context, tuple):
            return np.random.randint(num_context)
        else:
            return num_context

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        # Random number of context observations
        num_context = self._map_num_context(self.num_context)
        date = self.dates[idx]
        task = self.task_loader(date, num_context, self.num_target)
        return task


s = SimTrainer(paths)
s.train()
s.plot_prediction()
