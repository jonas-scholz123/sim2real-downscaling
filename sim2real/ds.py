# %%
import os
import random
import wandb
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import lab as B
from tqdm import tqdm
import numpy as np
from typing import Iterable
from dataclasses import asdict


import neuralprocesses.torch as nps
import deepsensor.torch
from deepsensor.data.utils import concat_tasks
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot.utils import plot_context_encoding, plot_offgrid_context

from sim2real.deepsensor.data.task import Task
from sim2real.utils import ensure_dir_exists, ensure_exists
from sim2real.datasets import load_elevation, load_era5
from sim2real.plots import save_plot
import cartopy.crs as ccrs
import cartopy.feature as feature

from sim2real.config import paths, names, data, out, Paths, opt


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
        self.time_freq = "D"
        self.opt = opt
        self.out = out

        B.set_global_device(self.opt.device)
        torch.set_default_device(self.opt.device)
        # B.cholesky_retry_factor = 1e6
        # B.epsilon = 1e-5

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
            context=self.processed, target=self.target, time_freq="D"
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
            train_set,
            shuffle=True,
            batch_size=self.opt.batch_size,
            collate_fn=concat_tasks,
        )

        self._init_log()

        self.model = ConvNP(self.data_processor, self.task_loader)
        self.model.model = self.model.model.to(self.opt.device)

        self._init_log()

    def _set_seeds(self):
        B.set_random_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)

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

    def _task_to_device(self, task: Task):
        def _to_device(item):
            if isinstance(item, nps.mask.Masked):
                item.mask = item.mask.to(self.device)
                item.y = item.y.to(self.device)

            if isinstance(item, torch.Tensor):
                return item.to(self.device)

            if isinstance(item, list):
                return [_to_device(el) for el in item]

            if isinstance(item, tuple):
                return tuple([_to_device(el) for el in item])

            return item

        # NOTE: this is not robust and would be much
        # more appropriate in modify_tensor or similar.

        for k in ["X_c", "X_t", "Y_c", "Y_t"]:
            task[k] = _to_device(task[k])

        return task

    def train_on_batch(self, tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        self.optimiser.zero_grad()
        task_losses = []
        for task in tasks:
            task_losses.append(self.model.loss_fn(task, normalise=True))
        mean_batch_loss = B.mean(torch.stack(task_losses))
        mean_batch_loss.backward()
        self.optimiser.step()
        return mean_batch_loss.detach().cpu().numpy()

    def train(self):
        epoch_losses = []

        train_iter = iter(self.train_loader)
        self.optimiser = optim.Adam(self.model.model.parameters(), lr=self.opt.lr)

        for epoch in self.pbar:
            batch_losses = []
            for _ in range(self.opt.batches_per_epoch):
                task = next(train_iter)
                # task = self._task_to_device(task)
                task["Y_c"][1].mask = task["Y_c"][1].mask[:, 0:1, :]
                batch_loss = self.train_on_batch(task)
                batch_losses.append(batch_loss)

            train_loss = np.mean(batch_losses)
            epoch_losses.append(train_loss)
            self._log(epoch, train_loss)

            if out.plots:
                for date in ["2022-01-01", "2022-06-01"]:
                    self.plot_prediction(name=f"epoch_{epoch}_{date}", date=date)

    def _init_log(self):
        self.pbar = tqdm(range(1, self.opt.num_epochs + 1))
        if self.out.wandb:
            config = {
                "opt": asdict(opt),
                "data": asdict(data),
                "model": "inferred",
            }
            self.wandb = wandb.init(project="climate-sim2real", config=config)

    def _log(self, epoch, train_loss):
        self.pbar.set_postfix({"loss": train_loss})
        if self.out.wandb:
            self.wandb.log({"epoch": epoch, "train_loss": train_loss})

    def plot_prediction(self, name=None, date="2022-01-01"):
        test_date = pd.Timestamp(date)
        task = self.task_loader(test_date, context_sampling=(30, "all"))
        # task = self.model.modify_task(task)
        # task = self._task_to_device(task)

        mean_ds, std_ds = self.model.predict(task, X_t=self.era5)

        coord_map = {
            names.lat: self.era5[names.lat],
            names.lon: self.era5[names.lon],
        }

        # Fix rounding errors along dimensions.
        mean_ds = mean_ds.assign_coords(coord_map)
        std_ds = std_ds.assign_coords(coord_map)
        err_da = mean_ds[names.temp] - self.era5

        proj = ccrs.TransverseMercator(central_longitude=10, approx=False)

        fig, axs = plt.subplots(
            subplot_kw={"projection": proj}, nrows=1, ncols=4, figsize=(10, 2.5)
        )

        sel = {names.time: task["time"]}

        self.era5.sel(sel).plot(cmap="seismic", ax=axs[0], transform=ccrs.PlateCarree())
        axs[0].set_title("ERA5")
        mean_ds[names.temp].sel(sel).plot(
            cmap="seismic", ax=axs[1], transform=ccrs.PlateCarree()
        )
        axs[1].set_title("ConvNP mean")
        std_ds[names.temp].sel(sel).plot(
            cmap="Greys", ax=axs[2], transform=ccrs.PlateCarree()
        )
        axs[2].set_title("ConvNP std dev")
        err_da.sel(sel).plot(cmap="seismic", ax=axs[3], transform=ccrs.PlateCarree())
        axs[3].set_title("ConvNP error")
        plot_offgrid_context(
            axs,
            task,
            self.data_processor,
            s=3**2,
            linewidths=0.5,
            add_legend=False,
            transform=ccrs.PlateCarree(),
        )

        bounds = [*data.bounds.lon, *data.bounds.lat]
        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.add_feature(feature.BORDERS, linewidth=0.25)
            # ax.add_feature(feature.LAKES, linewidth=0.25)
            # ax.add_feature(feature.RIVERS, linewidth=0.25)
            # ax.add_feature(feature.OCEAN)
            # ax.add_feature(feature.LAND)
            ax.coastlines(linewidth=0.25)

        if name is not None:
            ensure_exists(paths.out)
            save_plot(name, fig)
        else:
            plt.show()

        plt.clf()

    def loss_test(self):
        train_iter = iter(self.train_loader)
        task = next(train_iter)
        # task = self._task_to_device(task)
        task["Y_c"][1].mask = task["Y_c"][1].mask[:, 0:1, :]
        print("final: ", self.model.loss_fn(task, normalise=True))


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
            return np.random.randint(*num_context)
        else:
            return num_context

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        # Random number of context observations
        num_context = self._map_num_context(self.num_context)
        date = self.dates[idx]
        task = self.task_loader(date, num_context, self.num_target, deterministic=True)
        return task


s = SimTrainer(paths)
# s.loss_test()
s.train()
# s.plot_prediction()
