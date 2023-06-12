# %%
import os
import random
import wandb
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import lab as B
from tqdm import tqdm
import numpy as np
from typing import Iterable, Tuple
from dataclasses import asdict


import neuralprocesses.torch as nps
import deepsensor.torch
from deepsensor.data.utils import (
    concat_tasks,
    construct_x1x2_ds,
    construct_circ_time_ds,
)
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot import receptive_field, context_encoding, offgrid_context

from sim2real.utils import (
    ensure_dir_exists,
    ensure_exists,
    exp_dir_sim,
    save_model,
    load_weights,
)
from sim2real import utils
from sim2real.datasets import load_elevation, load_era5
from sim2real.plots import save_plot
import cartopy.crs as ccrs
import cartopy.feature as feature

from sim2real.config import (
    DataSpec,
    OptimSpec,
    OutputSpec,
    paths,
    names,
    data,
    out,
    Paths,
    opt,
)

import sim2real.config as cfg


class Taskset(Dataset):
    def __init__(
        self,
        time_range: Tuple[str, str],
        taskloader: TaskLoader,
        num_context,
        num_target,
        freq="H",
        deterministic=False,
    ) -> None:
        self.dates = pd.date_range(*time_range, freq=freq)
        self.num_context, self.num_target = num_context, num_target
        self.task_loader = taskloader
        self.deterministic = deterministic
        self.rng = np.random.default_rng(opt.seed + 1)

    def _map_num_context(self, num_context):
        """
        Map num_context specs to something understandable by TaskLoader.
        """
        if isinstance(num_context, list):
            return [self._map_num_context(el) for el in num_context]
        elif isinstance(num_context, tuple):
            return int(self.rng.integers(*num_context))
        else:
            return num_context

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        if idx == len(self) - 1 and self.deterministic:
            # Reset rng for deterministic
            self.rng = np.random.default_rng(opt.seed + 1)
        # Random number of context observations
        num_context = self._map_num_context(self.num_context)
        date = self.dates[idx]
        task = self.task_loader(
            date, num_context, self.num_target, deterministic=self.deterministic
        )
        return task


def sample_plot(model, task, task_loader):
    fig = context_encoding(model, task, task_loader)
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

    offgrid_context(
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
    def __init__(
        self,
        paths: Paths,
        opt: OptimSpec,
        out: OutputSpec,
        data: DataSpec,
        train: bool = True,
    ) -> None:
        self.paths = paths
        self.opt = opt
        self.out = out
        self.data = data

        self.exp_dir = exp_dir_sim(cfg.model)
        self.latest_path = f"{utils.weight_dir(self.exp_dir)}/latest.h5"
        self.best_path = f"{utils.weight_dir(self.exp_dir)}/best.h5"

        [ensure_dir_exists(path) for path in [self.latest_path, self.best_path]]

        B.set_global_device(self.opt.device)
        # torch.set_default_device(self.opt.device)
        B.cholesky_retry_factor = 1e8
        B.epsilon = 1e-12

        self.context_points = []
        self.target_points = []

        self.era5_raw = self.add_era5()
        self.aux_raw = self.add_aux()
        self.raw = [self.era5_raw, self.aux_raw]

        self.data_processor = self._init_data_processor()
        self.era5, self.aux = self.data_processor(self.raw)
        # Add spatiotemporal data.
        self.aux = self._expand_aux(self.aux)

        self._init_loaders()
        self._init_model()

        if train:
            self._init_log()

    def _init_loaders(self):
        # don't collate as we want a list of tasks,
        # not any tensors.
        collate_fn = lambda x: x
        self.task_loader = TaskLoader(
            context=[self.era5, self.aux], target=[self.era5], time_freq="H"
        )

        train_set = Taskset(
            data.train_dates,
            self.task_loader,
            self.context_points,
            self.target_points,
            deterministic=False,
        )

        cv_set = Taskset(
            data.cv_dates,
            self.task_loader,
            self.context_points,
            self.target_points,
            data.val_freq,
            deterministic=True,
        )

        test_set = Taskset(
            data.test_dates,
            self.task_loader,
            self.context_points,
            self.target_points,
            data.val_freq,
            deterministic=True,
        )

        # Don't turn into pytorch tensors. We just want the sampling functionality.
        self.train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=self.opt.batch_size,
            collate_fn=collate_fn,
        )

        self.cv_loader = DataLoader(
            cv_set,
            shuffle=False,
            batch_size=self.opt.batch_size_val,
            collate_fn=collate_fn,
        )

        self.test_loader = DataLoader(
            test_set,
            shuffle=False,
            batch_size=self.opt.batch_size_val,
            collate_fn=collate_fn,
        )

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
        self.context_points.append(self.data.era5_context)
        self.target_points.append(self.data.era5_target)
        return era5

    def add_aux(self):
        def _coarsen(high_res, low_res):
            """
            Coarsen factor for shrinking something high-res to low-res.
            """
            factor = self.data.aux_coarsen_factor * len(high_res) // len(low_res)
            return int(factor)

        aux = load_elevation()
        coarsen = {
            names.lat: _coarsen(aux[names.lat], self.era5_raw[names.lat]),
            names.lon: _coarsen(aux[names.lon], self.era5_raw[names.lon]),
        }
        aux = aux.coarsen(coarsen, boundary="trim").mean()
        self.context_points.append("all")
        return aux

    def _expand_aux(self, aux):
        x1x2_ds = construct_x1x2_ds(aux)
        aux["x1_arr"] = x1x2_ds["x1_arr"]
        aux["x2_arr"] = x1x2_ds["x2_arr"]
        times = self.era5_raw[names.time].values
        dates = pd.date_range(times.min(), times.max(), freq="H")

        # Day of year.
        doy_ds = construct_circ_time_ds(dates, freq="D")
        aux["cos_D"] = doy_ds["cos_D"]
        aux["sin_D"] = doy_ds["sin_D"]

        # Time of day.
        tod_ds = construct_circ_time_ds(dates, freq="H")
        aux["cos_H"] = tod_ds["cos_H"]
        aux["sin_H"] = tod_ds["sin_H"]
        return aux

    def eval_on_batch(self, tasks):
        with torch.no_grad():
            if not isinstance(tasks, list):
                tasks = [tasks]
            task_losses = []
            for task in tasks:
                task_losses.append(self.model.loss_fn(task, normalise=True))
            mean_batch_loss = B.mean(torch.stack(task_losses))

        return float(mean_batch_loss.detach().cpu().numpy())

    def evaluate(self):
        batch_losses = []
        for i, task in enumerate(self.cv_loader):
            batch_loss = self.eval_on_batch(task)
            batch_losses.append(batch_loss)
        return np.mean(batch_losses)

    def train_on_batch(self, tasks):
        if not isinstance(tasks, list):
            tasks = [tasks]
        self.optimiser.zero_grad()
        task_losses = []
        for task in tasks:
            task_losses.append(self.model.loss_fn(task, normalise=True))
        mean_batch_loss = B.mean(torch.stack(task_losses))
        mean_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 0.01)

        l = float(mean_batch_loss.detach().cpu().numpy())
        if l < -5:
            print("Bad loss!")
            self.plot_prediction(name=f"loss_{l}", task=task)
        self.optimiser.step()
        return l

    def overfit_train(self):
        """
        Overfit to just one task to see if the model works.
        """
        epoch_losses = []

        self.optimiser = optim.Adam(self.model.model.parameters(), lr=self.opt.lr)

        self.pbar = tqdm(range(self.start_epoch, self.opt.num_epochs + 1))

        if self.start_epoch == 1:
            val_loss = self.evaluate()
            self._log(0, val_loss, val_loss)

        for epoch in self.pbar:
            batch_losses = []
            for i in range(self.opt.batches_per_epoch):
                date = "2012-01-01"
                test_date = pd.Timestamp(date)
                task = self.task_loader(test_date, context_sampling=("all", "all"))
                if i == 0:
                    self.plot_prediction(name=f"epoch_{epoch}_test", task=task)
                try:
                    batch_loss = self.train_on_batch(task)
                except:
                    self.plot_prediction(name=f"epoch_{epoch}_test_broken", task=task)
                    return

                batch_losses.append(batch_loss)

            train_loss = np.mean(batch_losses)
            epoch_losses.append(train_loss)
            val_lik = self.evaluate()
            self._log(epoch, train_loss, val_lik)
            self._save_weights(epoch, val_lik)

    def train(self):
        epoch_losses = []

        train_iter = iter(self.train_loader)

        self.optimiser = optim.Adam(self.model.model.parameters(), lr=self.opt.lr)
        # self.optimiser = optim.SGD(self.model.model.parameters(), lr=self.opt.lr)

        self.pbar = tqdm(range(self.start_epoch, self.opt.num_epochs + 1))

        if self.start_epoch == 1:
            val_loss = self.evaluate()
            self._log(0, val_loss, val_loss)

        for epoch in self.pbar:
            batch_losses = []
            for i in range(self.opt.batches_per_epoch):
                # Usually one epoch would be going through the whole dataset.
                # This is too long for us and we want to monitor more often.
                # Even though it's a bit hacky we check if we've run out of tasks
                # and reset the data iterator if so.
                try:
                    task = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    task = next(train_iter)
                batch_loss = self.train_on_batch(task)
                batch_losses.append(batch_loss)

            train_loss = np.mean(batch_losses)
            epoch_losses.append(train_loss)
            val_lik = self.evaluate()
            self._log(epoch, train_loss, val_lik)
            self._save_weights(epoch, val_lik)

            if out.plots:
                for date in ["2022-01-01", "2022-06-01"]:
                    self.plot_prediction(name=f"epoch_{epoch}_{date}", date=date)

    def _init_model(self):
        # TODO: Custom model

        model = ConvNP(
            self.data_processor,
            self.task_loader,
            verbose=False,
            likelihood=cfg.model.likelihood,
            unet_channels=cfg.model.unet_channels,
            encoder_scales_learnable=cfg.model.encoder_scales_learnable,
            decoder_scale_learnable=cfg.model.decoder_scale_learnable,
        )
        self.best_val_loss = load_weights(None, self.best_path, loss_only=True)[1]
        self.num_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        print(f"Model number parameters: {self.num_params}")
        print(f"Model receptive field (fraction): {model.model.receptive_field}")

        if self.opt.start_from == "best":
            (
                model.model,
                self.best_val_loss,
                self.start_epoch,
                torch_state,
                numpy_state,
            ) = load_weights(model.model, self.best_path)
        elif self.opt.start_from == "latest":
            (
                model.model,
                self.best_val_loss,
                self.start_epoch,
                torch_state,
                numpy_state,
            ) = load_weights(model.model, self.latest_path)
        else:
            self.start_epoch = 0
            torch_state, numpy_state = None, None

        # Return RNG for seamless continuation.
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        # Start one epoch after where the last run started.
        self.start_epoch += 1
        print(f"Starting from episode {self.start_epoch}")
        model.model = model.model.to(self.opt.device)
        self.model = model

    def _save_weights(self, epoch, val_loss):
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        save_model(
            self.model.model,
            val_loss,
            epoch,
            spec=None,
            path=self.latest_path,
            torch_state=torch_state,
            numpy_state=numpy_state,
        )

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_model(
                self.model.model,
                val_loss,
                epoch,
                spec=None,
                path=self.best_path,
                torch_state=torch_state,
                numpy_state=numpy_state,
            )

    def _init_log(self):
        if self.out.wandb:
            config = {
                "opt": asdict(opt),
                "data": asdict(data),
                "model": asdict(cfg.model),
            }
            self.wandb = wandb.init(
                project="climate-sim2real", config=config, name=out.wandb_name
            )

    def _log(self, epoch, train_loss, val_loss):
        self.pbar.set_postfix({"train_loss": train_loss, "val_loss": val_loss})
        if self.out.wandb:
            self.wandb.log(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

    def plot_receptive_field(self):
        receptive_field(
            self.model.model.receptive_field,
            self.data_processor,
            ccrs.PlateCarree(),
            [*data.bounds.lon, *data.bounds.lat],
        )

        plt.gca().set_global()

    def plot_prediction(self, name=None, date="2022-01-01", task=None):
        if task is None:
            test_date = pd.Timestamp(date)
            task = self.task_loader(test_date, context_sampling=(30, "all"))

        mean_ds, std_ds = self.model.predict(task, X_t=self.era5_raw)

        coord_map = {
            names.lat: self.era5_raw[names.lat],
            names.lon: self.era5_raw[names.lon],
        }

        # Fix rounding errors along dimensions.
        mean_ds = mean_ds.assign_coords(coord_map)
        std_ds = std_ds.assign_coords(coord_map)
        err_da = mean_ds[names.temp] - self.era5_raw

        proj = ccrs.TransverseMercator(central_longitude=10, approx=False)

        fig, axs = plt.subplots(
            subplot_kw={"projection": proj}, nrows=1, ncols=4, figsize=(10, 2.5)
        )

        sel = {names.time: task["time"]}

        era5_plot = self.era5_raw.sel(sel).plot(
            cmap="seismic", ax=axs[0], transform=ccrs.PlateCarree()
        )
        cbar = era5_plot.colorbar
        vmin, vmax = cbar.vmin, cbar.vmax

        axs[0].set_title("ERA5")
        mean_ds[names.temp].sel(sel).plot(
            cmap="seismic",
            ax=axs[1],
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        axs[1].set_title("ConvNP mean")
        std_ds[names.temp].sel(sel).plot(
            cmap="Greys", ax=axs[2], transform=ccrs.PlateCarree()
        )
        axs[2].set_title("ConvNP std dev")
        err_da.sel(sel).plot(cmap="seismic", ax=axs[3], transform=ccrs.PlateCarree())
        axs[3].set_title("ConvNP error")
        offgrid_context(
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
            ax.coastlines(linewidth=0.25)

        if name is not None:
            ensure_exists(paths.out)
            save_plot(self.exp_dir, name, fig)
        else:
            plt.show()

        plt.close()
        plt.clf()


s = SimTrainer(paths, opt, out, data, train=True)
s.train()
