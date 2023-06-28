import os
import random
from typing import Tuple, Union
import wandb
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import lab as B
from tqdm import tqdm
import numpy as np
from dataclasses import asdict
from deepsensor.data.loader import TaskLoader
from deepsensor.model.models import ConvNP
from deepsensor.plot import receptive_field, offgrid_context
from sim2real.train.taskset import Taskset

from sim2real.utils import (
    ensure_dir_exists,
    ensure_exists,
    save_model,
    load_weights,
    get_default_data_processor,
)
from sim2real import keys, utils
from sim2real.plots import save_plot
from sim2real.modules import convcnp
import cartopy.crs as ccrs
import cartopy.feature as feature

from sim2real.config import (
    DataSpec,
    OptimSpec,
    OutputSpec,
    ModelSpec,
    Paths,
    names,
)

from abc import ABC, abstractmethod
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer(ABC):
    def __init__(
        self,
        paths: Paths,
        opt: OptimSpec,
        out: OutputSpec,
        data: DataSpec,
        mspec: ModelSpec,
    ) -> None:
        self.paths = paths
        self.opt = opt
        self.out = out
        self.data = data
        self.mspec = mspec

        self.loaded_checkpoint = False

        self.exp_dir = self._get_exp_dir(mspec)
        self.latest_path = f"{utils.weight_dir(self.exp_dir)}/latest.h5"
        self.best_path = f"{utils.weight_dir(self.exp_dir)}/best.h5"
        [ensure_dir_exists(path) for path in [self.latest_path, self.best_path]]

        if self.opt.device == "cuda":
            # Can't do this with MPS (need to change the deepsensor __init__ file).
            B.set_global_device(self.opt.device)
            torch.set_default_device(self.opt.device)

        self.data_processor = self._init_data_processor()
        self._init_dataloaders()
        self._init_model()
        self._init_log()

    @abstractmethod
    def _get_exp_dir(self, mspec: ModelSpec):
        """
        Returns: exp_dir: str, the string specifying the training directory.
            Should incorporate all relevant aspects of the model specification
            so that each path is associated with one architecture.
        """
        return

    @abstractmethod
    def _init_tasksets(self) -> Tuple[Taskset, Taskset, Taskset]:
        """
        Returns: (Taskset, Taskset, Taskset) representing
            (train, val, test) task loaders.
        """
        return

    def _init_dataloaders(self):
        train_set, cv_set, test_set = self._init_tasksets()

        if self.opt.device == "cuda":
            gen = torch.Generator(device="cuda")
        else:
            gen = torch.Generator()

        # Don't turn into pytorch tensors. We just want the sampling functionality.
        collate_fn = lambda x: x
        self.train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=self.opt.batch_size,
            collate_fn=collate_fn,
            generator=gen,
        )

        self.cv_loader = DataLoader(
            cv_set,
            shuffle=False,
            batch_size=self.opt.batch_size_val,
            collate_fn=collate_fn,
            generator=gen,
        )

        self.test_loader = DataLoader(
            test_set,
            shuffle=False,
            batch_size=self.opt.batch_size_val,
            collate_fn=collate_fn,
            generator=gen,
        )

    def _set_seeds(self):
        B.set_random_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)

    def _init_data_processor(self):
        return get_default_data_processor()

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

        self.optimiser.step()
        l = float(mean_batch_loss.detach().cpu().numpy())
        return l

    def train(self):
        epoch_losses = []

        train_iter = iter(self.train_loader)

        self.optimiser = optim.Adam(self.model.model.parameters(), lr=self.opt.lr)
        self.early_stopper = EarlyStopper(self.opt.early_stop_patience)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="min",
            factor=self.opt.scheduler_factor,
            patience=self.opt.scheduler_patience,
        )

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
            val_loss = self.evaluate()
            self._save_weights(epoch, val_loss)
            self._log(epoch, train_loss, val_loss)

            if self.out.plots:
                # TODO: move to config, child classes.
                for date in ["2022-01-01", "2022-06-01"]:
                    self.plot_prediction(name=f"epoch_{epoch}_{date}", date=date)

            if self.early_stopper.early_stop(val_loss):
                break

            self.scheduler.step(val_loss)

    def _init_model(self):
        # Construct custom model.
        model_kwargs = dict(
            dim_yc=self.mspec.dim_yc,
            dim_yt=self.mspec.dim_yt,
            points_per_unit=self.mspec.ppu,
            likelihood=self.mspec.likelihood,
            unet_channels=self.mspec.unet_channels,
            encoder_scales_learnable=self.mspec.encoder_scales_learnable,
            decoder_scale_learnable=self.mspec.decoder_scale_learnable,
            film=self.mspec.film,
            freeze_film=self.mspec.freeze_film,
        )

        # model = convcnp.from_taskloader(self.task_loader, **model_kwargs)

        # model = ConvNP(self.data_processor, self.task_loader, model)
        model = ConvNP(self.data_processor, self.task_loader)
        self.best_val_loss = load_weights(None, self.best_path, loss_only=True)[1]
        self.num_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        print(f"Model number parameters: {self.num_params}")
        print(f"Model receptive field (fraction): {model.model.receptive_field}")

        if self.opt.start_from == "best":
            model.model, self.best_val_loss, self.start_epoch = load_weights(
                model.model, self.best_path
            )
            print("Starting training from best.")
            self.loaded_checkpoint = True
        elif self.opt.start_from == "latest":
            model.model, self.best_val_loss, self.start_epoch = load_weights(
                model.model, self.latest_path
            )
            print("Starting training from latest.")
            self.loaded_checkpoint = True
        else:
            print("Starting training from scratch.")
            self.start_epoch = 0

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
            # Set this in case we're running on HPC where we can't run login command.
            os.environ["WANDB_API_KEY"] = keys.WANDB_API_KEY
            config = {
                "opt": asdict(self.opt),
                "data": asdict(self.data),
                "model": asdict(self.mspec),
            }
            self.wandb = wandb.init(
                project="climate-sim2real", config=config, name=self.out.wandb_name
            )

    def _log(self, epoch, train_loss, val_loss):
        self.pbar.set_postfix(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
            }
        )
        if self.out.wandb:
            self.wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                }
            )

    def plot_receptive_field(self):
        receptive_field(
            self.model.model.receptive_field,
            self.data_processor,
            ccrs.PlateCarree(),
            [*self.data.bounds.lon, *self.data.bounds.lat],
        )

        plt.gca().set_global()

    @abstractmethod
    def _plot_X_t(self):
        """
        For plotting, which target points should be predicted?
        Returns pd.DataFrame/Series or xr.DataArray/DataSet
        """
        return

    def plot_prediction(self, name=None, date="2022-01-01", task=None):
        if task is None:
            test_date = pd.Timestamp(date)
            task = self.task_loader(test_date, context_sampling=(30, "all"))

        mean_ds, std_ds = self.model.predict(task, X_t=self._plot_X_t())

        coord_map = {
            names.lat: self.var_raw[names.lat],
            names.lon: self.var_raw[names.lon],
        }

        # Fix rounding errors along dimensions.
        mean_ds = mean_ds.assign_coords(coord_map)
        std_ds = std_ds.assign_coords(coord_map)
        err_da = mean_ds[names.temp] - self.var_raw

        proj = ccrs.TransverseMercator(central_longitude=10, approx=False)

        fig, axs = plt.subplots(
            subplot_kw={"projection": proj}, nrows=1, ncols=4, figsize=(10, 2.5)
        )

        sel = {names.time: task["time"]}

        era5_plot = self.var_raw.sel(sel).plot(
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

        bounds = [*self.data.bounds.lon, *self.data.bounds.lat]
        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.add_feature(feature.BORDERS, linewidth=0.25)
            ax.coastlines(linewidth=0.25)

        if name is not None:
            ensure_exists(self.paths.out)
            save_plot(self.exp_dir, name, fig)
        else:
            plt.show()

        plt.close()
        plt.clf()
