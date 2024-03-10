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
from deepsensor import context_encoding
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP
from deepsensor.plot import receptive_field, offgrid_context
from sim2real.datasets import load_elevation
from sim2real.train.taskset import Taskset

from sim2real.utils import (
    ensure_dir_exists,
    ensure_exists,
    save_model,
    load_weights,
    get_default_data_processor,
)
from sim2real import keys, utils
from sim2real.plots import init_fig, save_plot
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
    def __init__(self, patience=1, min_delta=0.01):
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
        self.metrics = {}

        self.paths = paths
        self.opt = opt
        self.out = out
        self.data = data
        self.mspec = mspec

        self.loaded_checkpoint = False

        self._init_log()

        self.exp_dir = self._get_exp_dir(mspec)
        self.latest_path = f"{utils.weight_dir(self.exp_dir)}/latest.h5"
        self.best_path = f"{utils.weight_dir(self.exp_dir)}/best.h5"
        [ensure_dir_exists(path) for path in [self.latest_path, self.best_path]]

        if self.opt.device == "cuda":
            # Can't do this with MPS (need to change the deepsensor __init__ file).
            B.set_global_device(self.opt.device)
            torch.set_default_device(self.opt.device)
            self.gen = torch.Generator(device="cuda")
        else:
            self.gen = torch.Generator()

        self.data_processor = self._init_data_processor()
        self._init_dataloaders()
        self._init_model()

        self.sample_tasks = self._init_sample_tasks()

    def _init_sample_tasks(self):
        idx = len(self.val_set) // 2
        sample_tasks = [self.val_set[10], self.val_set[idx]]
        return sample_tasks

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

    def _to_dataloader(self, dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # Don't turn into pytorch tensors. We just want the sampling functionality.
            collate_fn=lambda x: x,
            generator=self.gen,
        )

    def _init_dataloaders(self):
        self.train_set, self.val_set, self.test_set = self._init_tasksets()

        self.train_loader = self._to_dataloader(
            self.train_set, self.opt.batch_size, shuffle=True
        )
        self.cv_loader = self._to_dataloader(self.val_set, self.opt.batch_size_val)
        self.test_loader = self._to_dataloader(self.test_set, self.opt.batch_size_val)

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

    def compute_loglik(self, loader, max_num_batches=100):
        batch_losses = []
        for i, task in tqdm(enumerate(loader)):
            if i == max_num_batches:
                break
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
        # torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 0.01)

        self.optimiser.step()
        l = float(mean_batch_loss.detach().cpu().numpy())
        return l

    def train(self):
        num_params_trainable = sum(
            p.numel() for p in self.model.model.parameters() if p.requires_grad
        )
        num_params = sum(p.numel() for p in self.model.model.parameters())

        print(f"Training {num_params_trainable} out of {num_params} parameters.")
        print(f"Starting from episode {self.start_epoch}")

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
            # Before training, get initial eval and plots.
            val_loss = self.compute_loglik(self.cv_loader)
            self.metrics[names.val_loss] = val_loss
            self.metrics[names.train_loss] = val_loss
            self.metrics[names.epoch] = 0
            self._log()
            self.plot_sample_tasks(0)

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
            val_loss = self.compute_loglik(self.cv_loader)
            self._save_weights(epoch, val_loss)
            self.metrics[names.val_loss] = val_loss
            self.metrics[names.train_loss] = train_loss
            self.metrics[names.epoch] = epoch
            self._log()

            if self.out.plots:
                self.plot_sample_tasks(epoch)

            if self.early_stopper.early_stop(val_loss):
                break

            self.scheduler.step(val_loss)

    def _init_model(self):
        # Construct custom model.
        model_kwargs = dict(
            dim_yc=self.mspec.dim_yc,
            dim_yt=self.mspec.dim_yt,
            internal_density=self.mspec.ppu,
            likelihood=self.mspec.likelihood,
            unet_channels=self.mspec.unet_channels,
            encoder_scales=self.mspec.encoder_scales,
            decoder_scale=self.mspec.decoder_scale,
            encoder_scales_learnable=self.mspec.encoder_scales_learnable,
            decoder_scale_learnable=self.mspec.decoder_scale_learnable,
            aux_t_mlp_layers=self.mspec.aux_t_mlp_layers,
        )

        model = ConvNP(self.data_processor, self.task_loader, **model_kwargs)
        print(model.model)
        try:
            self.best_val_loss = load_weights(None, self.best_path, loss_only=True)[1]
        except FileNotFoundError:
            self.best_val_loss = float("inf")

        print(f"Model receptive field (fraction): {model.model.receptive_field}")

        if self.opt.start_from == "best":
            try:
                model.model, self.best_val_loss, self.start_epoch = load_weights(
                    model.model, self.best_path
                )
                print(f"Loaded best weights from {self.best_path}.")
                self.loaded_checkpoint = True
            except:
                self.start_epoch = 0
        elif self.opt.start_from == "latest":
            try:
                model.model, self.best_val_loss, self.start_epoch = load_weights(
                    model.model, self.latest_path
                )
                print(f"Loaded latest weights from {self.latest_path}.")
                self.loaded_checkpoint = True
            except:
                self.start_epoch = 0
        else:
            print("Initialised random weights.")
            self.start_epoch = 0

        # Start one epoch after where the last run started.
        self.start_epoch += 1
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

    def _add_aux(
        self, res_factor=1.0
    ) -> Tuple[Union[xr.DataArray, pd.Series], Union[float, int, str]]:
        def _coarsen(high_res):
            """
            Coarsen factor for shrinking something high-res to PPU resolution.
            """
            factor = len(high_res) // (res_factor * self.mspec.ppu)
            return int(factor)

        aux = load_elevation()
        coarsen = {
            names.lat: _coarsen(aux[names.lat]),
            names.lon: _coarsen(aux[names.lon]),
        }

        aux = aux.coarsen(coarsen, boundary="trim").mean()
        return aux, "all"

    @abstractmethod
    def _wandb_config(self) -> dict:
        return

    def _init_log(self):
        if self.out.wandb:
            # Set this in case we're running on HPC where we can't run login command.
            os.environ["WANDB_API_KEY"] = keys.WANDB_API_KEY
            name = self.out.wandb_name or self.wandb_name
            self.wandb = wandb.init(
                project="climate-sim2real",
                config=self._wandb_config(),
                name=name,
                reinit=True,
            )

    def _log(self):
        self.compute_additional_metrics()
        self.metrics["best_val_loss"] = self.best_val_loss
        self.pbar.set_postfix(self.metrics)
        if self.out.wandb:
            self.wandb.log(self.metrics)

    def compute_additional_metrics(self):
        """
        Add additional custom entries to self.metrics that get logged.
        """
        return

    def overfit_train(self):
        """
        Overfit to just one task to see if the model works.
        """
        epoch_losses = []

        self.optimiser = optim.Adam(self.model.model.parameters(), lr=self.opt.lr)

        self.pbar = tqdm(range(self.start_epoch, self.opt.num_epochs + 1))

        if self.start_epoch == 1:
            val_loss = self.compute_loglik(self.cv_loader)
            self.metrics[names.val_loss] = val_loss
            self.metrics[names.train_loss] = val_loss
            self.metrics[names.epoch] = 0
            self._log()

        for epoch in self.pbar:
            batch_losses = []
            for i in range(self.opt.batches_per_epoch):
                date = self.val_set.times[0]
                test_date = pd.Timestamp(date)
                task = self.task_loader(test_date, context_sampling=("all", "all"))
                if i == 0 and self.out.plots:
                    self.plot_prediction(name=f"epoch_{epoch}_test", task=task)
                try:
                    batch_loss = self.train_on_batch(task)
                except Exception as e:
                    print(e)
                    if self.out.plots:
                        self.plot_prediction(
                            name=f"epoch_{epoch}_test_broken", task=task
                        )
                    return

                batch_losses.append(batch_loss)

            train_loss = np.mean(batch_losses)
            epoch_losses.append(train_loss)
            val_lik = self.compute_loglik(self.cv_loader)
            self.metrics[names.val_loss] = val_lik
            self.metrics[names.train_loss] = train_loss
            self.metrics[names.epoch] = epoch
            self._log()
            self._save_weights(epoch, val_lik)

    def plot_sample_tasks(self, epoch):
        for task in self.sample_tasks:
            time = task["time"]
            self.plot_prediction(task=task, name=f"epoch_{epoch}_{time}")

    def plot_receptive_field(self):
        receptive_field(
            self.model.model.receptive_field,
            self.data_processor,
            ccrs.PlateCarree(),
            [*self.data.bounds.lon, *self.data.bounds.lat],
        )

        plt.gca().set_global()

    def plot_example_task(self, task=None):
        if task is None:
            task = self.sample_tasks[0]
        fig = context_encoding(self.model, task, self.task_loader)
        save_plot(self.exp_dir, "example_task_model_inputs", fig)

    def context_target_plot(self):
        fig, axs = init_fig()
        ax = axs

        offgrid_context(
            ax,
            self.sample_tasks[0],
            self.data_processor,
            self.task_loader,
            plot_target=True,
            add_legend=True,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )

        save_plot(self.exp_dir, "example_task_context_target", fig)

    @abstractmethod
    def _plot_X_t(self):
        """
        For plotting, which target points should be predicted?
        Returns pd.DataFrame/Series or xr.DataArray/DataSet
        """
        return

    @abstractmethod
    def plot_prediction(self, task=None, name=None):
        return
