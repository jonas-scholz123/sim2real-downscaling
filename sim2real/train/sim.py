# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from typing import Tuple, Union

import deepsensor.torch
from deepsensor.data.utils import (
    construct_x1x2_ds,
    construct_circ_time_ds,
)
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.plot import context_encoding, offgrid_context

from sim2real.utils import exp_dir_sim

from sim2real import keys, utils
from sim2real.datasets import load_elevation, load_era5
from sim2real.plots import save_plot
from sim2real.modules import convcnp
from sim2real.train.taskset import Taskset
from sim2real.train.trainer import Trainer
import cartopy.crs as ccrs
import cartopy.feature as feature

from sim2real.config import (
    DataSpec,
    OptimSpec,
    OutputSpec,
    ModelSpec,
    Paths,
    paths,
    names,
    data,
    out,
    opt,
    model,
)


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


class SimTrainer(Trainer):
    def __init__(
        self,
        paths: Paths,
        opt: OptimSpec,
        out: OutputSpec,
        data: DataSpec,
        mspec: ModelSpec,
    ) -> None:
        super().__init__(paths, opt, out, data, mspec)

    def _get_data(self):
        context_points = []
        target_points = []

        self.var_raw, context, target = self._add_var()
        context_points.append(context)
        target_points.append(target)

        self.aux_raw, context = self._add_aux()
        context_points.append(context)

        self.raw = [self.var_raw, self.aux_raw]

        self.data_processor = self._init_data_processor()
        var, aux = self.data_processor(self.raw)

        # Add spatio-temporal data.
        x1x2_ds = construct_x1x2_ds(aux)
        aux["x1_arr"] = x1x2_ds["x1_arr"]
        aux["x2_arr"] = x1x2_ds["x2_arr"]
        times = self.var_raw[names.time].values
        dates = pd.date_range(times.min(), times.max(), freq="H")

        # Day of year.
        doy_ds = construct_circ_time_ds(dates, freq="D")
        aux["cos_D"] = doy_ds["cos_D"]
        aux["sin_D"] = doy_ds["sin_D"]

        # Time of day.
        tod_ds = construct_circ_time_ds(dates, freq="H")
        aux["cos_H"] = tod_ds["cos_H"]
        aux["sin_H"] = tod_ds["sin_H"]

        return [var, aux], [var], context_points, target_points

    def _get_exp_dir(self, mspec: ModelSpec):
        return exp_dir_sim(mspec)

    def _init_tasksets(self) -> Tuple[Taskset, Taskset, Taskset]:
        """
        Returns: (TaskLoader, TaskLoader, TaskLoader) representing
            (train, val, test) task loaders.
        """
        context, target, c_points, t_points = self._get_data()
        tl = TaskLoader(context, target, time_freq="H")
        self.task_loader = tl

        def taskset(dates, freq, deterministic):
            return Taskset(
                tl,
                c_points,
                t_points,
                self.opt,
                dates,
                freq,
                deterministic=deterministic,
            )

        train_set = taskset(self.data.train_dates, "H", False)
        val_set = taskset(self.data.train_dates, self.data.val_freq, True)
        test_set = taskset(self.data.train_dates, self.data.val_freq, True)

        return train_set, val_set, test_set

    def _add_var(
        self,
    ) -> Tuple[Union[xr.DataArray, pd.Series], Union[float, int], Union[float, int]]:
        era5 = load_era5()[names.temp]
        return era5, self.data.era5_context, self.data.era5_target

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

    def _add_aux(self) -> Tuple[Union[xr.DataArray, pd.Series], Union[float, int, str]]:
        def _coarsen(high_res, low_res):
            """
            Coarsen factor for shrinking something high-res to low-res.
            """
            factor = self.data.aux_coarsen_factor * len(high_res) // len(low_res)
            return int(factor)

        aux = load_elevation()
        coarsen = {
            names.lat: _coarsen(aux[names.lat], self.var_raw[names.lat]),
            names.lon: _coarsen(aux[names.lon], self.var_raw[names.lon]),
        }
        aux = aux.coarsen(coarsen, boundary="trim").mean()
        return aux, "all"

    def _plot_X_t(self):
        return self.var_raw

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


s = SimTrainer(paths, opt, out, data, model)
s.train()