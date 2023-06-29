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
from sim2real.plots import save_plot, init_fig
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
        context, target, c_points, t_points = self._get_data()
        tl = TaskLoader(context, target, time_freq="H")
        self.task_loader = tl

        def taskset(dates, freq, deterministic):
            return Taskset(
                tl,
                c_points,
                t_points,
                self.opt,
                time_range=dates,
                freq=freq,
                deterministic=deterministic,
            )

        train_set = taskset(self.data.train_dates, "H", False)
        val_set = taskset(self.data.cv_dates, self.data.val_freq, True)
        test_set = taskset(self.data.test_dates, self.data.val_freq, True)

        return train_set, val_set, test_set

    def _add_var(
        self,
    ) -> Tuple[Union[xr.DataArray, pd.Series], Union[float, int], Union[float, int]]:
        """
        Returns: (var, context_points, target_points)
        var: The pandas/xarray dataset representing the variable of interest e.g. Temperature.
        context_points: fraction or number of context points
        target_points: fraction or number of target points
        """
        era5 = load_era5()[names.temp]
        return era5, self.data.era5_context, self.data.era5_target

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


if __name__ == "__main__":
    s = SimTrainer(paths, opt, out, data, model)
    # s.plot_example_task()
    # s.context_target_plot()
    s.train()
