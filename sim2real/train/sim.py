# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from typing import Tuple, Union
import cartopy.crs as ccrs
import cartopy.feature as feature

import deepsensor.torch
from deepsensor.data.utils import (
    construct_x1x2_ds,
    construct_circ_time_ds,
)
from deepsensor.data.loader import TaskLoader
from deepsensor.plot import offgrid_context

from sim2real.utils import exp_dir_sim, ensure_exists
from sim2real.plots import save_plot

from sim2real.datasets import load_elevation, load_era5
from sim2real.train.taskset import Taskset
from sim2real.train.trainer import Trainer

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
        print(coarsen)
        aux = aux.coarsen(coarsen, boundary="trim").mean()
        return aux, "all"

    def _plot_X_t(self):
        return self.var_raw

    def plot_prediction(self, task=None, name=None):
        if task is None:
            task = self.sample_tasks[0]

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


if __name__ == "__main__":
    s = SimTrainer(paths, opt, out, data, model)

    for t in s.sample_tasks:
        # s.plot_prediction(t)
        s.plot_example_task()

    # s.train()
