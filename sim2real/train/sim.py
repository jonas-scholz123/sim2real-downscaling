# %%
from dataclasses import asdict
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Union

from deepsensor.data.utils import (
    construct_x1x2_ds,
    construct_circ_time_ds,
)
from deepsensor.data.loader import TaskLoader

from sim2real.utils import exp_dir_sim, ensure_exists
from sim2real.plots import save_plot
from sim2real import plots

from sim2real.datasets import load_era5
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
        self.wandb_name = None
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
        tl = TaskLoader(
            context,
            target,
            time_freq="H",
            discrete_xarray_sampling=not self.data.era5_interpolation,
        )
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
                split=self.data.era5_split,
                frac_power=self.data.frac_power,
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

    def _plot_X_t(self):
        return self.var_raw

    def plot_prediction(self, task=None, name=None):
        """
        Plot truth, predicted mean, std, errors in a row.
        """
        if task is None:
            task = self.sample_tasks[0]

        mean_ds, std_ds = self.model.predict(task, X_t=self._plot_X_t())
        mean_ds_dense, std_ds_dense = self.model.predict(task, X_t=self.aux_raw)

        coord_map = {
            names.lat: self.var_raw[names.lat],
            names.lon: self.var_raw[names.lon],
        }

        # Fix rounding errors along dimensions.
        mean_ds = mean_ds.assign_coords(coord_map)
        std_ds = std_ds.assign_coords(coord_map)
        err_da = mean_ds[names.temp] - self.var_raw

        sel = {names.time: task["time"]}
        era5_data = self.var_raw.sel(sel)
        mean_data = mean_ds_dense[names.temp].sel(sel)
        std_data = std_ds_dense[names.temp].sel(sel)
        error_data = err_da.sel(sel)

        fig = plots.plot_era5_prediction(
            era5_data,
            mean_data,
            std_data,
            error_data,
            self.data_processor,
            task,
        )

        if name is not None:
            ensure_exists(self.paths.out)
            save_plot(self.exp_dir, name, fig)
        else:
            plt.show()

        plt.close()
        plt.clf()

    def _wandb_config(self) -> dict:
        config = {
            "opt": asdict(self.opt),
            "data": asdict(self.data),
            "model": asdict(self.mspec),
        }
        return config


if __name__ == "__main__":
    s = SimTrainer(paths, opt, out, data, model)
    s.train()
