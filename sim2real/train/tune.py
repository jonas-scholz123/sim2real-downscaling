# %%
from typing import Tuple, Union
import xarray as xr
import pandas as pd

from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
import deepsensor.torch
from deepsensor.data.utils import construct_x1x2_ds, construct_circ_time_ds


from sim2real.config import (
    DataSpec,
    ModelSpec,
    OptimSpec,
    OutputSpec,
    Paths,
    TuneSpec,
    opt,
    out,
    model,
    data,
    names,
    paths,
    tune,
)
from sim2real.train.taskset import Taskset
from sim2real.train.trainer import Trainer
from sim2real.utils import exp_dir_sim, exp_dir_sim2real, load_weights, weight_dir
from sim2real.datasets import DWDSTationData, load_elevation


class Sim2RealTrainer(Trainer):
    def __init__(
        self,
        paths: Paths,
        opt: OptimSpec,
        out: OutputSpec,
        data: DataSpec,
        mspec: ModelSpec,
        tspec: TuneSpec,
    ) -> None:
        self.dwd_raw = DWDSTationData(paths)
        self.val_frac = 0.2
        self.tspec = tspec

        super().__init__(paths, opt, out, data, mspec)
        self._load_initial_weights()

    def _load_initial_weights(self):
        if self.loaded_checkpoint:
            # We've loaded a Sim2Real checkpoint and don't need anything further.
            return

        sim_exp_dir = exp_dir_sim(self.mspec)
        pretrained_path = f"{weight_dir(sim_exp_dir)}/best.h5"

        # We need to load the best pretrained model weights.
        self.model.model, self.best_val_loss, self.start_epoch = load_weights(
            self.model.model, pretrained_path
        )

        if self.best_val_loss == float("inf"):
            raise FileNotFoundError(
                "Could not load appropriate pre-trained weights for this model configuration."
            )

        print("Starting training from best pretrained.")
        self.loaded_checkpoint = True

    def _get_exp_dir(self, mspec: ModelSpec):
        return exp_dir_sim2real(mspec, self.tspec)

    def _dwd_to_taskset(
        self, dwd: DWDSTationData, deterministic, set_task_loader=False
    ):
        # Add dwd data.
        df = dwd.to_deepsensor_df()
        context_points = [self.data.dwd_context]
        target_points = [self.data.dwd_target]

        # Add auxilliary data.
        aux, aux_context_points = self._add_aux()
        context_points.append(aux_context_points)

        # Normalise.
        station_df, aux = self.data_processor([df, aux])

        # Add spatio-temporal data.
        x1x2_ds = construct_x1x2_ds(aux)
        aux["x1_arr"] = x1x2_ds["x1_arr"]
        aux["x2_arr"] = x1x2_ds["x2_arr"]
        dts = df.index.get_level_values(names.time).unique()
        dates = pd.date_range(dts.min(), dts.max(), freq="H")

        # Day of year.
        doy_ds = construct_circ_time_ds(dates, freq="D")
        aux["cos_D"] = doy_ds["cos_D"]
        aux["sin_D"] = doy_ds["sin_D"]

        # Time of day.
        tod_ds = construct_circ_time_ds(dates, freq="H")
        aux["cos_H"] = tod_ds["cos_H"]
        aux["sin_H"] = tod_ds["sin_H"]

        context = [station_df, aux]
        target = station_df
        tl = TaskLoader(context, target, links=[(0, 0)], time_freq="H")

        if set_task_loader:
            # Need the task loader for inferring model etc.
            self.task_loader = tl

        return Taskset(
            tl,
            context_points,
            target_points,
            self.opt,
            dts,
            freq="H",
            deterministic=deterministic,
        )

    def _init_tasksets(self) -> Tuple[Taskset, Taskset, Taskset]:
        train, val, test = self.dwd_raw.train_val_test_split(self.val_frac)
        # TODO: Specify model manually for more control. This might lead to bad PPU otherwise.
        train = self._dwd_to_taskset(train, False, set_task_loader=True)
        val = self._dwd_to_taskset(val, True)
        test = self._dwd_to_taskset(test, True)
        return train, val, test

    def _add_aux(self) -> Tuple[Union[xr.DataArray, pd.Series], Union[float, int, str]]:
        def _coarsen(high_res, low_res):
            """
            Coarsen factor for shrinking something high-res to low-res.
            """
            factor = self.data.aux_coarsen_factor * len(high_res) // len(low_res)
            return int(factor)

        idx = self.dwd_raw.to_deepsensor_df().index
        lats = idx.get_level_values(names.lat).unique()
        lons = idx.get_level_values(names.lon).unique()

        aux = load_elevation()
        coarsen = {
            names.lat: _coarsen(aux[names.lat], lats),
            names.lon: _coarsen(aux[names.lon], lons),
        }
        aux = aux.coarsen(coarsen, boundary="trim").mean()
        return aux, "all"

    def _plot_X_t(self):
        return self.dwd_raw.df


if __name__ == "__main__":
    s2r = Sim2RealTrainer(paths, opt, out, data, model, tune)
    # s2r.context_target_plot()
    s2r.plot_example_task()
    s2r.train()
