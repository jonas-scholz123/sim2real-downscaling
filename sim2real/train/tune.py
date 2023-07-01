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
from sim2real.utils import (
    exp_dir_sim,
    exp_dir_sim2real,
    load_weights,
    weight_dir,
    split,
)
from sim2real.datasets import (
    DWDSTationData,
    load_elevation,
    load_station_splits,
    load_time_splits,
)


def sample_dates(time_split, set_name, num, seed=42):
    """
    Randomly sample num dates from time_split from the right set.
    """
    return (
        time_split[time_split[names.set] == set_name]
        .sample(num, random_state=seed)
        .index.sort_values()
    )


def sample_stations(station_split, set_name, num):
    """
    Deterministically take the first num stations in a predefined order.
    """
    return list(
        station_split[station_split[names.set] == set_name]
        .sort_values(names.order)
        .index[:num]
    )


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
        self.full = self.dwd_raw.full()
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

    def _to_deepsensor_df(self, df):
        return df.reset_index().set_index([names.time, names.lat, names.lon])[
            [names.temp]
        ]

    def _get_exp_dir(self, mspec: ModelSpec):
        return exp_dir_sim2real(mspec, self.tspec)

    def gen_trainset(
        self,
        c_stations,
        c_num,
        t_stations,
        t_num,
        times,
        set_task_loader: bool = False,
        deterministic: bool = False,
    ):
        c_df, _ = split(self.full, times, c_stations)
        t_df, _ = split(self.full, times, t_stations)

        c_df = self._to_deepsensor_df(c_df)
        t_df = self._to_deepsensor_df(t_df)

        # How many context/target points (i.e. stations) are used?
        context_points = [c_num]
        target_points = [t_num]

        # Add auxilliary data.
        aux, aux_context_points = self._add_aux()
        self.raw_aux = aux
        context_points.append(aux_context_points)

        # Normalise.
        c_df, t_df, aux = self.data_processor([c_df, t_df, aux])

        # Add spatio-temporal data.
        x1x2_ds = construct_x1x2_ds(aux)
        aux["x1_arr"] = x1x2_ds["x1_arr"]
        aux["x2_arr"] = x1x2_ds["x2_arr"]

        # NOTE: Can't just use times variable because the index name is still un-processed.
        dts = c_df.index.get_level_values("time").unique()

        # Day of year.
        doy_ds = construct_circ_time_ds(dts, freq="D")
        aux["cos_D"] = doy_ds["cos_D"]
        aux["sin_D"] = doy_ds["sin_D"]

        # Time of day.
        tod_ds = construct_circ_time_ds(dts, freq="H")
        aux["cos_H"] = tod_ds["cos_H"]
        aux["sin_H"] = tod_ds["sin_H"]

        tl = TaskLoader([c_df, aux], t_df, links=[(0, 0)], time_freq="H")

        if set_task_loader:
            # Need the task loader for inferring model etc.
            self.task_loader = tl

        return Taskset(
            tl,
            context_points,
            target_points,
            self.opt,
            datetimes=dts,
            deterministic=deterministic,
        )

    def _init_tasksets(self) -> Tuple[Taskset, Taskset, Taskset]:
        # Load all data.
        time_split = load_time_splits()
        stat_split = load_station_splits()

        # Split our restricted data into train/val sets.
        # n = num stations, m = num_times/num_tasks
        n_all = tune.num_stations
        n_val = int(tune.val_frac_stations * n_all)
        n_train = n_all - n_val

        m_all = tune.num_tasks
        m_val = int(tune.val_frac_times * m_all)
        m_train = m_all - m_val

        train_dates = sample_dates(time_split, names.train, m_train)
        val_dates = sample_dates(time_split, names.val, m_val)
        test_dates = time_split[time_split[names.set] == names.test].index

        train_stations = sample_stations(stat_split, names.train, n_train)
        val_stations = sample_stations(stat_split, names.val, n_val)

        # 1000 -> all stations labelled "TEST".
        test_stations = sample_stations(stat_split, names.test, 1000)

        # TODO: Think abt set_task_loader
        train = self.gen_trainset(
            train_stations,
            self.data.dwd_context,
            train_stations,
            self.data.dwd_target,
            train_dates,
            set_task_loader=False,
            deterministic=False,
        )

        val = self.gen_trainset(
            train_stations,
            "all",
            val_stations,
            "all",
            val_dates,
            set_task_loader=True,
            deterministic=True,
        )

        test = self.gen_trainset(
            train_stations + val_stations,
            "all",
            test_stations,
            "all",
            test_dates,
            set_task_loader=False,
            deterministic=True,
        )

        return train, val, test

    def _add_aux(self) -> Tuple[Union[xr.DataArray, pd.Series], Union[float, int, str]]:
        def _coarsen(high_res, low_res):
            """
            Coarsen factor for shrinking something high-res to low-res.
            """
            factor = self.data.aux_coarsen_factor * len(high_res) // len(low_res)
            return int(factor)

        lats = self.full[names.lat].unique()
        lons = self.full[names.lon].unique()

        aux = load_elevation()
        coarsen = {
            names.lat: _coarsen(aux[names.lat], lats),
            names.lon: _coarsen(aux[names.lon], lons),
        }
        aux = aux.coarsen(coarsen, boundary="trim").mean()
        return aux, "all"

    def _plot_X_t(self):
        return self.raw_aux


if __name__ == "__main__":
    s2r = Sim2RealTrainer(paths, opt, out, data, model, tune)
    s2r.plot_prediction()
    # s2r.plot_example_task()
    s2r.train()
