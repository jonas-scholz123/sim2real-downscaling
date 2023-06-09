# %%
import copy
from dataclasses import asdict
from itertools import product
from typing import Dict, Tuple, Union
import xarray as xr
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import cache
import torch

from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
import deepsensor.torch
from deepsensor.data.utils import construct_x1x2_ds, construct_circ_time_ds
from deepsensor.plot import offgrid_context


from sim2real.config import (
    DataSpec,
    ModelSpec,
    OptimSpec,
    OutputSpec,
    Paths,
    TuneSpec,
    TunerType,
    opt,
    out,
    model,
    data,
    names,
    paths,
    tune,
)
from sim2real.plots import init_fig
from sim2real.train.taskset import Taskset
from sim2real.train.trainer import Trainer
from sim2real.train.tuners import film_tuner, long_range_tuner, naive_tuner
from sim2real.utils import (
    exp_dir_sim,
    exp_dir_sim2real,
    load_weights,
    sample_dates,
    sample_stations,
    weight_dir,
    split_df,
    ensure_exists,
)
from sim2real.plots import save_plot
from sim2real.datasets import (
    DWDSTationData,
    load_elevation,
    load_station_splits,
    load_time_splits,
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

        self.wandb_name = f"{tspec.tuner} N_stat={self.tspec.num_stations} N_tasks={self.tspec.num_tasks}"

        super().__init__(paths, opt, out, data, mspec)
        self._load_initial_weights()
        self._apply_tuner()

        self.plot_train_val()

    def _apply_tuner(self):
        if self.tspec.tuner == TunerType.naive:
            tuner = naive_tuner
        elif self.tspec.tuner == TunerType.film:
            tuner = film_tuner
        elif self.tspec.tuner == TunerType.long_range:
            tuner = long_range_tuner
        else:
            raise NotImplementedError(f"Tuner {self.tspec.tuner} not yet implemented.")

        self.model.model = tuner(self.model.model, self.tspec)

    def _load_initial_weights(self):
        if self.loaded_checkpoint:
            # We've loaded a Sim2Real checkpoint and don't need anything further.
            return

        sim_exp_dir = exp_dir_sim(self.mspec)
        pretrained_path = f"{weight_dir(sim_exp_dir)}/best.h5"

        print(f"Loading best pre-trained weights from: {pretrained_path}.")
        # We need to load the best pretrained model weights.
        try:
            self.model.model, _, _ = load_weights(
                self.model.model,
                pretrained_path,
            )
            self.best_val_loss = float("inf")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Not pre-trained weights available for this architecture."
            )

        self.start_epoch = 1
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
        split: bool = False,
        frac_power: int = 1,
    ):
        c_df, _ = split_df(self.full, times, c_stations)
        t_df, _ = split_df(self.full, times, t_stations)

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
            split=split,
            frac_power=frac_power,
        )

    def _init_tasksets(self) -> Tuple[Taskset, Taskset, Taskset]:
        # Load all data.
        time_split = load_time_splits()
        stat_split = load_station_splits()

        self.time_split = time_split
        self.stat_split = stat_split

        # Split our restricted data into train/val sets.
        # n = num stations, m = num_times/num_tasks
        n_all = tune.num_stations
        n_val = int(tune.val_frac_stations * n_all)
        n_train = n_all - n_val

        m_all = tune.num_tasks
        m_val = int(tune.val_frac_times * m_all)
        m_train = m_all - m_val

        self.train_dates = sample_dates(time_split, names.train, m_train)
        self.val_dates = sample_dates(time_split, names.val, m_val)
        self.test_dates = time_split[time_split[names.set] == names.test].index

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
            self.train_dates,
            set_task_loader=False,
            deterministic=False,
            split=True,
            frac_power=self.tspec.frac_power,
        )

        val = self.gen_trainset(
            train_stations,
            "all",
            val_stations,
            "all",
            self.val_dates,
            set_task_loader=True,
            deterministic=True,
            split=False,
        )

        # Train stations at val times.
        temporal_val = self.gen_trainset(
            train_stations,
            (0.9, 0.9),
            train_stations,
            # This is overwritten and doesn't matter:
            "split",
            self.val_dates,
            deterministic=True,
            split=True,
            frac_power=1,
        )

        spatial_val = self.gen_trainset(
            train_stations,
            "all",
            val_stations,
            "all",
            self.train_dates,
            deterministic=True,
            split=False,
        )

        test = self.gen_trainset(
            train_stations + val_stations,
            "all",
            test_stations,
            "all",
            self.test_dates,
            set_task_loader=False,
            deterministic=True,
            split=False,
        )

        self.train_stations = train_stations
        self.val_stations = val_stations
        self.test_stations = test_stations

        self.temporal_val_loader = self._to_dataloader(
            temporal_val, self.opt.batch_size_val
        )
        self.spatial_val_loader = self._to_dataloader(
            spatial_val, self.opt.batch_size_val
        )

        return train, val, test

    def _init_sample_tasks(self):
        tl = self.test_set.task_loader
        context_points = (self.tspec.num_stations, "all")
        context_points = "all"

        valid_dates = set(self.test_dates)

        tasks = []
        for dt in self.out.sample_dates:
            dt = pd.to_datetime(dt)
            if dt not in valid_dates:
                print(f"Warning: {dt} not in test set for sample tasks. Falling back.")
                tasks.append(self.test_set[0])
            else:
                tasks.append(tl(dt, context_points, "all"))
        return tasks

    def _plot_X_t(self):
        return self.raw_aux

    def plot_prediction(self, task=None, name=None):
        def lons_and_lats(df):
            lats = df.index.get_level_values(names.lat)
            lons = df.index.get_level_values(names.lon)
            return lons, lats

        if task is None:
            task = self.sample_tasks[0]
        else:
            task = copy.deepcopy(task)

        # Get temperature at all stations on the task date.
        truth = self.get_truth(task["time"])

        mean_ds, std_ds = self.model.predict(task, X_t=truth)
        # Fix rounding errors along dimensions.
        err_da = mean_ds[names.temp] - truth[names.temp]
        err_da = err_da.dropna()

        # Higher resolution prediction everywhere.
        mean_ds, std_ds = self.model.predict(
            task, X_t=self.raw_aux, resolution_factor=1
        )

        proj = ccrs.TransverseMercator(central_longitude=10, approx=False)

        fig, axs = plt.subplots(
            subplot_kw={"projection": proj},
            nrows=1,
            ncols=4,
            figsize=(10, 2.5),
        )

        transform = ccrs.PlateCarree()
        vmin, vmax = 0.9 * truth[names.temp].min(), 1.1 * truth[names.temp].max()

        s = 3**2

        cmap = "seismic"

        axs[0].set_title("Truth")
        im = axs[0].scatter(
            *lons_and_lats(truth),
            s=s,
            c=truth[names.temp],
            transform=transform,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        fig.colorbar(im, ax=axs[0])

        im = mean_ds[names.temp].plot(
            cmap=cmap,
            ax=axs[1],
            transform=transform,
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )
        axs[1].set_title("ConvNP mean")

        im = std_ds[names.temp].plot(
            cmap="Greys",
            ax=axs[2],
            transform=transform,
            extend="both",
        )
        axs[2].set_title("ConvNP std dev")

        axs[3].set_title("ConvNP error")

        biggest_err = err_da.abs().max()
        im = axs[3].scatter(
            *lons_and_lats(err_da),
            s=s,
            c=err_da,
            cmap=cmap,
            vmin=-biggest_err,
            vmax=biggest_err,
            transform=transform,
        )
        fig.colorbar(im, ax=axs[3])

        # Don't add context points to mean prediction.
        context_axs = [ax for i, ax in enumerate(axs) if i != 1]

        offgrid_context(
            context_axs,
            task,
            self.data_processor,
            s=3**2,
            linewidths=0.5,
            add_legend=False,
            transform=ccrs.PlateCarree(),
        )

        bounds = [*self.data.bounds.lon, *self.data.bounds.lat]
        for ax in axs:
            ax.set_extent(bounds, crs=transform)
            ax.add_feature(feature.BORDERS, linewidth=0.25)
            ax.coastlines(linewidth=0.25)

        if name is not None:
            ensure_exists(self.paths.out)
            save_plot(self.exp_dir, name, fig)
        else:
            plt.show()

        plt.close()
        plt.clf()

    def plot_train_val(self):
        fig, axs = init_fig()
        self.dwd_raw.plot_stations(
            self.train_stations, "o", "C0", ax=axs[0], label="Train"
        )
        self.dwd_raw.plot_stations(self.val_stations, "s", "C1", ax=axs[0], label="Val")
        axs[0].legend()
        save_plot(self.exp_dir, "train_val_stations", fig)

    def get_truth(self, dt, station_ids=None):
        df = self.dwd_raw.at_datetime(dt).loc[dt]
        if station_ids is not None:
            df = df.query(f"{names.station_id} in @station_ids")
        df = df.reset_index()[[names.lat, names.lon, "geometry", names.temp]]
        df = df.set_index([names.lat, names.lon])
        return df

    def _wandb_config(self) -> dict:
        config = {
            "opt": asdict(self.opt),
            "data": asdict(self.data),
            "model": asdict(self.mspec),
            "tune": asdict(self.tspec),
        }
        return config

    def compute_additional_metrics(self):
        """
        Add additional custom entries to self.metrics that get logged.
        """
        self.metrics[names.val_spatial_loss] = self.compute_loglik(
            self.spatial_val_loader
        )
        self.metrics[names.val_temporal_loss] = self.compute_loglik(
            self.temporal_val_loader
        )
        return


# TODO: Should use generate_tspecs.
def run_experiments(nums_stations, nums_tasks, tuners):
    tspec = tune
    for num_stations, num_tasks, tuner in product(nums_stations, nums_tasks, tuners):
        # TODO: Need to modify LR?

        tspec.num_stations = num_stations
        tspec.num_tasks = num_tasks
        tspec.tuner = tuner

        s2r = Sim2RealTrainer(paths, opt, out, data, model, tspec)
        s2r.train()


if __name__ == "__main__":
    nums_stations = [500, 100, 20]  # 4, 20, 100, 500?
    nums_tasks = [400]  # 400, 80, 16
    tuners = [TunerType.long_range]
    run_experiments(nums_stations, nums_tasks, tuners)
