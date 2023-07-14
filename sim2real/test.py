# %%
from dataclasses import replace
from itertools import product
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import lab as B

from sim2real.config import (
    ModelSpec,
    OptimSpec,
    TuneSpec,
    TunerType,
    paths,
    Paths,
    names,
    model,
    tune,
    opt,
)
from sim2real.datasets import (
    DWDSTationData,
    load_elevation,
    load_station_splits,
    load_time_splits,
)
from sim2real.train.taskset import Taskset
from sim2real.utils import (
    ensure_dir_exists,
    exp_dir_sim2real,
    get_default_data_processor,
    load_weights,
    sample_dates,
    sample_stations,
    split_df,
    weight_dir,
)
from sim2real.modules import convcnp

from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
import deepsensor.torch
from deepsensor.data.utils import construct_x1x2_ds, construct_circ_time_ds
from deepsensor.plot import offgrid_context


def batch_loglik(model, tasks):
    with torch.no_grad():
        if not isinstance(tasks, list):
            tasks = [tasks]
        task_losses = []
        for task in tasks:
            task_losses.append(model.loss_fn(task, normalise=True))
        mean_batch_loss = B.mean(torch.stack(task_losses))

    return float(mean_batch_loss.detach().cpu().numpy())


def compute_loglik(model, loader):
    batch_losses = []
    for task in loader:
        batch_loss = batch_loglik(model, task)
        batch_losses.append(batch_loss)
    return np.mean(batch_losses)


class Evaluator:
    def __init__(
        self,
        paths: Paths,
        mspec: ModelSpec,
        tspec: TuneSpec,
        opt: OptimSpec,
        num_samples: int,
    ):
        self.mspec = mspec
        self.gen = torch.Generator(device=opt.device)
        self.opt = opt
        self.full = DWDSTationData(paths).full()
        self.results_path = paths.test_results
        self.num_samples = num_samples
        self.data_processor = get_default_data_processor()
        self._load_results()

        self._init_testloader(tspec)
        self.model = self._init_model(mspec)

    def _init_model(self, mspec):
        # Construct custom model.
        model_kwargs = dict(
            dim_yc=mspec.dim_yc,
            dim_yt=mspec.dim_yt,
            points_per_unit=mspec.ppu,
            likelihood=mspec.likelihood,
            unet_channels=mspec.unet_channels,
            encoder_scales=mspec.encoder_scales,
            decoder_scale=mspec.decoder_scale,
            encoder_scales_learnable=mspec.encoder_scales_learnable,
            decoder_scale_learnable=mspec.decoder_scale_learnable,
            film=mspec.film,
            freeze_film=mspec.freeze_film,
        )

        model = convcnp.from_taskloader(None, **model_kwargs)

        model = ConvNP(self.data_processor, self.task_loader, model)
        return model

    def _init_weights(self, tspec):
        exp_dir = exp_dir_sim2real(self.mspec, tspec)
        best_path = f"{weight_dir(exp_dir)}/best.h5"

        self.model.model, _, _ = load_weights(self.model.model, best_path)
        print(f"Loaded best weights from {best_path}.")
        self.model.model = self.model.model.to(self.opt.device)

    def _init_testloader(self, tune):
        # Load all data.
        time_split = load_time_splits()
        stat_split = load_station_splits()

        # Split our restricted data into train/val sets.
        # n = num stations, m = num_times/num_tasks
        n_all = tune.num_stations
        n_val = int(tune.val_frac_stations * n_all)
        n_train = n_all - n_val

        self.test_dates = time_split[time_split[names.set] == names.test].index[
            : self.num_samples
        ]

        train_stations = sample_stations(stat_split, names.train, n_train)
        val_stations = sample_stations(stat_split, names.val, n_val)

        # 1000 -> all stations labelled "TEST".
        test_stations = sample_stations(stat_split, names.test, 1000)

        test = self.gen_trainset(
            # This matches reality, where we split all available stations into train + val.
            train_stations,
            "all",
            val_stations,
            "all",
            self.test_dates,
            deterministic=True,
            split=False,
        )

        self.train_stations = train_stations
        self.val_stations = val_stations
        self.test_stations = test_stations

        return DataLoader(
            test,
            batch_size=self.num_samples,
            shuffle=False,
            # Don't turn into pytorch tensors. We just want the sampling functionality.
            collate_fn=lambda x: x,
            generator=self.gen,
        )

    def _to_deepsensor_df(self, df):
        return df.reset_index().set_index([names.time, names.lat, names.lon])[
            [names.temp]
        ]

    def _add_aux(self):
        def _coarsen(high_res):
            """
            Coarsen factor for shrinking something high-res to PPU resolution.
            """
            factor = len(high_res) // self.mspec.ppu
            return int(factor)

        aux = load_elevation()
        coarsen = {
            names.lat: _coarsen(aux[names.lat]),
            names.lon: _coarsen(aux[names.lon]),
        }

        aux = aux.coarsen(coarsen, boundary="trim").mean()
        return aux, "all"

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

    def _load_results(self):
        try:
            self.df = pd.read_csv(self.results_path)
            print(f"Loaded previous results from {self.results_path}")
        except FileNotFoundError:
            print("No previous results file exists, creating empty DataFrame.")
            self.df = pd.DataFrame(
                columns=["num_stations", "num_tasks", "tuner", "nll"]
            )
            return

    def save(self):
        ensure_dir_exists(self.results_path)
        self.df.to_csv(self.results_path, index=False)

    def evaluate_loglik(self, tspec):
        test_loader = self._init_testloader(tspec)
        return compute_loglik(self.model, test_loader)

    def evaluate_determinstic(self, metrics):
        pass

    def evaluate_model(self, tspec: TuneSpec):
        # Load the right weights.
        self._init_weights(tspec)
        nll = self.evaluate_loglik(tspec)
        self._set_result(tspec, "nll", nll)
        pass

    def _set_result(self, tspec: TuneSpec, key, val):
        df = self.df
        df = df[df["num_stations"] == tspec.num_stations]
        df = df[df["num_tasks"] == tspec.num_tasks]
        df = df[df["tuner"] == str(tspec.tuner)]

        if not df.empty:
            df[key] = val
            return

        # Otherwise, need to create new row.
        record = {
            key: val,
            "num_stations": tspec.num_stations,
            "num_tasks": tspec.num_tasks,
            "tuner": str(tspec.tuner),
        }
        self.df = self.df.append(record, ignore_index=True)


def generate_tspecs(init_tspec, nums_stations, nums_tasks, tuners):
    tspecs = []
    for num_stations, num_tasks, tuner in product(nums_stations, nums_tasks, tuners):
        # Make a copy of the initial tspec.
        tspec = replace(init_tspec)

        # Modify.
        tspec.num_stations = num_stations
        tspec.num_tasks = num_tasks
        tspec.tuner = tuner

        tspecs.append(tspec)
    return tspecs


num_samples = 5

nums_stations = [500, 100, 20]  # 4, 20, 100, 500?
nums_tasks = [400]  # 400, 80, 16
tuners = [TunerType.naive, TunerType.film, TunerType.long_range]

e = Evaluator(paths, model, tune, opt, num_samples)
# %%
tspecs = generate_tspecs(tune, nums_stations, nums_tasks, tuners)

for tspec in tqdm(tspecs):
    try:
        e.evaluate_model(tspec)
        e.save()
    except FileNotFoundError:
        continue

# %%
e.df
