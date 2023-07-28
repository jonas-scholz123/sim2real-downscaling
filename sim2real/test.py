# %%

import copy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import replace
from itertools import product
import torch
from torch.utils.data import DataLoader

import deepsensor.torch
from deepsensor.model.convnp import ConvNP
from deepsensor.model.nps import convert_task_to_nps_args
from deepsensor.plot import offgrid_context

import lab as B

from neuralprocesses.numdata import num_data
from neuralprocesses.dist import MultiOutputNormal

from tqdm import tqdm
from sim2real.datasets import load_station_splits, load_time_splits
from sim2real.plots import init_fig, save_plot
from sim2real.train.tune import Sim2RealTrainer
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
from sim2real.utils import (
    ensure_dir_exists,
    exp_dir_sim2real,
    exp_dir_sim,
    load_weights,
    sample_dates,
    sample_stations,
    weight_dir,
)

from sim2real.plots import adjust_plot

import shapely.vectorized
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import xarray as xr
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev, MeanStddev
import cartopy.feature as feature
import geopandas as gpd


def loglik(
    model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    batch_size=16,
    normalise=False,
    added_var=0.0,
    **kw_args,
):
    """Log-likelihood objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        batch_size (int, optional): Batch size to use for sampling. Defaults to 16.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """

    state = B.global_random_state(B.dtype(xt))
    float = B.dtype_float(yt)
    float32 = B.promote_dtypes(float, np.float32)

    # Sample in batches to alleviate memory requirements.
    logpdfs = None
    done_num_samples = 0
    while done_num_samples < num_samples:
        # Limit the number of samples at the batch size.
        this_num_samples = min(num_samples - done_num_samples, batch_size)

        # Perform batch.
        state, pred = model(
            state,
            contexts,
            xt,
            num_samples=this_num_samples,
            dtype_enc_sample=float,
            dtype_lik=float32,
            **kw_args,
        )

        pred = MultiOutputNormal.diagonal(pred.mean, pred.var + added_var, pred.shape)
        this_logpdfs = pred.logpdf(B.cast(float32, yt))

        # If the number of samples is equal to one but `num_samples > 1`, then the
        # encoding was a `Dirac`, so we can stop batching. Also, set `num_samples = 1`
        # because we only have one sample now. We also don't need to do the
        # `logsumexp` anymore.
        if num_samples > 1 and B.shape(this_logpdfs, 0) == 1:
            logpdfs = this_logpdfs
            num_samples = 1
            break

        # Record current samples.
        if logpdfs is None:
            logpdfs = this_logpdfs
        else:
            # Concatenate at the sample dimension.
            logpdfs = B.concat(logpdfs, this_logpdfs, axis=0)

        # Increase the counter.
        done_num_samples += this_num_samples

    # Average over samples. Sample dimension should always be the first.
    logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)

    if normalise:
        # Normalise by the number of targets.
        logpdfs = logpdfs / B.cast(float32, num_data(xt, yt))

    B.set_global_random_state(state)
    return logpdfs


class Evaluator(Sim2RealTrainer):
    def __init__(self, paths, opt, out, data, mspec, tspec, num_samples, load=True):
        super().__init__(paths, opt, out, data, mspec, tspec)
        self.results_path = paths.test_results
        self.num_samples = num_samples
        self._load_results(load)

    def _init_weights_era5_baseline(self):
        # Load weights
        exp_dir = exp_dir_sim(self.mspec)
        best_path = f"{weight_dir(exp_dir)}/best.h5"

        self.model.model, _, _ = load_weights(self.model.model, best_path)
        print(f"Loaded best ERA5 weights from {best_path}.")
        self.model.model = self.model.model.to(self.opt.device)

    def evaluate_era5_baseline(self, tspec: TuneSpec, added_var=0.0):
        self._init_weights_era5_baseline()

        self.test_loader = self._init_testloader(tspec)

        nlls = self.evaluate_loglik(self.test_set, added_var)

        # No station tasks were used for training.
        tspec.num_tasks = 0
        tspec.tuner = TunerType.none

        self._set_result(tspec, "nll", np.mean(nlls))
        self._set_result(tspec, "nll", np.mean(nlls))
        self._set_result(tspec, "nll_std", np.std(nlls) / np.sqrt(len(nlls)))

        df = self.deterministic_results(self.test_set)
        sqrt_N = np.sqrt(df.shape[0])
        mae = (df["T2M_pred"] - df["T2M_truth"]).abs().mean()
        self._set_result(tspec, "mae", mae)
        mae_std = (df["T2M_pred"] - df["T2M_truth"]).abs().std() / sqrt_N
        self._set_result(tspec, "mae_std", mae_std)

        return df, nlls

    def evaluate_model(self, tspec: TuneSpec):
        # Load the right weights.
        self._init_weights(tspec)

        # And the right context stations.
        # test_loader = self._init_testloader(tspec)
        self.test_loader = self._init_testloader(tspec)

        nlls = self.evaluate_loglik(self.test_set)
        self._set_result(tspec, "nll", np.mean(nlls))
        self._set_result(tspec, "nll_std", np.std(nlls) / np.sqrt(len(nlls)))

        df = self.deterministic_results(self.test_set)
        sqrt_N = np.sqrt(df.shape[0])
        mae = (df["T2M_pred"] - df["T2M_truth"]).abs().mean()
        self._set_result(tspec, "mae", mae)
        mae_std = (df["T2M_pred"] - df["T2M_truth"]).abs().std() / sqrt_N
        self._set_result(tspec, "mae_std", mae_std)

        return df, nlls

    # def evaluate_loglik(self, test_loader):
    #    return self.compute_loglik(test_loader)

    def evaluate_loglik(self, test_set, added_var=0.0):
        with torch.no_grad():
            task_losses = []
            for task in tqdm(iter(test_set)):
                task_losses.append(
                    float(
                        self.loss_fn(task, normalise=True, added_var=added_var)
                        .detach()
                        .cpu()
                    )
                )

        return task_losses

    def loss_fn(self, task, num_lv_samples=8, normalise=False, added_var=0.0):
        task = ConvNP.check_task(task)
        context_data, xt, yt, model_kwargs = convert_task_to_nps_args(task)

        logpdfs = loglik(
            self.model.model,
            context_data,
            xt,
            yt,
            **model_kwargs,
            num_samples=num_lv_samples,
            normalise=normalise,
            added_var=added_var,
        )

        loss = -B.mean(logpdfs)

        return loss

        # logpdfs = loglik(self.model, )

    def deterministic_results(self, task_set):
        dfs = []
        for t in tqdm(iter(task_set)):
            df = self.deterministic_results_task(t)
            dfs.append(df)
        return pd.concat(dfs)

    def deterministic_results_task(self, task):
        # Get temperature at all target stations on the task date.
        truth = self.get_truth(task["time"], station_ids=self.test_stations)
        mean_ds, _ = self.model.predict(task, X_t=truth)
        truth.index = mean_ds.index
        return truth.join(mean_ds, lsuffix="_truth", rsuffix="_pred")

    def _load_results(self, load=True):
        if not load:
            self.res = pd.DataFrame(
                columns=["num_stations", "num_tasks", "tuner", "pretrained", "nll"]
            )
            return
        try:
            self.res = pd.read_csv(self.results_path)
            print(f"Loaded previous results from {self.results_path}")

            if "pretrained" not in self.res.columns:
                self.res["pretrained"] = True
        except FileNotFoundError:
            print("No previous results file exists, creating empty DataFrame.")
            self.res = pd.DataFrame(
                columns=["num_stations", "num_tasks", "tuner", "pretrained", "nll"]
            )
            return

    def save(self):
        ensure_dir_exists(self.results_path)
        self.res.to_csv(self.results_path, index=False)

    def _init_testloader(self, tune):
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

        self.test_dates = sample_dates(time_split, names.test, self.num_samples)

        train_stations = sample_stations(stat_split, names.train, n_train)
        val_stations = sample_stations(stat_split, names.val, n_val)

        # 1000 -> all stations labelled "TEST".
        test_stations = sample_stations(stat_split, names.test, 1000)

        self.train_stations = train_stations
        self.val_stations = val_stations
        self.test_stations = test_stations

        print("test checksum: ", sum(test_stations))
        print("val checksum: ", sum(val_stations))
        print("train checksum: ", sum(train_stations))

        self.test_set = self.gen_trainset(
            train_stations + val_stations,
            "all",
            test_stations,
            "all",
            self.test_dates,
            set_task_loader=True,
            deterministic=True,
            split=False,
        )

        self.val_set = self.gen_trainset(
            train_stations,
            "all",
            val_stations,
            "all",
            self.test_dates,
            deterministic=True,
        )

        return self._to_dataloader(self.test_set, self.num_samples)

    def _set_result(self, tspec: TuneSpec, key, val):
        self._set_result_inner(
            tspec.num_stations,
            tspec.num_tasks,
            str(tspec.tuner),
            not tspec.no_pretraining,
            key,
            val,
        )

    def _set_result_inner(self, num_stations, num_tasks, tuner, pretrained, key, val):
        df = self.res
        df = df[df["num_stations"] == num_stations]
        df = df[df["num_tasks"] == num_tasks]
        df = df[df["tuner"] == tuner]
        df = df[df["pretrained"] == pretrained]

        if not df.empty:
            idx = df.index[0]
            self.res.loc[idx, key] = val
            return

        # Otherwise, need to create new row.
        record = {
            key: val,
            "num_stations": num_stations,
            "num_tasks": num_tasks,
            "tuner": tuner,
            "pretrained": pretrained,
        }

        idx = 0 if self.res.empty else self.res.index.max() + 1
        self.res.loc[idx] = record

    def _init_weights(self, tspec, which="best"):
        exp_dir = exp_dir_sim2real(self.mspec, tspec)
        best_path = f"{weight_dir(exp_dir)}/{which}.h5"

        self.model.model, _, _ = load_weights(self.model.model, best_path)
        print(f"Loaded best weights from {best_path}.")
        self.model.model = self.model.model.to(self.opt.device)

    def plot_locations(self, station_idss, markers="osv", labels=None):
        fig, axs = init_fig()

        for i, station_ids in enumerate(station_idss):
            if labels is not None:
                label = labels[i]
            else:
                label = None
            self.dwd_raw.plot_stations(
                station_ids, markers[i], ax=axs[0], color=f"C{i}", label=label
            )
            axs[0].legend()
        return fig, axs

    def predict(self, task):
        if task is None:
            task = self.sample_tasks[0]
        else:
            task = copy.deepcopy(task)

        mean_ds, std_ds = self.model.predict(
            task, X_t=self.raw_aux, resolution_factor=1
        )

        return mean_ds, std_ds

    def alps_plot(self, task, fig=None, axs=None):
        lo = 9
        hi = 48.3

        if task is None:
            task = self.sample_tasks[0]
        else:
            task = copy.deepcopy(task)

        if fig is None:
            fig, axs = plt.subplots(2, 1, sharex=True)
        mean, std = self.predict(task)
        mean[names.temp].where(
            (mean[names.lon] > lo) & (mean[names.lat] < hi), drop=True
        ).plot(ax=axs[0], cmap="coolwarm")
        e.raw_aux[names.height].where(
            (e.raw_aux[names.lon] > lo) & (e.raw_aux[names.lat] < hi), drop=True
        ).plot(ax=axs[1], cmap="viridis")
        fig.suptitle("")
        axs[0].set_xlabel("")

        return fig, axs

    def var_offset_grid_search(self, t, ns):
        self._init_weights_era5_baseline()

        results = []
        stds = []
        for n in ns:
            self.test_loader = self._init_testloader(t)
            nlls = self.evaluate_loglik(self.test_set, n)
            result = np.mean(nlls)
            std = np.std(nlls)
            stds.append(std)
            results.append(result)
        return results, stds


def generate_tspecs(
    init_tspec: TuneSpec,
    nums_stations,
    nums_tasks,
    tuners,
    include_real_only: bool = False,
):
    tspecs = []

    # Cover ERA5 only baseline.
    if TunerType.none in tuners:
        for num_stations in nums_stations:
            # Make a copy of the initial tspec.
            tspec = replace(init_tspec)

            # Modify.
            tspec.no_pretraining = False
            tspec.num_stations = num_stations
            tspec.tuner = TunerType.none

            tspecs.append(tspec)

        tuners = [t for t in tuners if t != TunerType.none]

    if include_real_only:
        for num_stations, num_tasks in product(nums_stations, nums_tasks):
            # Make a copy of the initial tspec.
            tspec = replace(init_tspec)

            # Modify.
            tspec.no_pretraining = True
            tspec.num_stations = num_stations
            tspec.num_tasks = num_tasks
            tspec.tuner = TunerType.naive

            tspecs.append(tspec)

    for num_stations, num_tasks, tuner in product(nums_stations, nums_tasks, tuners):
        # Make a copy of the initial tspec.
        tspec = replace(init_tspec)

        # Modify.
        tspec.num_stations = num_stations
        tspec.num_tasks = num_tasks
        tspec.tuner = tuner
        tspec.no_pretraining = False

        tspecs.append(tspec)
    return tspecs

if __name__ == "__main__":
    num_samples = 32

    nums_stations = [500, 100, 20]  # 4, 20, 100, 500?
    nums_tasks = [16, 80, 400, 10000]  # 400, 80, 16
    # tuners = [TunerType.naive, TunerType.film, TunerType.none]
    tuners = [TunerType.none]
    include_real_only = True
    # %%
    e = Evaluator(paths, opt, out, data, model, tune, num_samples, False)

    ns = np.array([0.05, 0.1, 0.15, 0.2, 0.25])

    all_results, all_stds = [], []

    for num_stations in nums_stations:
        t = replace(
            tune,
            num_stations=num_stations,
            num_tasks=0,
            tuner=TunerType.none,
            era5_frac=0.0,
        )
        e._init_weights_era5_baseline()
        ns = [0.05, 0.1, 0.15, 0.2, 0.25]
        results, stds = e.var_offset_grid_search(t, ns)
        all_results.append(results)
        all_stds.append(stds)
    # %%

    # %%
    ns = np.array(ns)
    plt.figure(figsize=(6, 3))
    for i in range(len(all_results)):
        results = all_results[i]
        stds = all_stds[i]
        xs = ns + 0.003 * (i - 1)
        plt.errorbar(
            xs,
            results,
            yerr=np.array(stds) / np.sqrt(len(stds)),
            label=f"$N_{{stat}} = {nums_stations[i]}$",
            fmt=".",
        )
        adjust_plot()
    plt.ylabel("Negative Log Likelihood")
    plt.xlabel("Variance offset")
    plt.legend()
    save_plot(None, "Optimal Variance Offset")

    low = 1e-3
    high = 0.5
    tolerance = 0.1
    low_val = np.mean(e.evaluate_era5_baseline(t, low)[1])
    high_val = np.mean(e.evaluate_era5_baseline(t, high)[1])

    e._init_weights_era5_baseline()
    print(high_val, low_val)
    while abs(high_val - low_val) > tolerance:
        e.test_loader = e._init_testloader(t)
        mid = high - low / 2
        print(mid)
        mid_val = np.mean(e.evaluate_loglik(e.test_set, mid))
        print(mid_val)

        if high_val < low_val:
            low = mid
            low_val = mid_val
        else:
            high = mid
            high_val = mid_val
    # %%
    task = e.test_set[120]
    e.plot_prediction(task)
    # %%
    tspecs = generate_tspecs(
        tune,
        nums_stations,
        nums_tasks,
        tuners,
        include_real_only=include_real_only,
    )

    det_results = []
    nll_results = []
    for tspec in tqdm(tspecs):
        try:
            print(f"N={tspec.num_stations}, M={tspec.num_tasks}, Tuner={tspec.tuner}")
            if tspec.tuner == TunerType.none:
                det_result, nlls = e.evaluate_era5_baseline(tspec)
                det_results.append(det_result)
                nll_results.append(nlls)
                continue

            det_result, nlls = e.evaluate_model(tspec)
            det_results.append(det_result)
            nll_results.append(nlls)

            # e.plot_locations(
            #    [e.train_stations, e.val_stations, e.test_stations],
            #    labels=["Train", "Val", "Test"],
            # )
            # plt.show()
            # for task in iter(e.test_set):
            #    e.plot_prediction(task)
            # plt.show()
            # e.save()
        except FileNotFoundError as err:
            print(f"Not found: {err}")
            continue
        e.save()
    e.save()
    # %%
    e.res.set_index(["num_stations", "num_tasks", "tuner"])
    ## %%

    t = replace(tune, num_tasks=10000, num_stations=500, era5_frac=0.0)

    e._init_weights_era5_baseline()
    e._init_testloader(t)
    # e._init_weights(t, which="best")
    # %%
    import lab as B

    # B.epsilon = 1e-3
    # test_task = e.test_set[0]
    # %%
    # diffs = [el["T2M_pred"] - el["T2M_truth"] for el in det_results]
    # names = [f"N={s.num_stations} {str(s.tuner)[10:]}" for s in tspecs]
    # fig, ax = plt.subplots(1, 1)
    # ax.boxplot(diffs)
    # ax.set_xticklabels(names, rotation=60)
    # save_plot(None, "error_boxplots", fig=fig)
    ## %%
    ## fig, axs = plt.subplots(1, 3)
    #
    # lim = 2
    # for diff, name in zip(diffs[:lim], names[:lim]):
    #    plt.hist(diff, bins=30, label=name, alpha=0.5)
    # plt.legend()
    ## %%
    # fig, ax = plt.subplots(1, 1)
    # ax.hist(nll_results[1], bins=30)
    # ax.set_xlabel("NLL")
    # ax.set_ylabel("Count")
    # ax.set_title("NLL Distribution")
    # save_plot(None, "nll_hist_tuned", fig=fig)
    ## %%
    # e = Evaluator(paths, opt, out, data, model, tune, num_samples)
    ## %%
    # tspec = replace(tune, no_pretraining=False, num_tasks=10000, num_stations=500)
    #
    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # e.test_loader = e._init_testloader(tspec)
    # t = e.test_set[240]
    #
    # e._init_weights_era5_baseline()
    # e.alps_plot(t, fig=fig, axs=axs[:, 0])
    #
    # e._init_weights(tspec)
    # e.alps_plot(t, fig=fig, axs=axs[:, 1])
    #
    # fig.suptitle("")
    #
    # axs[1, 1].set_ylabel("")
    # axs[0, 1].set_ylabel("")
    # axs[0, 0].set_title("")
    # axs[0, 1].set_title("")
    #
    # save_plot(None, "alps", fig)
    ## %%
    # axs[0, 0]
    # %%


    # fig, axs, crs = init_fig(ret_transform=True)
    #
    # placement_plot(test_tasks[0], X_new_df, e.data_processor, crs, figsize=8, ax=axs[0])
    # plt.show()
    ## %%
    # deepsensor.plot.acquisition_fn(
    #    test_tasks[0], acquisition_fn_ds, X_new_df, e.data_processor, crs, cmap=cmap
    # )
