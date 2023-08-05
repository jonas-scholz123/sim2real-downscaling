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

    def evaluate_era5_baseline(self, tspec: TuneSpec, added_var=0.15):
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

        nlls = self.evaluate_loglik(self.test_set, added_var=0.15)
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

    def predict(self, task, resolution_factor=1):
        if task is None:
            task = self.sample_tasks[0]
        else:
            task = copy.deepcopy(task)

        mean_ds, std_ds = self.model.predict(
            task, X_t=self.raw_aux, resolution_factor=resolution_factor
        )

        return mean_ds, std_ds

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
    num_samples = 512

    nums_stations = [20, 100]  # 4, 20, 100, 500?
    nums_tasks = [16, 80, 400, 2000, 10000]  # 400, 80, 16
    tuners = [TunerType.naive, TunerType.film]
    # tuners = [TunerType.naive]
    include_real_only = True
    e = Evaluator(paths, opt, out, data, model, tune, num_samples, False)

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
            # e.test_loader = e._init_testloader(tspec)
            # for task in iter(e.test_set):
            #   e.plot_prediction(task)
            # plt.show()
            # e.save()
        except FileNotFoundError as err:
            print(f"Not found: {err}")
            continue
        e.save()
    e.save()
    # %%
    e.res
    e._load_results()
    e.res
    # %%

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
    # %%
    e.res.set_index(["num_stations", "num_tasks", "tuner"])
    ## %%

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
    ## %%
    # axs[0, 0]

    # %%

    def lons_and_lats(df):
        lats = df.index.get_level_values(names.lat)
        lons = df.index.get_level_values(names.lon)
        return lons, lats

    t = replace(tune, num_tasks=10000, num_stations=500, era5_frac=0.0)
    # e._init_weights_era5_baseline()
    e._init_weights(t)
    e._init_testloader(t)
    task = e.test_set[0]

    # %%
    # %%
    e.context_target_plot()

    # %%

    df = pd.DataFrame()
    df["x1"] = [0, 1 / 200]
    df["x2"] = [0, 1 / 200]
    df = df.set_index(["x1", "x2"])
    df = e.data_processor.map(df, unnorm=True)
    lats = df.index.get_level_values("LAT")
    lons = df.index.get_level_values("LON")
    lat_lengthscale = lats[1] - lats[0]
    lon_lengthscale = lons[1] - lons[0]
    # %%
    from scipy.spatial.distance import cdist

    def task_smallest_sep(task):
        ys = task["X_c"][0][0, :]
        xs = task["X_c"][0][1, :]

        if len(ys) == 0 or len(xs) == 0:
            raise ValueError

        coordinates = list(zip(xs, ys))
        dm = cdist(coordinates, coordinates, metric="euclid")
        dm[dm == 0.0] += 100
        return dm.min()

    smallest_separations = []
    for i in range(100):
        try:
            smallest_separations.append(task_smallest_sep(e.train_set[i]))
        except:
            continue

    smallest_separations = sorted(smallest_separations)
    # smallest_separation = smallest_separations[int(len(smallest_separations) / 5)]
    smallest_separation = smallest_separations[0]

    # %%
    def artefact_plot(
        e,
        task,
        lat,
        lon,
        margin,
        ax=None,
        smallest_separation=None,
        plot_context=True,
        cbar=False,
        vmin=None,
        vmax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        da = e._add_aux(10)[0]

        sel = (
            (da["LAT"] < lat + margin)
            & (da["LAT"] > lat - margin)
            & (da["LON"] < lon + margin)
            & (da["LON"] > lon - margin)
        )

        da = da.where(sel, drop=True)

        ys = task["X_c"][0][0, :]
        xs = task["X_c"][0][1, :]
        mean_ds, std_ds = e.model.predict(
            task, X_t=da, resolution_factor=1, unnormalise=False
        )

        # Unnormalise.
        m, s = e.data_processor.norm_params["T2M"].values()
        mean_ds["T2M"] = (mean_ds["T2M"] * s) + m

        im = mean_ds["T2M"].plot(
            ax=ax, cmap="coolwarm", add_colorbar=cbar, vmin=vmin, vmax=vmin
        )

        x1s = mean_ds.x1
        ax.set_yticks(np.arange(x1s.min(), x1s.max(), 1 / 200))
        ax.set_yticklabels([])

        x2s = mean_ds.x2
        ax.set_xticks(
            np.arange(x2s.min(), x2s.max(), 1 / 200),
        )
        ax.set_xticklabels([])

        ax.grid(linewidth=0.05, color="black")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        ax.tick_params("both", bottom=False, left=False)

        ax.set_aspect("equal")

        if plot_context:
            ax.scatter(
                xs,
                ys,
                color="none",
                edgecolor="k",
                linewidth=0.5,
                s=3**2 / margin,
                label="Context",
            )

        if smallest_separation is not None:
            circs = [
                plt.Circle(
                    (x, y),
                    smallest_separation,
                    color="k",
                    fill=False,
                    linestyle="--",
                    label="Shortest Signal" if i == 0 else None,
                )
                for i, (x, y) in enumerate(zip(xs, ys))
            ]

            for circ in circs:
                ax.add_patch(circ)

        return da, im

    # %%
    num_times = 5
    fig, axs = plt.subplots(2, num_times, figsize=(2.2 * num_times, 5))
    seed = 44
    np.random.seed(seed)
    times = np.random.choice(e.test_set.times, num_times)

    dense = (50.7, 10.2)
    for i, time in enumerate(times):
        sparse_task = e.test_set.task_loader(
            time, context_sampling=[60, "all"], seed_override=seed
        )
        dense_task = e.test_set.task_loader(
            time, context_sampling=["all", "all"], seed_override=seed
        )

        sparse = (53.35, 10.4)
        _, im = artefact_plot(
            e,
            sparse_task,
            dense[0],
            dense[1],
            1.5,
            smallest_separation=None,
            plot_context=True,
            ax=axs[0, i],
        )

        vmin = im.norm._vmin
        vmax = im.norm._vmax

        if True:
            aux = sparse_task["Y_c"][1]
            aux[0] *= 0
            aux[0] += 0.0
            aux[1] *= 0
            aux[1] += 0.4
            aux[2] *= 0
            aux[2] += 0.4

        _, im = artefact_plot(
            e,
            sparse_task,
            dense[0],
            dense[1],
            1.5,
            smallest_separation=None,
            plot_context=True,
            ax=axs[1, i],
        )
        axs[0, i].set_title(time)
    axs[0, 0].set_ylabel("Aux Data visible", fontsize=16)
    axs[1, 0].set_ylabel("Aux Data hidden", fontsize=16)
    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    save_plot(None, "artefacts_hide_aux", ext="png", dpi=200)
    # %%
    e._init_weights_era5_baseline()
    e.plot_prediction(e.train_set[20], cmap="jet")
    save_plot(None, "sample_prediction", ext="png", dpi=300)

    # %%
    # %%
    truth = e.get_truth(sparse_task["time"])["T2M"]
    truth = e.data_processor.map(truth)

    m, s = e.data_processor.norm_params["T2M"].values()
    truth = (truth * s) + m
    # %%

    fig, ax = plt.subplots(1, 1)

    _, im = artefact_plot(
        e,
        sparse_task,
        sparse[0],
        sparse[1],
        1.0,
        smallest_separation=smallest_separation,
        plot_context=True,
        ax=ax,
    )

    vmin = im.norm._vmin
    vmax = im.norm._vmax

    ax.set_aspect("equal")
    ax.scatter(
        truth.index.get_level_values("x2"),
        truth.index.get_level_values("x1"),
        s=1.5**2,
        c=truth[names.temp],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    # %%
    from pprint import pprint

    # %%
    elev = artefact_plot(
        e,
        sparse_task,
        dense[0],
        dense[1],
        1.5,
        smallest_separation=None,
        plot_context=True,
        ax=axs[0, i],
    )
    # mean_ds, std_ds = e.model.predict(sparse_task, X_t=truth, unnormalise=True)
    # %%
    margins = [1.2, 0.7]

    time = e.test_set[0]["time"]

    sparse_task = e.test_set.task_loader(
        time, context_sampling=[60, "all"], seed_override=seed
    )

    dense_task = e.test_set.task_loader(
        time, context_sampling=["all", "all"], seed_override=seed
    )
    # %%

    fig, axs = plt.subplots(
        2,
        2 * len(margins) + 1,
        figsize=(12, 7),
        gridspec_kw={"width_ratios": [1, 1, 0.2, 1, 1]},
    )

    axs[0, 2].remove()
    axs[1, 2].remove()

    # Dense:
    dense = (50.7, 10.2)
    # Sparse:
    sparse = (53.35, 10.4)

    if True:
        for task, start_i in zip([dense_task, sparse_task], [0, 3]):
            for i, margin in enumerate(margins, start_i):
                for j, (lat, lon) in enumerate([dense, sparse]):
                    if margin == margins[-1]:
                        sep = smallest_separation
                    else:
                        sep = None

                    elev = artefact_plot(
                        e,
                        task,
                        lat,
                        lon,
                        margin,
                        smallest_separation=sep,
                        ax=axs[j, i],
                        plot_context=True,
                    )

    axs[0, 1].legend(
        loc="center",
        bbox_to_anchor=[0.5, 0.5],
        bbox_transform=fig.transFigure,
        ncol=2,
    )
    axs[0, 0].set_ylabel("Dense-Station Region", fontsize=16)
    axs[1, 0].set_ylabel("Sparse-Station Region", fontsize=16)

    axs[0, 0].set_title("Dense Context Points")
    axs[0, 1].set_title("Zoomed In")

    axs[0, 3].set_title("Sparse Context Points")
    axs[0, 4].set_title("Zoomed In")

    plt.subplots_adjust(hspace=0.2, wspace=0.05)
    save_plot(None, "artefacts", ext="png", dpi=350)
    # %%
    # %%

    for i in range(10, 20):
        print(i)
        e.plot_prediction(e.train_set[i])

    # %%
    axs[0, 0].get_xlim()
    # %%

    mean_ds, std_ds = e.model.predict(
        task, X_t=e.raw_aux, resolution_factor=1, unnormalise=True
    )
    # %%
    fig, ax = plt.subplots(1, 1)
    mean_ds["T2M"].plot(ax=ax)
    # %%

    e = Evaluator(paths, opt, out, data, model, tune, num_samples)

    # %%
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

        fig.suptitle("")
        axs[0].set_xlabel("")
        return fig, axs

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    e._init_weights_era5_baseline()
    e.alps_plot(t, fig=fig, axs=axs[:, 0])

    e._init_weights(tspec)
    e.alps_plot(t, fig=fig, axs=axs[:, 1])

    fig.suptitle("")

    axs[1, 1].set_ylabel("")
    axs[0, 1].set_ylabel("")
    axs[0, 0].set_title("")
    axs[0, 1].set_title("")

    # save_plot(None, "alps", fig)
    # %%

    # %%
    def alps_plot():
        lo = 9
        hi = 48.3

        tspec = replace(
            tune,
            no_pretraining=False,
            num_tasks=10000,
            num_stations=20,
        )

        e.test_loader = e._init_testloader(tspec)
        t = e.test_set[3]
        fig, axs = plt.subplots(
            5,
            2,
            sharex=True,
            gridspec_kw={"height_ratios": [1.5, 0.8, 1, 1, 1]},
            figsize=(6, 4),
        )

        vmin = 2
        vmax = 13

        cmap = "plasma"
        # Add elevation data

        axs[1, 0].remove()
        axs[1, 1].remove()

        elev = e.raw_aux[names.height].where(
            (e.raw_aux[names.lon] > lo) & (e.raw_aux[names.lat] < hi), drop=True
        )

        im = axs[0, 0].imshow(elev)
        formatter = lambda x, _: f"{x:g}m"
        cbar = fig.colorbar(
            im, ax=axs[0, 0], orientation="horizontal", format=formatter
        )
        cbar.ax.set_xticks([500, 1500, 2500])

        e._init_weights_era5_baseline()
        mean, std = e.predict(t)
        temp = (
            mean[names.temp]
            .where((mean[names.lon] > lo) & (mean[names.lat] < hi), drop=True)
            .sel({names.time: t["time"]})
        )
        im = axs[0, 1].imshow(temp, cmap=cmap, vmin=vmin, vmax=vmax)

        formatter = lambda x, _: f"{x:g}°C"
        cbar = fig.colorbar(
            im, ax=axs[0, 1], orientation="horizontal", format=formatter
        )

        for ax in axs[0, :]:
            ax.set_xticks([])
            ax.set_yticks([])

        nums_tasks = [400, 10000]
        nums_stations = [20, 100, 500]

        for i, num_tasks in enumerate(nums_tasks, 0):
            for j, num_stations in enumerate(nums_stations, 2):
                tspec = replace(
                    tune,
                    no_pretraining=False,
                    num_tasks=num_tasks,
                    num_stations=num_stations,
                )
                e._init_weights(tspec, "best")
                mean, std = e.predict(t)
                temp = (
                    mean[names.temp]
                    .where((mean[names.lon] > lo) & (mean[names.lat] < hi), drop=True)
                    .sel({names.time: t["time"]})
                )

                ax = axs[j, i]
                axs[j, 0].set_ylabel(f"$N_{{s}}= {num_stations}$", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                im = ax.imshow(temp, cmap=cmap, vmin=vmin, vmax=vmax)

        for ax, num_times in zip(axs[2, :], nums_tasks):
            ax.set_title(f"$N_{{times}} = {num_times}$")

        axs[0, 0].set_title("Elevation")
        axs[0, 1].set_title("Sim Only")

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        save_plot(None, "alps_plot")

    alps_plot()

    # %%

    def results_plot():
        e._load_results()
        df = e.res
        nums_stations = [20, 100, 500]
        nums_tasks = [16, 80, 400, 2000, 10000]
        ylabels = ["Negative Log-Likelihood $\mathcal{L}$", "Mean Absolute Error"]

        fig, axss = plt.subplots(2, len(nums_stations), figsize=(6, 5), sharex=False)
        for j, (quantity, ylabel) in enumerate(zip(["nll", "mae"], ylabels)):
            axs = axss[j]
            for i, num_stations in enumerate(nums_stations):
                df = e.res[
                    (e.res["num_stations"] == num_stations)
                    & (e.res["pretrained"] == True)
                ]

                xs = np.array(range(len(nums_tasks)))
                xs_film = xs + 0.1

                film_ys = df[
                    df["num_tasks"].isin(nums_tasks)
                    & (df["tuner"] == str(TunerType.film))
                ].sort_values("num_tasks")[quantity]

                naive_ys = df[
                    df["num_tasks"].isin(nums_tasks)
                    & (df["tuner"] == str(TunerType.naive))
                ].sort_values("num_tasks")[quantity]

                sim_only = float(
                    e.res[
                        (e.res["num_stations"] == num_stations)
                        & (e.res["tuner"] == str(TunerType.none))
                    ][quantity]
                )
                # axs[i].axhline(sim_only, label="Sim Only", linestyle="--", color="r")
                axs[i].plot(xs, naive_ys, "x", label="Sim2Real")
                # axs[i].plot(xs_film, film_ys, "x", label="FiLM")
                axs[i].set_title(f"$N_{{stations}} = {num_stations}$")
                axs[i].set_xticks(range(len(nums_tasks)))
                axs[i].set_xticklabels(nums_tasks)

                axs[i].set_xlim(-0.5, len(nums_tasks) - 0.5)
                axss[1, i].set_xlabel("$N_{times}$")
            axs[0].set_ylabel(ylabel)
        axs[0].legend(
            loc="center",
            bbox_transform=fig.transFigure,
            bbox_to_anchor=(0.5, 0.5),
            ncol=2,
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.4)

    results_plot()
    # %%
    e._load_results()

    nums_stations = [20, 100, 500]
    nums_tasks = [16, 80, 400, 10000]

    fig, axs = plt.subplots(1, len(nums_stations), figsize=(6, 3))

    for i, num_stations in enumerate(nums_stations):
        df = e.res[e.res["tuner"] == str(TunerType.naive)]
        df = df[df["num_stations"] == num_stations]

        xs = np.array(range(len(nums_tasks)))
        xs_film = xs + 0.1

        real_only = df[
            df["num_tasks"].isin(nums_tasks) & (df["pretrained"] == False)
        ].sort_values("num_tasks")["nll"]

        sim2real = df[
            df["num_tasks"].isin(nums_tasks) & (df["pretrained"] == True)
        ].sort_values("num_tasks")["nll"]

        axs[i].plot(xs, sim2real, "x", label="Sim2Real")
        axs[i].plot(xs_film, real_only, "x", label="Real Only")
        axs[i].set_title(f"$N_{{stations}} = {num_stations}$")
        axs[i].set_xticks(range(len(nums_tasks)))
        axs[i].set_xticklabels(nums_tasks)
        axs[i].set_xlim(-0.5, len(nums_tasks) - 0.5)
        axs[i].set_xlabel("$N_{times}$")
        axs[0].set_ylabel("Negative Log-Likelihood $\mathcal{L}$")
        axs[i].legend()
    plt.tight_layout()
    save_plot(None, "results_sim2real_vs_real")
    # %%

    def superresolution_plot():
        # nums_stations = [20, 100, 500]
        nums_stations = [20, 100, 500]
        from deepsensor.plot import offgrid_context

        # High res features.

        # fig, axs = plt.subplots(2, len(nums_stations) + 1)
        fig, axs, transform = init_fig(
            2, len(nums_stations) + 1, (1.7 * (len(nums_stations) + 1), 5), True
        )
        axs = axs.reshape(2, len(nums_stations) + 1)

        mean_cmap = "coolwarm"
        std_cmap = "viridis_r"
        pad = 0.05

        # 1, 3, 4

        tspec = replace(
            tune,
            no_pretraining=False,
            num_tasks=10000,
            num_stations=20,
        )
        e.test_loader = e._init_testloader(tspec)
        task = e.test_set[9]
        e._init_weights_era5_baseline()
        mean, std = e.predict(task)

        mean = mean["T2M"].sel({"TIME": task["time"]})
        std = std["T2M"].sel({"TIME": task["time"]})
        mean.plot(
            ax=axs[0, 0],
            transform=transform,
            cbar_kwargs={"orientation": "horizontal", "pad": pad},
            cmap=mean_cmap,
            robust=True,
        )
        std.plot(
            ax=axs[1, 0],
            transform=transform,
            cbar_kwargs={"orientation": "horizontal", "pad": pad},
            cmap=std_cmap,
            robust=True,
        )

        axs[0, 0].set_title("Sim Only")
        axs[1, 0].set_title("")

        for i, num_stations in enumerate(nums_stations, 1):
            tspec = replace(
                tune,
                no_pretraining=False,
                num_tasks=10000,
                num_stations=num_stations,
            )
            e._init_weights(tspec)

            mean, std = e.predict(task)
            mean = mean["T2M"].sel({"TIME": task["time"]})
            std = std["T2M"].sel({"TIME": task["time"]})
            mean.plot(
                ax=axs[0, i],
                transform=transform,
                cbar_kwargs={"orientation": "horizontal", "pad": pad},
                cmap=mean_cmap,
                robust=True,
            )
            std.plot(
                ax=axs[1, i],
                transform=transform,
                cbar_kwargs={"orientation": "horizontal", "pad": pad},
                cmap=std_cmap,
                robust=True,
            )

            axs[0, i].set_title(f"$N_{{stations}} = {num_stations}$")
            axs[1, i].set_title("")

        for ax in axs.flatten():
            cbar = ax.collections[0].colorbar
            cbar.ax.set_xlabel("")
            cbar.ax.set_xticklabels(
                [f"{int(x)}°C" for x in cbar.get_ticks()]
            )  # set ticks of your format

        offgrid_context(
            axs[1, :],
            task,
            e.data_processor,
            transform=transform,
            add_legend=False,
            s=2**2,
            linewidth=0.4,
        )

        axs[0, 0].text(
            -0.07,
            0.55,
            "Temperature Mean $\\mu$",
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            transform=axs[0, 0].transAxes,
        )
        axs[1, 0].text(
            -0.07,
            0.55,
            "Temperature Std. Dev. $\\sigma$",
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            transform=axs[1, 0].transAxes,
        )

        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        save_plot(None, "superresolution", ext="png", dpi=350)

    superresolution_plot()
    # %%

    fig, axs, transform = init_fig(
        2, len(nums_stations) + 1, (1.7 * (len(nums_stations) + 1), 6), True
    )

    axs = axs.reshape(2, len(nums_stations) + 1)
    # %%
    ax.collections[0].colorbar
    # %%
    deepsensor.plot.context_encoding(e.model, e.train_set[5], e.task_loader)
    # %%

    task = e.train_set[5]
    e.plot_prediction(task)
    deepsensor.plot.feature_maps(e.model, task, 5, cmap="jet")
