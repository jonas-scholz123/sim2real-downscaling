# %%

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import replace
from itertools import product
import torch
from torch.utils.data import DataLoader

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
    load_weights,
    sample_dates,
    sample_stations,
    weight_dir,
)


class Evaluator(Sim2RealTrainer):
    def __init__(self, paths, opt, out, data, mspec, tspec, num_samples):
        super().__init__(paths, opt, out, data, mspec, tspec)
        self.results_path = paths.test_results
        self.num_samples = num_samples
        self._load_results()

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

    def evaluate_loglik(self, test_set):
        with torch.no_grad():
            task_losses = []
            for task in iter(test_set):
                task_losses.append(
                    float(self.model.loss_fn(task, normalise=True).detach().cpu())
                )

        return task_losses

    def deterministic_results(self, task_set):
        dfs = []
        for t in iter(task_set):
            df = self.deterministic_results_task(t)
            dfs.append(df)
        return pd.concat(dfs)

    def deterministic_results_task(self, task):
        # Get temperature at all target stations on the task date.
        truth = self.get_truth(task["time"], station_ids=self.test_stations)
        mean_ds, _ = self.model.predict(task, X_t=truth)
        truth.index = mean_ds.index
        return truth.join(mean_ds, lsuffix="_truth", rsuffix="_pred")

    def _load_results(self):
        try:
            self.res = pd.read_csv(self.results_path)
            print(f"Loaded previous results from {self.results_path}")
        except FileNotFoundError:
            print("No previous results file exists, creating empty DataFrame.")
            self.res = pd.DataFrame(
                columns=["num_stations", "num_tasks", "tuner", "nll"]
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

        return self._to_dataloader(self.test_set, self.num_samples)

    def _set_result(self, tspec: TuneSpec, key, val):
        df = self.res
        df = df[df["num_stations"] == tspec.num_stations]
        df = df[df["num_tasks"] == tspec.num_tasks]
        df = df[df["tuner"] == str(tspec.tuner)]

        if not df.empty:
            idx = df.index[0]
            self.res.loc[idx, key] = val
            return

        # Otherwise, need to create new row.
        record = {
            key: val,
            "num_stations": tspec.num_stations,
            "num_tasks": tspec.num_tasks,
            "tuner": str(tspec.tuner),
        }
        self.res = self.res.append(record, ignore_index=True)

    def _init_weights(self, tspec):
        exp_dir = exp_dir_sim2real(self.mspec, tspec)
        best_path = f"{weight_dir(exp_dir)}/best.h5"

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

out.wandb = False
e = Evaluator(paths, opt, out, data, model, tune, num_samples)
# %%
tspecs = generate_tspecs(tune, nums_stations, nums_tasks, tuners)

det_results = []
nll_results = []

for tspec in tqdm(tspecs):
    try:
        print(f"N={tspec.num_stations}, M={tspec.num_tasks}, Tuner={tspec.tuner}")
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
    except FileNotFoundError:
        continue
e.save()
# %%
diffs = [el["T2M_pred"] - el["T2M_truth"] for el in det_results]
names = [f"N={s.num_stations} {str(s.tuner)[10:]}" for s in tspecs]
fig, ax = plt.subplots(1, 1)
ax.boxplot(diffs)
ax.set_xticklabels(names, rotation=60)
save_plot(None, "error_boxplots", fig=fig)
# %%
# fig, axs = plt.subplots(1, 3)

lim = 2
for diff, name in zip(diffs[:lim], names[:lim]):
    plt.hist(diff, bins=30, label=name, alpha=0.5)
plt.legend()
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(nll_results[0], bins=30)
ax.set_xlabel("NLL")
ax.set_ylabel("Count")
ax.set_title("NLL Distribution")
save_plot(None, "nll_hist_tuned", fig=fig)
