# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.vectorized
import xarray as xr

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import replace

import deepsensor.torch
from deepsensor.plot import offgrid_context

import lab as B

from sim2real.config import opt, out, model, data, paths, tune
from sim2real.utils import ensure_dir_exists
from sim2real.plots import adjust_plot, init_fig, save_plot
from sim2real.test import Evaluator

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev, MeanStddev


def ger_mask(ds):
    lats = ds["LAT"].values
    lons = ds["LON"].values

    country = gpd.read_file(paths.shapefile)
    geom = country.geometry[0]

    y, x = np.meshgrid(lats, lons)
    in_shape = shapely.vectorized.contains(geom, x, y)
    mask = xr.DataArray(in_shape.T, dims=ds.dims, coords=ds.coords).astype(int)
    return mask


def save_acq_ds(acquisition_fn_ds, num_stations, tuned):
    finetuned_str = "_tuned" if tuned else ""
    fpath = f"{paths.active_learning_dir}/acq_{num_stations}{finetuned_str}.nc"
    ensure_dir_exists(fpath)
    acquisition_fn_ds.to_netcdf(fpath)


def load_acq_ds(num_stations, tuned):
    finetuned_str = "_tuned" if tuned else ""
    fpath = f"{paths.active_learning_dir}/acq_{num_stations}{finetuned_str}.nc"
    with xr.open_dataset(fpath) as ds:
        da = ds["acquisition_fn"]
    return da


def save_X_new_df(X_new_df, num_stations, tuned):
    finetuned_str = "_tuned" if tuned else ""
    fpath = f"{paths.active_learning_dir}/x_{num_stations}{finetuned_str}.csv"
    ensure_dir_exists(fpath)
    X_new_df.to_csv(fpath)


def load_X_new_df(num_stations, tuned):
    finetuned_str = "_tuned" if tuned else ""
    fpath = f"{paths.active_learning_dir}/x_{num_stations}{finetuned_str}.csv"
    return pd.read_csv(fpath).set_index("iteration")


def active_learning_run(e, num_stations, tuned):
    t = replace(tune, num_tasks=10000, num_stations=num_stations, era5_frac=0.0)

    if tuned:
        e._init_weights(t)
    else:
        e._init_weights_era5_baseline()
    e.test_loader = e._init_testloader(t)

    X_t = e.raw_aux
    X_s = e.raw_aux.coarsen({"LAT": 5, "LON": 5}, boundary="trim").mean()

    greedy_alg = GreedyAlgorithm(
        model=e.model,
        X_t=X_t,
        X_s=X_s,
        X_s_mask=ger_mask(X_s),
        X_t_mask=ger_mask(X_t),
        N_new_context=5,
    )

    test_tasks = [e.test_set[i] for i in range(10)]

    # acquisition_fn = Stddev(e.model)
    acquisition_fn = MeanStddev(e.model)
    if isinstance(acquisition_fn, Stddev):
        minmax = "max"
    else:
        minmax = "min"

    X_new_df, acquisition_fn_ds = greedy_alg(acquisition_fn, test_tasks, minmax)

    save_X_new_df(X_new_df, num_stations, tuned)
    save_acq_ds(acquisition_fn_ds, num_stations, tuned)


def placement_plot(
    task, X_new_df, data_processor, crs, extent=None, figsize=3, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": crs}, figsize=(figsize, figsize)
        )
    offgrid_context(
        ax,
        task,
        data_processor,
        s=3**2,
        linewidths=0.2,
        transform=crs,
        add_legend=False,
    )
    ax.scatter(
        *X_new_df.values.T[::-1],
        s=3**2,
        c="r",
        linewidths=1,
        marker="x",
        transform=crs,
        label="",
    )
    ax.legend(labels=["Existing", "Proposed"], loc="best")


# %%

if __name__ == "__main__":
    nums_stations = [500, 100, 20]

    e = Evaluator(paths, opt, out, data, model, tune, 1024, False)
    for num_stations in nums_stations:
        for tuned in [True, False]:
            print(f"N Stations = {num_stations}, Tuned = {tuned}")
            active_learning_run(e, num_stations, tuned)
# %%
e = Evaluator(paths, opt, out, data, model, tune, 1024, False)


# %%

fig, axs, crs = init_fig(ret_transform=True)
placement_plot(e.test_set[0], X_new_df, e.data_processor, out.data_crs, ax=axs[0])
#
# placement_plot(test_tasks[0], X_new_df, e.data_processor, crs, figsize=8, ax=axs[0])
# plt.show()
# %%


def acquisition_fn_plot(
    task,
    acquisition_fn_ds,
    X_new_df,
    data_processor,
    crs,
    axes,
    cmap="Greys_r",
    figsize=3,
    add_colorbar=True,
    max_ncol=5,
    **kwargs,
):
    if "time" in acquisition_fn_ds.dims:
        # Average over time
        acquisition_fn_ds = acquisition_fn_ds.mean("time")
    if "sample" in acquisition_fn_ds.dims:
        # Average over samples
        acquisition_fn_ds = acquisition_fn_ds.mean("sample")

    iters = acquisition_fn_ds.iteration.values
    if iters.size == 1:
        n_iters = 1
    else:
        n_iters = len(iters)
    ncols = np.min([max_ncol, n_iters])

    # axes = axes.ravel()
    if add_colorbar:
        min, max = acquisition_fn_ds.min(), acquisition_fn_ds.max()
    else:
        # Use different colour scales for each iteration
        min, max = None, None
    for i, iteration in enumerate(iters):
        ax = axes[i]
        acquisition_fn_ds.sel(iteration=iteration).plot(
            ax=ax, cmap=cmap, vmin=min, vmax=max, add_colorbar=False, **kwargs
        )

        ax.set_title(f"Iteration {iteration}")
        ax.scatter(
            *X_new_df.loc[slice(0, iteration)].values.T[::-1],
            s=5**2,
            c="r",
            marker="x",
            linewidths=1,
            label="Proposed",
            **kwargs,
        )

    offgrid_context(
        axes,
        task,
        data_processor,
        s=2**2,
        linewidths=0.3,
        colors=["black"],
        add_legend=False,
        **kwargs,
    )


def station_placement_plot(num_stations):
    fig, axs, transform = init_fig(2, 5, (10, 5.0), True)

    for i, tuned in enumerate([False, True]):
        X_new_df = load_X_new_df(num_stations, tuned)
        acquisition_fn_ds = load_acq_ds(num_stations, tuned)

        acquisition_fn_plot(
            e.test_set[0],
            acquisition_fn_ds,
            X_new_df,
            e.data_processor,
            transform,
            axs[5 * i : 5 * i + 5],
            transform=transform,
            add_colorbar=True,
        )
    axs[0].legend(
        labels=["Proposed", "Existing"],
        loc="lower center",
        bbox_to_anchor=[2.9, 1.15],
        ncol=2,
    )

    axs[0].text(
        -0.07,
        0.55,
        "Pretraining Only",
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        transform=axs[0].transAxes,
    )

    axs[5].text(
        -0.07,
        0.55,
        "Finetuned",
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        transform=axs[5].transAxes,
    )

    fig.suptitle(
        f"$N_{{stations}} = {num_stations}$", horizontalalignment="left", x=0.15, y=0.97
    )
    save_plot(None, f"sensor_placement_N_stat_{num_stations}", fig=fig)


for num_stations in [20, 100, 500]:
    station_placement_plot(num_stations)
# %%
e.plot_train_val()
