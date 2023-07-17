import os
from typing import Tuple
import pandas as pd
import torch
from sim2real.config import TuneSpec, paths, ModelSpec, opt, names, data
import numpy as np
from deepsensor.data.processor import DataProcessor


def ensure_dir_exists(fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)


def ensure_exists(dirpath):
    os.makedirs(dirpath, exist_ok=True)


def get_model_dir(m: ModelSpec):
    channel_str = str(m.unet_channels)[1:-1].replace(", ", "_")
    if m.film:
        channel_str += "film"
    return f"{paths.out}/{m.likelihood}/ppu_{m.ppu}_channels_{channel_str}_dimyt_{m.dim_yt}_dim_yc_{m.dim_yc}"


def exp_dir_sim(m: ModelSpec):
    model_dir = get_model_dir(m)
    path = f"{model_dir}/sim"
    ensure_exists(path)
    return path


def exp_dir_sim2real(m: ModelSpec, t: TuneSpec):
    model_dir = get_model_dir(m)

    sim2real_str = "real" if t.no_pretraining else "sim2real"

    path = f"{model_dir}/{sim2real_str}_N{t.num_stations}_M{t.num_tasks}/{t.tuner}"
    ensure_exists(path)
    return path


def weight_dir(exp_dir):
    path = f"{exp_dir}/weights"
    ensure_exists(path)
    return path


def save_model(
    model, objective_val, epoch, spec, path, torch_state=None, numpy_state=None
):
    torch.save(
        {
            "weights": model.state_dict(),
            "objective": objective_val,
            "epoch": epoch,
            "spec": spec,
            "torch_state": torch_state,
            "numpy_state": numpy_state,
        },
        path,
    )


def get_default_data_processor():
    x1_min, x1_max = data.bounds.lat
    x2_min, x2_max = data.bounds.lon

    return DataProcessor(
        time_name=names.time,
        x1_name=names.lat,
        x2_name=names.lon,
        x1_map=(x1_min, x1_max),
        x2_map=(x2_min, x2_max),
        norm_params=data.norm_params,
    )


def load_weights(model, path, loss_only=False):
    state = torch.load(path, map_location=opt.device)
    val_loss = state["objective"]
    if loss_only:
        return None, val_loss, None

    weights = state["weights"]
    model.load_state_dict(weights)
    return (model, val_loss, state["epoch"])


def split_df(df, dts, station_ids) -> Tuple[pd.DataFrame]:
    """
    Split a dataframe by BOTH datetimes and station ids.

    Returns: (split, remainder): pd.DataFrame
    """

    split = df.query(f"{names.station_id} in @station_ids and {names.time} in @dts")
    remainder = df.query(
        f"{names.station_id} not in @station_ids and {names.time} not in @dts"
    )

    return split, remainder


def sample_dates(time_split, set_name, num, seed=42):
    """
    Randomly sample num dates from time_split from the right set.
    """
    df = time_split[time_split[names.set] == set_name]

    if num > df.shape[0]:
        return df.index.sort_values()
    return df.sample(num, random_state=seed).index.sort_values()


def sample_stations(station_split, set_name, num):
    """
    Deterministically take the first num stations in a predefined order.
    """
    return list(
        station_split[station_split[names.set] == set_name]
        .sort_values(names.order)
        .index[:num]
    )
