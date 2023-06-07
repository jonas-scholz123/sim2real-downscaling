import os
import torch
from sim2real.config import paths
import numpy as np


def ensure_dir_exists(fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)


def ensure_exists(dirpath):
    os.makedirs(dirpath, exist_ok=True)


def exp_dir_sim():
    # TODO: add model spec into path.
    path = f"{paths.out}/modelspec/sim"
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


def load_weights(model, path, loss_only=False):
    try:
        state = torch.load(path)
        val_loss = state["objective"]
        if loss_only:
            return None, val_loss, None

        weights = state["weights"]
        model.load_state_dict(weights)
        return (
            model,
            val_loss,
            state["epoch"],
            state["torch_state"],
            state["numpy_state"],
        )
    except (FileNotFoundError, KeyError):
        return model, float("inf"), 0, None, None
