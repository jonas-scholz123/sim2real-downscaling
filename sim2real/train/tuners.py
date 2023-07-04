from sim2real.config import TunerType
from torch import nn

from sim2real.modules.film import FiLM


def film_tuner(model: nn.Module) -> nn.Module:
    model.requires_grad_(False)
    for i, module in enumerate(model.modules()):
        if type(module) == FiLM:
            module.requires_grad_(True)

    return model


def naive_tuner(model: nn.Module) -> nn.Module:
    # Tune everything...
    model.requires_grad_(True)

    # ...Except lengthscales.
    model.encoder.coder[2].requires_grad_(False)
    model.decoder[1].coder[0].requires_grad_(False)
    return model
