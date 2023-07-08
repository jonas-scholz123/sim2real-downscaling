from sim2real.config import TuneSpec, TunerType
from torch import nn

from sim2real.modules.film import FiLM


def film_tuner(model: nn.Module, tspec: TuneSpec) -> nn.Module:
    model.requires_grad_(False)
    for i, module in enumerate(model.modules()):
        if type(module) == FiLM:
            module.requires_grad_(True)

    return model


def naive_tuner(model: nn.Module, tspec: TuneSpec) -> nn.Module:
    # Tune everything...
    model.requires_grad_(True)

    # ...Except lengthscales.
    model.encoder.coder[2].requires_grad_(False)
    model.decoder[1].coder[0].requires_grad_(False)
    return model


def long_range_tuner(model: nn.Module, tspec: TuneSpec) -> nn.Module:
    level = tspec.frequency_level
    # Start: tune everything.
    model = naive_tuner(model)

    # Don't tune high-frequency layers
    for module in model.decoder[0].before_turn_layers[:level]:
        module.requires_grad_(False)
    for module in model.decoder[0].after_turn_layers[level:]:
        module.requires_grad_(False)
    return model
