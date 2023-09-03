# %%
from sim2real.config import (
    DataSpec,
    ModelSpec,
    OptimSpec,
    OutputSpec,
    Paths,
    TuneSpec,
    TunerType,
    pretrain_opt as opt,
    out,
    model,
    data,
    names,
    paths,
    tune,
)

from sim2real.train.tune import Sim2RealTrainer

from dataclasses import replace

mlp_capacity = 128
nums_mlp_layers = [3]

# No pretraining.
exp_opt = replace(opt, start_from="best", num_epochs=300)

# Maximum number of real data.
exp_tune = replace(tune, num_stations=500, num_tasks=10000, no_pretraining=True)


for num_mlp_layers in nums_mlp_layers:
    mlp_channels = (mlp_capacity,) * num_mlp_layers
    exp_model = replace(model, aux_t_mlp_layers=mlp_channels, dim_aux_t=0)
    trainer = Sim2RealTrainer(paths, exp_opt, out, data, exp_model, exp_tune)
    trainer.train()

# %%
