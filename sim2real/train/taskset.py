import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
import deepsensor.torch
from deepsensor.data.loader import TaskLoader

from sim2real.config import OptimSpec


class Taskset(Dataset):
    """
    Wrapper around deepsensor.TaskLoader that acts as an interface to
    pytorch DataLoaders.
    """

    def __init__(
        self,
        time_range: Tuple[str, str],
        taskloader: TaskLoader,
        num_context,
        num_target,
        opt: OptimSpec,
        freq="H",
        deterministic=False,
    ) -> None:
        self.dates = pd.date_range(*time_range, freq=freq)
        self.num_context, self.num_target = num_context, num_target
        self.task_loader = taskloader
        self.deterministic = deterministic
        self.seed = opt.seed + 1
        self.rng = np.random.default_rng(self.seed)

    def _map_num_context(self, num_context):
        """
        Map num_context specs to something understandable by TaskLoader.
        """
        if isinstance(num_context, list):
            return [self._map_num_context(el) for el in num_context]
        elif isinstance(num_context, tuple):
            return int(self.rng.integers(*num_context))
        else:
            return num_context

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        if idx == len(self) - 1 and self.deterministic:
            # Reset rng for deterministic
            self.rng = np.random.default_rng(self.seed)
        # Random number of context observations
        num_context = self._map_num_context(self.num_context)
        date = self.dates[idx]
        task = self.task_loader(
            date, num_context, self.num_target, deterministic=self.deterministic
        )
        return task
