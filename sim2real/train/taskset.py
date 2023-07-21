import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset
import numpy as np
from typing import Iterable, Tuple
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
        taskloader: TaskLoader,
        num_context,
        num_target,
        opt: OptimSpec,
        datetimes: Iterable[pd.Timestamp] = None,
        time_range: Tuple[str, str] = None,
        freq="H",
        deterministic=False,
        frac_power=1,
        split=False,
    ) -> None:
        """
        Must define either datetimes or time_range & freq.
        """
        if datetimes is not None:
            if len(datetimes) == 2:
                print(
                    """WARNING: only two dates were provided.
                    Did you mean to specify a time_range instead?"""
                )
            self.times = list(datetimes)
        else:
            self.times = list(pd.date_range(*time_range, freq=freq))
        self.num_context, self.num_target = deepcopy(num_context), deepcopy(num_target)
        self.task_loader = taskloader
        self.deterministic = deterministic
        self.seed = opt.seed + 1
        self.rng = np.random.default_rng(self.seed)
        self.frac_power = frac_power
        self.split = split

        if split:
            self.low, self.high = self.num_context[0]
            self.num_context[0] = "split"
            self.num_target = "split"

    def _map_num_context(self, num_context):
        """
        Map num_context specs to something understandable by TaskLoader.
        """
        if isinstance(num_context, list):
            return [self._map_num_context(el) for el in num_context]
        elif isinstance(num_context, tuple) and isinstance(num_context[0], int):
            return int(self.rng.integers(*num_context))
        else:
            return num_context

    def get_split_frac(self):
        # Sample between lower and upper frac and take power to drive towards lower end.
        return self.rng.uniform(self.low, self.high) ** self.frac_power

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        if idx == 0 and self.deterministic:
            # Reset rng for deterministic
            self.rng = np.random.default_rng(self.seed)
        # Random number of context observations
        num_context = self._map_num_context(self.num_context)
        date = self.times[idx]

        if self.split:
            split_frac = self.get_split_frac()
        else:
            # default.
            split_frac = 0.5

        task = self.task_loader(
            date,
            num_context,
            self.num_target,
            datewise_deterministic=self.deterministic,
            split_frac=split_frac,
        )

        return task
