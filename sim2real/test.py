# %%
from sim2real.datasets import DWDSTationData
from sim2real.config import paths, names, data, out
from sim2real.plots import init_fig
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    dwd_sd = DWDSTationData(paths)
    test_stations = pd.read_feather(paths.dwd_test_stations)[names.station_name]
    # train, val, test = dwd_sd.train_val_test_split()

    import deepsensor.torch
    from deepsensor.data.loader import TaskLoader
    from deepsensor.data.processor import DataProcessor
    from deepsensor.model.models import ConvNP

    dp = DataProcessor(
        time_name=names.time,
        x1_name=names.lat,
        x2_name=names.lon,
        x1_map=data.bounds.lat,
        x2_map=data.bounds.lon,
    )
    df = dwd_sd.to_deepsensor_df()
    # %%
    station_df = dp(df)

    tl = TaskLoader(context=[station_df], target=station_df, links=[(0, 0)])
    print(tl)

    model = ConvNP(dp, tl)
    task = tl(pd.to_datetime("2022-01-01"), "split", "split", split_frac=0.1)
    # %%
    fig, ax = init_fig(figsize=(7, 7))
    from deepsensor import plot

    # fig, ax = plt.subplots(1, 1)

    plot.offgrid_context(
        ax,
        task,
        dp,
        tl,
        plot_target=True,
        add_legend=True,
        linewidths=0.5,
        transform=out.data_crs,
    )
    plt.show()
    # %%
    task
