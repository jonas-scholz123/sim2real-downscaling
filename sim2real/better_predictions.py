# %%
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch

num_ctx = 45
num_target = 300


x_c_0 = torch.rand(1, 2, num_ctx)
aux_x = torch.rand(1, 1, 200)
aux_y = torch.rand(1, 1, 200)
x_c = (x_c_0, (aux_x, aux_y))

y_c_0 = 10 * torch.rand(1, 1, num_ctx)
y_c_aux = torch.rand(1, 7, 200, 200)

y_c = (y_c_0, y_c_aux)

x_t = torch.rand(1, 2, num_target)


def hold_one_out(x_c, y_c, num_groups):
    context = torch.concatenate([x_c[0], y_c[0]], axis=1)
    context = context[torch.randperm(context.size()[0])]
    splits = torch.tensor_split(context, num_groups, axis=-1)

    results = []
    for i in range(num_groups):
        # merged = torch.concatenate([s for j, s in enumerate(splits) if i != j], axis=-1)
        merged = splits[i]
        split_x = merged[:, :2, :]
        split_y = merged[:, -1:, :]

        # This splits' temperature measurements concatd with aux data.
        split_x_c = (split_x, x_c[1])
        split_y_c = (split_y, y_c[1])
        results.append((split_x_c, split_y_c))

    return results


def exclude_within_radius(x_t, x_c, radius):
    # Calculate pairwise distances between coordinates in x_t and x_c
    distances = torch.tensor(cdist(x_t[0, :, :].T, x_c[0][0, :, :].T))

    # Find the minimum distance for each coordinate in x_t to any coordinate in x_c
    min_distances = distances.min(axis=1)

    # Filter out coordinates in a that are within the radius of any coordinate in b
    return x_t[:, :, min_distances.values > radius]


def plot_context_target(x_c, x_t):
    plt.scatter(x_c[0][0, 0, :], x_c[0][0, 1, :])
    plt.scatter(x_t[0, 0, :], x_t[0, 1, :], s=0.4)
    plt.show()


plot_context_target(x_c, x_t)
plot_context_target(x_c, exclude_within_radius(x_t, x_c, 0.1))

splits = hold_one_out(x_c, y_c, 5)

xc_i, yc_i = splits[0]
# %%
exclude_within_radius(x_t, x_c, 0.1)
# %%
coords = zip(xc_i[0][0, 0, :], xc_i[0][0, 1, :])
{next(iter(coords)): "a"}
# %%
list(zip(x_t[0, 0, :], x_t[0, 1, :]))
# %%
x_t[0, 0, 0] = 3.0
x_t[0, 0, 0]

a = torch.zeros(1, 43)

a[0, 0] = 12
a
