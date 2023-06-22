from torch import nn
import torch
import lab as B


class FiLM(nn.Module):
    def __init__(self, n_features, freeze: bool = True) -> None:
        super().__init__()
        device = B.ActiveDevice.active_name
        self.scales = nn.parameter.Parameter(torch.zeros(n_features, device=device) + 1)
        self.biases = nn.parameter.Parameter(torch.zeros(n_features, device=device))

        # Freeze film layer during pretraining if wanted.
        self.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor):
        # 1st dim: batch size
        # 2nd dim: number of feature maps
        # 3rd dim: 1st dimension of 2d feature maps
        # 4th dim: 2nd dimension of 2d feature maps
        # We want to scale each feature map, so we broadcast tensors to the right shape.
        return self.scales[None, :, None, None] * x + self.biases[None, :, None, None]
