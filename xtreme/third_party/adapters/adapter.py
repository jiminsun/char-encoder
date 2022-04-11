import torch.nn as nn


class Adapter(nn.Module):
    def __init__(
            self,
            hidden_size,
            bottleneck_size,
    ):
        super().__init__()
        self.down_projection = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.ReLU()
        self.up_projection = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input):
        h = self.down_projection(input)     # [B x L x H] -> [B x L x R]
        h = self.activation(h)
        h = self.up_projection(h)           # [B x L x R] -> [B x L x H]
        return h
