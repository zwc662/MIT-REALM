from typing import Type

import flax.linen as nn


class SigmoidReshape(nn.Module):
    net_cls: Type[nn.Module]
    gamma: float

    @nn.compact
    def __call__(self, *args, **kwargs):
        out = self.net_cls()(*args, **kwargs)
        # (0, 1) -> (-1/1-gamma, 0)
        return -nn.sigmoid(out) / (1 - self.gamma)
