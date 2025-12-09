import numpy as np
import primitives as p


class Conv1DLayer:
    # Pocitejme s tim, ze stride = 1 a padding = 0.
    def __init__(self, out_channels: int, kernel_size: int):
        self.kernels = None # Toto budou vahy konvolucni vrstvy
        raise NotImplementedError("Conv1DLayer.__init__ is not implemented")

    def forward(self, x: p.Tensor) -> p.Tensor:
        raise NotImplementedError("Conv1DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("Conv1DLayer.parameters is not implemented")


class MaxPool1DLayer:
    def __init__(self, pool_size: int, stride: int):
        raise NotImplementedError("MaxPool1DLayer.__init__ is not implemented")

    def forward(self, x: p.Tensor) -> p.Tensor:
        raise NotImplementedError("MaxPool1DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("MaxPool1DLayer.parameters is not implemented")
