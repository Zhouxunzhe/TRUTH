from torch import nn


class LlavaProjector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.multi_modal_projector

    def forward(self, inputs):
        return self.model.forward(inputs)
