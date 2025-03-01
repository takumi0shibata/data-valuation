import torch
import torch.nn as nn
    

class ConcatenateLayer(nn.Module):
    def __init__(self, dim):
        super(ConcatenateLayer, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, self.dim)


class FeaturesModel(nn.Module):
    def __init__(
        self,
        readability_size: int = 35,
        linguistic_size: int = 51,
        num_labels=1
    ):
        super(FeaturesModel, self).__init__()
        self.linear = nn.Linear(readability_size+linguistic_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, inputs):
        output = self.linear(inputs)
        output = self.sigmoid(output)
        return output