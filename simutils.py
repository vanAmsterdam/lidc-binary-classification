import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class LinearRegressionModel(nn.Module):
    def __init__(self, p, weights = None, bias = None):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        if weights is not None:
            self.linear.weight = Parameter(torch.Tensor([weights]))
        if bias is not None:
            self.linear.bias = Parameter(torch.Tensor([bias]))

    def forward(self, x):
        return self.linear(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, p, weights = None, bias = None):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        if weights is not None:
            self.linear.weight = Parameter(torch.Tensor([weights]))
        if bias is not None:
            self.linear.bias = Parameter(torch.Tensor([bias]))

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# model_modules["Logistic"](3, (1,1,1), 0).forward(torch.zeros([1,3]))

