# generate bootstrapped samples for simulated datatask

#%%
HOME_PATH = "/local_scratch/wamsterd/git/lidc-representation/"
import os
# os.chdir(HOME_PATH)
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
# import simutils
# from simutils import *
# from importlib import import_module


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


torch.manual_seed(12345)

distributiondict = {"Bernoulli": dist.Bernoulli,
                    "Normal": dist.Normal}
model_modules = {
    "Linear": LinearRegressionModel,
    "Logistic": LogisticRegressionModel
}

GEN_MODEL = "confounder_only"
N_SAMPLES = 1e02


#%% import model specification
model_df = pd.read_csv(os.path.join(HOME_PATH, "experiments", "sims", GEN_MODEL + ".csv"))
model_df.set_index("variable", drop = False, inplace = True)
model_df["param_tuple"] = model_df[[x for x in model_df.columns if "param" in x]].apply(
    lambda x: (*x.dropna(),), axis = 1)
all_vars = model_df.variable.tolist()
n_vars = len(all_vars)
noise_vars = model_df.loc[model_df.type == "noise", "variable"]
dependent_vars = model_df.loc[model_df.type == "dependent", "variable"]

# setup dicts to go from variable name to column index and vice-versa
var2idx = dict(zip(all_vars, range(n_vars)))
idx2var = dict(zip(range(n_vars), all_vars))

# assert variable has variable_model iff variable_type == dependent
# assert ordering of structural assignments

def build_dataset(model, N = 100):
    vars = model.variable.tolist()
    n_vars = len(vars)
    X = torch.zeros([N, n_vars], requires_grad = False)

    for var, row in model.iterrows():
        column_idx = var2idx[var]
        if row["type"] == "noise":
            distribution = distributiondict[row["distribution"]]
            params = row["param_tuple"]
            fn = distribution(*params)
            X[:, column_idx] = fn.sample(torch.Size([N]))
        else:
            betas = model["b_"+var].values
            bias = row["param_1"]
            model_type = row["variable_model"]
            variable_model = model_modules[model_type](len(betas), betas, bias)
            distribution = row["distribution"]
            MU = variable_model.forward(X).squeeze()
            if distribution == "Normal":
                X[:, column_idx] = MU
            elif distribution == "Bernoulli":
                fn = distributiondict[distribution](MU)
                X[:, column_idx] = fn.sample().squeeze()
            
    Y = X[:, -1]
    X = X[:, :-1]

    return X, Y

print(build_dataset(model_df, 10))


