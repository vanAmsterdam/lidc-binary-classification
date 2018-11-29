# generate bootstrapped samples for simulated datatask

#%%
import torch
import pyro
import pyro.distributions as dist
import pandas as pd
import os
# from importlib import import_module

torch.manual_seed(12345)

distributiondict = {"Bernoulli": dist.Bernoulli,
                    "Normal": dist.Normal}
GEN_MODEL = "confounder_only"
N_SAMPLES = 1e02


#%% import model specification
model_df = pd.read_csv(os.path.join("experiments", "sims", GEN_MODEL + ".csv"))
# model_df["param_tuple"] = 

noise_vars = model_df.loc[model_df.type == "noise", "variable"]

print(model_df)

def sample_noise_vars(model):
    noise_vars = model[model.type == "noise"]
    samples = {}
    for _, var in noise_vars.iterrows():
        var_name = var["variable"]
        distribution = distributiondict[var["distribution"]]
        # distribution = import_module("pyro.distributions." + var["distribution"])
        # params = (var["param_1"], var["param_2"])
        params = (var["param_1"],)
        x = pyro.sample(var, distribution(*params))
        samples[var_name] = x 
    return samples



print(sample_noise_vars(model_df))

def sample_dependent_vars(model):
    dependent_vars = model[model.type == "dependent"]


