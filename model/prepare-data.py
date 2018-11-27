
import ptvsd
import pandas as pd
import numpy as np
import os
import shutil
import feather
import matplotlib.pyplot as plt
from tqdm import tqdm


IMG_SOURCE_PATH = "/local_scratch/wamsterd/data/lidc/allnods2D/"
IMG_DEST_PATH = "/local_scratch/wamsterd/git/lidc-representation/data"
DF_PATH = os.path.join("resources", "annotation_df.feather")
FILE_EXT = ".png"

# load annotation df
df_ann = feather.read_dataframe(DF_PATH)

print(df_ann.shape)
print(df_ann.head())


# check available imgs
source_img_fnames = os.listdir(IMG_SOURCE_PATH)
source_img_idx = [x[:-(len(FILE_EXT))] for x in source_img_fnames]
df_ann = df_ann[df_ann.nodule_idx.isin(source_img_idx)]
print(df_ann.shape)

measure_vars = ["calcification", "internalstructure", "lobulation", "malignancy", "margin", "sphericity", "spiculation", "sublety", "texture"]
group_vars = ["nodule_idx", "nodule_number", "patient_id", "scan_id", "patient_number"]
agg_dict = dict(zip(measure_vars, [["median", "mean", "min", "max", "var"]] * len(measure_vars)))
agg_dict["annotation_id"] = "count"

# df = df_ann.groupby(group_vars, as_index = False).agg(
#     {'spiculation': ['median', 'mean', 'min', 'max', 'var'], 
#      "annotation_id": 'count'}).rename(columns = {'annotation_id':'n_annotations'})
df = df_ann.groupby(group_vars, as_index = False).agg(
    agg_dict).rename(columns = {'annotation_id':'n_annotations'})

new_colnames = ["_".join(x).strip() for x in df.columns.values]
new_colnames = [x.rstrip("_") for x in new_colnames]
df.columns = new_colnames

print(df.head())
print(df.shape)


# plt.hist(df.spiculation_mean)


df["label"] = (df.spiculation_mean >= 2).astype(int)
df["id"] = df.nodule_idx
ids = df.id


np.random.seed(12345)
valid_prop = .0
test_prop  = .2
valid_size = int(len(ids) * valid_prop)
test_size = int(len(ids) * test_prop)

valid_ids = list(np.random.choice(ids, replace = False, size = valid_size))
test_ids  = list(np.random.choice(list(set(ids) - set(valid_ids)), size = test_size))
train_ids = list(set(ids) - (set(valid_ids +  test_ids)))
split_dict = dict(zip(train_ids + valid_ids + test_ids,
                     ["train"] *len(train_ids) + ["valid"]*len(valid_ids) + ["test"] * len(test_ids)))

df["split"] = df.id.map(split_dict)
# print(df.split.value_counts())
df["out_name"] = df.id.apply(lambda x: x + FILE_EXT)
# df["out_name"] = df.apply(lambda x: x["pid"] + "_" + x["voi_name"] + FILE_EXT, axis = 1)
df["out_dir"] = df.apply(lambda x: os.path.join(x["split"], x["out_name"]), axis = 1)
df["orig_path"] = df.out_name.apply(lambda x: os.path.join(IMG_SOURCE_PATH, x))
# df["label"] = df.voi_name.apply(lambda x: int(bool(body_regex.match(x))))
df.head()


if not os.path.isdir(IMG_DEST_PATH):
    os.makedirs(IMG_DEST_PATH)
    os.makedirs(os.path.join(IMG_DEST_PATH, "train"))
#     os.makedirs(os.path.join(IMG_DEST_PATH, "valid"))
    os.makedirs(os.path.join(IMG_DEST_PATH, "test"))

    
for i, row in tqdm(df.iterrows()):
    in_path = row["orig_path"]
    out_path = os.path.join(IMG_DEST_PATH, row["split"], row["out_name"])
    shutil.copy(in_path, out_path)


# print(nodule_df.pivot_table(values = "nodule_idx", index = "borderline", columns = "malignant", aggfunc='count', fill_value = 0))
# feather.write_dataframe(nodule_df, os.path.join("resources", "nodule_df.feather"))
df.to_csv(os.path.join("data", "labels.csv"), index = False)

