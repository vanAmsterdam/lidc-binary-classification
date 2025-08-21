import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from preprocessingutils import pwr_transform
import os
from pathlib import Path

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dir', default='', type=str,
                    help='location of files, should contain subdirs with splits with subdirs imgs and masks; if empty, starts from parent dir of this folder')


def measure_slice(x, mask=None):
    '''
    calculate slice size and variance
    '''
    assert isinstance(x, np.ndarray)
    if mask is None:
        mask = np.ones_like(x)
    assert isinstance(mask, np.ndarray)

    size     = mask.sum()
    variance = x[np.nonzero(mask)].std() 
    img_max  = x[np.nonzero(mask)].max()
    img_min  = x[np.nonzero(mask)].min()
    img_mean = x[np.nonzero(mask)].mean()

    return size, variance, img_max, img_min, img_mean

args = parser.parse_args()
if args.dir == '':
    args.dir = Path(Path.cwd().parent / 'data' / 'nodules2d')
dfs = {}

imgs = os.listdir(os.path.join(args.dir, "imgs"))
imgs = [x for x in imgs if x.endswith('.png')]
df = pd.DataFrame({'img_name': imgs, 
                    'size': np.zeros((len(imgs, ))), 
                    'variance': np.zeros((len(imgs,))),
                    'max': np.zeros((len(imgs,))),
                    'min': np.zeros((len(imgs,))),
                    'mean': np.zeros((len(imgs,)))})

for i, img_name in tqdm(enumerate(imgs)):
    img  = np.array(Image.open(os.path.join(args.dir, "imgs", img_name))) / 255
    mask = (np.array(Image.open(os.path.join(args.dir, "masks", img_name))) / 255).astype(np.int16)

    size, variance, img_max, img_min, img_mean = measure_slice(img, mask)
    df.iloc[i, 1] = size
    df.iloc[i, 2] = variance
    df.iloc[i, 3] = img_max
    df.iloc[i, 4] = img_min
    df.iloc[i, 5] = img_mean
    
df = df.rename(columns={'img_name': 'name'})
# df["name"] = df.name.apply(lambda x: os.path.join("imgs", x))

# normalize values
# scalar_vars = ["size", "variance", "img_min", "img_max", "img_mean"]
# df[scalar_vars] = df[scalar_vars].apply(pwr_transform)

# print(df.head())
df.to_csv(os.path.join(args.dir, "measurements.csv"), index=False)



