import os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from PIL import Image
sys.path.append("./")
from preprocessingutils import pwr_transform

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--out-dir', default='', help='place to put the slices (leave empty for default)')
parser.add_argument('--in-dir', default='', help='input directory (leave empty for default)')
parser.add_argument('--min-size', default=20, type=int, help='minimal size of noduls in mm')
parser.add_argument('--splits', default='train,valid', help='which spits to create, separate with ,')
parser.add_argument('--manual-seed', default=12345, help='seed for generating splits')
parser.add_argument('--silent', action='store_false', dest='verbose', help='dont print stuff')
parser.add_argument('--valid_min', default=int(1000), type=int, help='minimal number of validation and test imgs, otherwise do 20%')
parser.add_argument('--no-imgs', action='store_false', dest='copy_imgs', help='dont copy, only update dataframe')
parser.add_argument('--out-size', default=70, type=int, help='out size, uses center crop, if None, no resizing will be done')
parser.add_argument('--test', action='store_true', help='do only a few imgs for testing the code')
parser.set_defaults(verbose=True, copy_imgs=True, test=False)

def get_center_crop_bbox(in_size, out_size):
    """
    get bounding box for center crop
    in_size is (width, height for PIL.Image)
    """

    center = np.array(in_size) / 2
    left   = int(center[0] - out_size / 2)
    right  = int(left + out_size)
    upper  = int(center[0] - out_size / 2)
    lower  = int(upper + out_size)
    
    return (left, upper, right, lower)

def main(args):
    # find location for resources
    resourcedir = Path.cwd().parent / 'resources'

    # load dataframe with annotation data per nodule, made in the step 'lidc-preprocessing'
    df_ann = pd.read_csv(resourcedir / "annotation_df.csv")

    # show source files
    imgs = os.listdir(os.path.join(args.in_dir, "imgs"))
    imgs = [x for x in imgs if x.endswith(".png")]
    if args.verbose:
        print(f"found {len(imgs)} files")

    # img files are like 0001n01a2s086.png
    # imgs = imgs[:10]
    pids      = [x.split("n")[0] for x in imgs]
    nods      = [re.search(r"(?<=n)\d+", x).group() for x in imgs]
    anns      = [re.search(r"(?<=a)\d", x).group() for x in imgs]
    zvals     = [re.search(r"(?<=s)\d+", x).group() for x in imgs]
    ann_ids   = [x.split("s")[0] for x in imgs]
    nod_ids   = [x.split("a")[0] for x in imgs]
    slice_ids = [x.split(".png")[0] for x in imgs]
    nodule_slice_ids = [f"{nod_id}s{zval}" for nod_id, zval in zip(nod_ids, zvals)]

    slice_df = pd.DataFrame({
        'in_name': imgs,
        'pid': pids,
        'nodule_idx': nods,
        "annotation_idx": anns,
        "annotation_id": ann_ids,
        "nodule_id": nod_ids,
        "zval": zvals,
        "slice_id": slice_ids,
        "nodule_slice_id": nodule_slice_ids 
    })

    # add max number of annotations per nodule
    annotation_counts = df_ann.groupby('nodule_id').nodule_id.count().reset_index(name="annotation_count")
    slice_df = pd.merge(slice_df, annotation_counts, on="nodule_id")
    max_annotation_count_pid = slice_df.groupby("pid").annotation_count.max().reset_index(name='max_ann_count_per_pid')
    slice_df = pd.merge(slice_df, max_annotation_count_pid, on="pid")

    slice_counts = slice_df.groupby(["nodule_id", "zval"]).size().reset_index(name="slice_count")
    slice_df = pd.merge(slice_df, slice_counts, on=["nodule_id", "zval"])
    slice_df["all_anns_agree"] = slice_df.slice_count == slice_df.max_ann_count_per_pid

    slice_df.to_csv(resourcedir / "slice_df.csv", index=False)

    # keep only those slices where all annotators included the slice in their segmentation
    df = slice_df[(slice_df.all_anns_agree)]

    # import measurements
    measurements = pd.read_csv(os.path.join(args.in_dir, "measurements.csv"))
    df = pd.merge(df, measurements, left_on="in_name", right_on="name")

    # keep only slices greater than the cutoff
    df = df[df["size"] > args.min_size]

    print(f"number of slices left: {len(df)}")

    slices_per_pid    = len(df) / len(df.pid.unique())
    # divide by 4 because only 1 of the annotations gets selected
    slices_per_nodule = (len(df) / len(df.nodule_id.unique())) / 4

    np.random.seed(args.manual_seed)
    VALID_PROP = 0.3
    TEST_PROP  = 0.0
    
    df["uid"] = df.nodule_id
    # df.set_index("slice_id", drop=False, inplace=True)
    uids = df['uid'].unique().tolist()

    # valid_size  = int(min(args.valid_min, int(len(uids) * VALID_PROP * slices_per_pid / 4)) / (slices_per_pid / 4))
    # test_size   = int(min(args.valid_min, int(len(uids) * TEST_PROP * slices_per_pid / 4)) / (slices_per_pid / 4))
    valid_size  = int(min(args.valid_min, int(len(uids) * VALID_PROP * slices_per_nodule)) / (slices_per_nodule))
    test_size   = int(min(args.valid_min, int(len(uids) * TEST_PROP * slices_per_nodule)) / (slices_per_nodule))

    test_uids  = list(np.random.choice(uids, replace = False, size = test_size))
    valid_uids = list(np.random.choice(list(set(uids) - set(test_uids)), size = valid_size))
    train_uids = list(set(uids) - (set(valid_uids +  test_uids)))
    split_dict = dict(zip(train_uids + valid_uids + test_uids,
                        ["train"] *len(train_uids) + ["valid"]*len(valid_uids) + ["test"] * len(test_uids)))

    df["split"] = df.uid.map(split_dict)

    # normalize continuous variables
    cont_vars = ["size", "variance", "min", "max", "mean"]
    train_idxs = np.where(df.uid.isin(train_uids))
    df[cont_vars] = df[cont_vars].apply(pwr_transform, train_ids=train_idxs)

    # average measurements over annotations, pick single slice per measurement
    df = df.groupby("nodule_slice_id").agg({
        'size': 'mean', 
        'variance': 'mean',
        "min": 'mean',
        "max": 'mean',
        "mean": 'mean',
        'in_name': 'first',
        'split': 'first',
    })

    df["name"] = df.in_name.apply(lambda x: os.path.join("imgs", x))

    if args.test:
        df = df.iloc[:10,]

    if args.out_size:
        print("resizing and saving images")

        # create output directories
        for split in args.splits.split(","):
            for subdir in ["imgs", "masks"]:
                if not os.path.isdir(os.path.join(args.out_dir, split, subdir)):
                    os.makedirs(os.path.join(args.out_dir, split, subdir))

        # crop and copy images
        for slice_id, row in tqdm(df.iterrows()):
            img = Image.open(os.path.join(args.in_dir, 'imgs', row['in_name']), 'r')
            img_crop = img.crop(get_center_crop_bbox(img.size, args.out_size))
            img_crop.save(os.path.join(args.out_dir, row["split"], "imgs", row["in_name"]))

            mask = Image.open(os.path.join(args.in_dir, 'masks', row['in_name']), 'r')
            mask_crop = mask.crop(get_center_crop_bbox(mask.size, args.out_size))
            mask_crop.save(os.path.join(args.out_dir, row["split"], "masks", row["in_name"]))

    else:
        if args.copy_imgs:
            print("copying images")
            for split in args.splits.split(","):
                for subdir in ["imgs", "masks"]:
                    if not os.path.isdir(os.path.join(args.out_dir, split, subdir)):
                        os.makedirs(os.path.join(args.out_dir, split, subdir))

            for slice_id, row in tqdm(df.iterrows()):
                shutil.copy(os.path.join(args.in_dir, 'imgs', row["in_name"]),
                            os.path.join(args.out_dir, row["split"], "imgs", row["in_name"]))
                shutil.copy(os.path.join(args.in_dir, 'masks', row["in_name"]),
                            os.path.join(args.out_dir, row["split"], 'masks', row["in_name"]))

    df.to_csv(os.path.join(args.out_dir, "labels.csv"), index=False)
    print(df.split.value_counts())

if __name__ == "__main__":
    args = parser.parse_args()
    if args.out_dir == '':
        args.out_dir = (Path.cwd().parent) / 'data' / 'slices'
    if args.in_dir == '':
        args.in_dir = (Path.cwd().parent) / 'data' / 'nodules2d'
    main(args)
