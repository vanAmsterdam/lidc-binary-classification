"""
helper script to create experiments using the pre-processed lidc data
current implementation: only 2d slices
"""

#%%
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

from argparse import ArgumentParser

def fetch_data(args):
    annotation_df = pd.read_csv(args.data_dir.parent / "annotation_df.csv")
    # fetch the consensus measurements
    measurement_df = pd.read_csv(args.data_dir / "measurements_consensus.csv")

    # calculate outcome by averaging over annotations
    outcomes = annotation_df.groupby("nodule_id")[args.outcome].mean()
    outcome_df = pd.DataFrame(outcomes)

    # binarize at median cutoff
    outcome_df['label'] = outcomes.apply(lambda x: 1 if x > outcomes.median() else 0)

    # from the measurement_df, take the first mid-slice
    mid_slice_df = measurement_df[measurement_df["is_middle_slice"]]
    # merge slice information into the outcome_df
    outcome_df = outcome_df.merge(mid_slice_df[["id", "nodule_id"]], on=["nodule_id"], how="inner")

    # grab the patient id from the nodule id (e.g. 0001n01, 0001 is the patient id)
    outcome_df["patient_id"] = outcome_df["nodule_id"].str.extract(r"^(\d+)n")[0]

    return outcome_df

def center_crop_image(img_path: Path, output_size=(70, 70)):
    img = Image.open(img_path)
    width, height = img.size
    crop_width, crop_height = output_size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    img = img.crop((left, top, right, bottom))
    return img

# def prep_split(args):
#     # Split the dataset into train/val/test sets
#     train_df, val_df, test_df = np.split(outcome_df.sample(frac=1, random_state=42), [int(.8 * len(outcome_df)), int(.9 * len(outcome_df))])
#     return train_df, val_df, test_df

#%%
def main():
    parser = ArgumentParser(description="Create experiments using pre-processed LIDC data")
    parser.add_argument("--data-dir", type=Path, default="data/nodules2d", help="Path to the pre-processed LIDC data directory")
    parser.add_argument("--output-dir", type=Path, default="experiments", help="Path to the output directory for experiment results")
    parser.add_argument("--outcome", type=str, default="malignancy", help="Outcome variable to use ")
    parser.add_argument("--tag", type=str, required=False, help="Tag for the experiment")
    parser.add_argument('--out-size', default=70, type=int, help='out size, uses center crop, if None, no resizing will be done')
    parser.add_argument('--stack-images', action='store_true', help='Whether to stack images into a single array')

    args = parser.parse_known_args()[0]
    args = parser.parse_args()

    print(f"preparing experiment for outcome {args.outcome}")

    outcome_df = fetch_data(args)

    # create output directory
    if args.tag is None:
        tag = args.outcome
    else:
        tag = args.tag

    exp_dir = args.output_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    # if cropping, copy all cropped images to new directory
    if args.out_size is not None:
        img_list = []
        img_dir = args.data_dir / "imgs"
        cropped_dir = exp_dir / "cropped_images"
        cropped_dir.mkdir(parents=True, exist_ok=True)

        # get ids from outcome_df
        img_ids = outcome_df['id'].values
        outcome_df['img_path'] = [cropped_dir / f"{img_id}.png" for img_id in img_ids]
        print(f"cropping and copying {len(img_ids)} images to {cropped_dir}")
        for img_id in tqdm(img_ids):
            img_path = img_dir / f"{img_id}.png"
            cropped_img = center_crop_image(img_path, output_size=(args.out_size, args.out_size))
            cropped_img.save(cropped_dir / f"{img_id}.png")
            if args.stack_images:
                # convert to numpy array and save as .npy file
                img_list.append(np.array(cropped_img))

        if args.stack_images:
            imgs = np.stack(img_list, axis=0, dtype=np.uint8)
            labels = outcome_df['label'].values
            np.savez_compressed(exp_dir / "imgs_labels.npz", imgs=imgs, labels=labels)
        # check if the number of images in the cropped directory matches the number of ids
        if len(list(cropped_dir.glob("*.png"))) != len(img_ids):
            print("Warning: Some images were not cropped successfully.")
    else:
        outcome_df['img_path'] = [args.data_dir / "imgs" / f"{img_id}.png" for img_id in img_ids]

    # write outcome_df
    outcome_df.to_csv(exp_dir / "labels.csv", index=False)

if __name__ == "__main__":
    main()