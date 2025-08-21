"""
Generall data pre-processing, used in all experimental settings
"""

# # Pre-process LIDC CT scans to extract labelled nodules

from tqdm import tqdm
import os, sys
import pickle
sys.path.append("./")
import numpy as np
import pandas as pd
import pylidc as pl
import png
from preprocessingutils import *
import concurrent.futures
import nibabel as nib
from PIL import Image
from pathlib import Path

# note that this script assumes its evaluated from the local directory;
# if you want the data to be stored somewhere else, adapt this,
# or create a simlink for data

preprocessdir = Path.cwd()
homedir       = preprocessdir.parent
DATA_DIR = homedir / 'data'
if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)
LOG_FILE = Path("log") / "lidc-preprocessing.log"
if not LOG_FILE.parent.exists():
    os.makedirs(LOG_FILE.parent)
RESOURCES_DIR = homedir / 'resources'
if not RESOURCES_DIR.exists():
    os.makedirs(RESOURCES_DIR)
MAX_WORKES = 1
OUT_SIZE = (69,69,69)
OUT_SHAPE_2D = (180,180)
SPACING_2D = .5
OUT_SIZE_MM_2D = tuple(np.array(OUT_SHAPE_2D) * SPACING_2D)
# MIN_MM2  = 5**2
MIN_MM2 = 0
WRITE_NORESAMP_NODULES = False # export non-resampled nodules
DO_RESAMP = False# resample and crop for 3D nodules
DO_SLICES = True # generate 2D slices

TEST_MODE = False
print(f"test mode: {TEST_MODE}")

file = open(LOG_FILE, "w+")

# The LIDC database contains annotations of up to 4 radiologist per nodule.
# We need to combine these annotations. Luckily, the pylidc module provides a way to cluster annotations from overlapping nodules
# It turns out that 'nodule_id' does not refer to a nodule at all, they do not overlap.
# Luckily, pylidc has functionality built in to determine which nodules belong together
# 
# Extract annotations to dataframe (note: using pd.read_sql_table might be better but I couldn't figure out which connection to use)
# ## Load scans with pylidc
# Create dataframe with scan information

scans = pl.query(pl.Scan).all()
scan_dict = {}
for scan in scans:
    patient_id = scan.patient_id[-4:]
    if patient_id in scan_dict.keys():
        print(f"patient with multiple scans: {patient_id}; ", end="")
        patient_id = str(format(int(patient_id) + int(2000)))
        print(f"new id: {patient_id}")
    scan_dict[patient_id] = scan
assert len(scan_dict.keys()) == 1018


if not (RESOURCES_DIR / "scan_df.csv").exists():
    scan_df_dict = {}
    print("preparing scan dataframe")
    for patient_id, scan in tqdm(scan_dict.items()):   # TODO add scan-id here
        scan_df_dict[patient_id] = {
            'contrast_used':        scan.contrast_used,
            'id':                   scan.id,
            'is_from_initial':      scan.is_from_initial,
            'patient_id_lidc':      scan.patient_id,
            'pixel_spacing':        scan.pixel_spacing,
            'series_instance_uid':  scan.series_instance_uid,
            'slice_spacing':        scan.slice_spacing,
            'slice_thickness':      scan.slice_thickness,
            'spacing_x':            scan.spacings[0],
            'spacing_y':            scan.spacings[1],
            'spacing_z':            scan.spacings[2],
            'study_instance_uid':   scan.study_instance_uid
        }
    scan_df = pd.DataFrame.from_dict(scan_df_dict, orient="index")
    scan_df.index = ["{:04d}".format(int(x)) for x in scan_df.index.values]
    scan_df.to_csv(RESOURCES_DIR / 'scan_df.csv', index=True, index_label="patient_id")
else:
    scan_df = pd.read_csv(RESOURCES_DIR / 'scan_df.csv')

# scans can contain multiple annotations
#  each annooation has an id, there are nodule ids, but these don't coincide accross annotations, while in reality, some annotations concern the same actual nodule. This data is combined in the 'nodule_number' column, which numbers the nodules for each patient
# Add the patient number as a column to de DataFrame, and create an actual nodule ID based on the patient number and the nodule number

# Takes a long time, so this is stored and can be picked up here:

# cluster nodules

if not os.path.exists(os.path.join(DATA_DIR, "nodule-clusters")):
    os.makedirs(os.path.join(DATA_DIR, "nodule-clusters"))

clustered_annotations = {}

if TEST_MODE:
    scan_dict = {k: scan_dict[k] for k in list(scan_dict.keys())[:5]}

nodule_files = os.listdir(os.path.join(DATA_DIR, "nodule-clusters"))
patients_with_nodules = list(set([x.split("n")[0] for x in nodule_files]))

for patient_id, scan in scan_dict.items():
    if patient_id in patients_with_nodules:
        nodule_ids = [x for x in nodule_files if x.startswith(patient_id)]
        for nodule in nodule_ids:
            nodule_id = nodule.rstrip(".pkl")
            with open(os.path.join(DATA_DIR, "nodule-clusters", nodule), "rb") as f:
                clustered_annotations[nodule_id] = pickle.load(f)
    else:
        print("")
        print("extracting nodules for patient {}".format(patient_id), end="")
        for i, clustered_annotation in enumerate(scan.cluster_annotations()):
            print(" n{:02d}".format(i+1), end="")
            if not isinstance(clustered_annotation, list):
                clustered_annotation = [clustered_annotation]
            nodule_id = "{}n{:02d}".format(patient_id, i+1)
            clustered_annotations[nodule_id] = clustered_annotation
            with open(os.path.join(DATA_DIR, "nodule-clusters", nodule_id + ".pkl"), "wb") as f:
                pickle.dump(clustered_annotation, f)


# export all annotations in flat dict 
# TODO: do this earlier for prettier looping
# TODO: sort keys on patient id, to actually benefit from loading scans only once per patient...
anns = {}
nodule_ids = list(clustered_annotations.keys())
nodule_ids.sort()
for nodule_id in nodule_ids:
    annotation_list = clustered_annotations[nodule_id]
    for i, ann in enumerate(annotation_list):
        annotation_id = "{}a{}".format(nodule_id, i+1)
        anns[annotation_id] = ann

if not (RESOURCES_DIR / "annotation_df.csv").exists():
    # annotation_dfs = {k: pd.concat([annotation_to_df(ann) for ann in cluster]) for k, cluster in clustered_annotations.items()}
    annotation_dfs = {}
    for nodule_id, cluster in clustered_annotations.items():
        try:
            annotation_dfs[nodule_id] = annotation_list_to_df(cluster)
        except:
            print("annotation to df failed for nodule {}".format(nodule_id))
            print("annotation to df failed for nodule {}".format(nodule_id), file=file)
    # annotation_dfs = {k: annotation_list_to_df(cluster) for k, cluster in clustered_annotations.items()}
    annotation_df = pd.concat(annotation_dfs)
    annotation_df.reset_index(inplace=True)
    annotation_df.rename(index=str, columns={'level_0': 'nodule_id'}, inplace=True)
    annotation_df = annotation_df.drop(["level_1"], axis="columns")
    annotation_df["annotation_id"] = annotation_df[["nodule_id", "annotation_idx"]].apply(lambda x: "{}a{}".format(*x), axis=1)
    annotation_df["nodule_idx"] = [x[:4]+x[5:] for x in annotation_df["nodule_id"]]

    annotation_df.to_csv(RESOURCES_DIR / "annotation_df.csv", index=False)
else:
    annotation_df = pd.read_csv(RESOURCES_DIR / "annotation_df.csv")

# write out non-resampled nodules
# TODO load scan per patient id, not per annotation id (takes way longer)
if WRITE_NORESAMP_NODULES: 
    print("saving non-resampled nodules and masks")
    for nodule_id, annotation_list in tqdm(clustered_annotations.items()):
        for i, ann in enumerate(annotation_list):
            annotation_id = "{}a{}".format(nodule_id, i+1)
            if not os.path.exists(os.path.join(DATA_DIR, "nodules3d-noresamp", "{}.npy".format(annotation_id))):
                try:
                    vol  = ann.scan.to_volume()
                    mask = ann.boolean_mask()
                    bbox = ann.bbox()
                    nodule = vol[bbox]
                    np.save(os.path.join(DATA_DIR, "nodules3d-noresamp", "{}.npy".format(annotation_id)), nodule)
                    np.save(os.path.join(DATA_DIR, "masks3d-noresamp", "{}.npy".format(annotation_id)), mask)
                except:
                    print(f"annotation id {annotation_id} failed")


# ## Resample and crop
# CT scanners can have different intercepts and slopes for converting the raw voxel data to Hounsfield Units, which represent radiodensity.
# This information can be extracted from the dicom headers and used to get all images on a uniform scale
# 
# Adapted from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
# 
# Secondly we will pick a random segmentation of the nodule and extract a bounding box around the nodule from the scan, along with the actual segmentation which is representated as a boolean mask. Set the seed to select the random annotation


for out_dir in ["nodules3d", "nodules2d", "masks3d", "masks2d"]:
    if not os.path.exists(os.path.join(DATA_DIR, out_dir)):
        os.makedirs(os.path.join(DATA_DIR, out_dir))
#%%
if DO_RESAMP: 
    print('resampling')
    last_pid = ""
    for ann_id, ann in tqdm(anns.items()):
        current_pid = ann_id.split("n")[0]
        if current_pid != last_pid:
            try:
                scan = annotation_list[0].scan
            except: 
                print(f"loading scan for patient id {current_pid}, annotation id {ann_id} failed")
                print(f"loading scan for patient id {current_pid}, annotation id {ann_id} failed", file=file)
            last_pid = current_pid
        if not os.path.exists(os.path.join(DATA_DIR, "nodules3d", ann_id+".npy")):
            resample_and_crop_annotation(ann_id, ann, 
                os.path.join(DATA_DIR, "nodules3d"), 
                os.path.join(DATA_DIR, "masks3d"),
                scan=scan,
                size_mm=OUT_SIZE[0])
        else:
            print(f"{ann_id}.npy already exists")

# make niftis, for radiomics extraction
for out_dir in ["nodules", "masks"]:
    if not os.path.exists(os.path.join(DATA_DIR, "niftis", out_dir)):
        os.makedirs(os.path.join(DATA_DIR, "niftis", out_dir))
nods = os.listdir(os.path.join(DATA_DIR, "nodules3d"))
print("converting nodule numpy arrays to niftis")
for nod in tqdm(nods):
    ann_id = os.path.splitext(nod)[0]
    out_name = ann_id+".nii.gz"
    if not os.path.exists(os.path.join(DATA_DIR, "niftis", "nodules", out_name)):
        nod_npy = np.load(os.path.join(DATA_DIR, "nodules3d", nod))
        nii_img = nib.Nifti1Image(nod_npy.astype(np.float64), np.eye(4))
        nib.save(nii_img, os.path.join(DATA_DIR, "niftis", "nodules", out_name))

masks = os.listdir(os.path.join(DATA_DIR, "masks3d"))
print("converting mask numpy arrays to niftis")
for mask in tqdm(masks):
    ann_id = os.path.splitext(mask)[0]
    out_name = ann_id + ".nii.gz"
    if not os.path.exists(os.path.join(DATA_DIR, "niftis", "masks", out_name)):
        mask_npy = np.load(os.path.join(DATA_DIR, "masks3d", mask))
        nii_img = nib.Nifti1Image(mask_npy.astype(np.float64), np.eye(4))
        nib.save(nii_img, os.path.join(DATA_DIR, "niftis", "masks", out_name))
    
# ### Generate 2D slices based on the nodules
# Take all slices from the non-resampled nodules
# 


for out_dir in ["imgs", "masks"]:
    if not os.path.exists(os.path.join(DATA_DIR, "nodules2d", out_dir)):
        os.makedirs(os.path.join(DATA_DIR, "nodules2d", out_dir))

existing_files = os.listdir(os.path.join(DATA_DIR, "nodules2d", "imgs"))
existing_slices  = list(set([x.split("s")[0] for x in existing_files]))
existing_nodules = list(set([x.split("a")[0] for x in existing_files]))

if DO_SLICES:
    print('creating slices')
    last_pid = ""
    for ann_id, ann in tqdm(anns.items()):
        current_pid = ann_id.split("n")[0]
        current_nodule_id = ann_id.split("a")[0]
        if current_nodule_id in existing_nodules:
            continue
        if current_pid != last_pid:
            try:
                print(f"loading scan for patient {current_pid}")
                scan         = ann.scan
                volume       = scan.to_volume()
                scan_spacing = scan.pixel_spacing
                intercept, slope = get_intercept_and_slope(scan)

                # fix slope and intercept (these are scanner settings; slope can be 0 or -1024 (big difference!))
                volume *= np.array(slope, dtype=np.int16)
                volume += np.array(intercept, dtype=np.int16)
                
            except Exception as e: 
                print(f"loading scan for patient {current_pid}, annotation id {ann_id} failed: {e}")
                print(f"loading scan for patient {current_pid}, annotation id {ann_id} failed: {e}", file=file)
                continue
            last_pid = current_pid

        if not ann_id in existing_slices:
            print("slicing annotation {}".format(ann_id))

            # crop and normalize
            try:
                nodule, mask, zvals = crop_nodule_tight_z(ann, volume, scan_spacing=scan_spacing, out_size_cm=OUT_SIZE_MM_2D[0] / 10)
                if zvals.shape[0] < nodule.shape[2]:
                    print(f"length of zvals ({zvals.shape[0]}) smaller than z dimension of nodule ({nodule.shape})")
                    print(f"length of zvals ({zvals.shape[0]}) smaller than z dimension of nodule ({nodule.shape})", file=file)
                    new_zvals = np.zeros((nodule.shape[2],))
                    new_zvals[:zvals.shape[0]] = zvals
                    new_zvals[zvals.shape[0]:] = zvals.max() + 1 + np.arange(len(new_zvals) - len(zvals))
                    zvals = new_zvals.astype(int)
            except Exception as e:
                print(f"cropping failed, skipping...: {e}")
                print(f"cropping failed, skipping...: {e}", file=file)
                continue
            
            nodule = normalize_to_8bit(nodule, in_min = -2200.0, in_max = 1000.0, center=0.0)
            mask   = normalize_to_8bit(mask,   in_min = 0.0, in_max = 1.0)

            num_slices = nodule.shape[2]
            j = int(0)
            # export as images
            for slice_index in range(num_slices):
                slice_i, mask_i, zval_i  = nodule[:,:,slice_index], mask[:,:,slice_index], zvals[slice_index]
                if mask_i.sum() > (MIN_MM2 / (scan_spacing**2)):
                    j += 1
                    slice_id = "{}s{:03d}".format(ann_id, zval_i)
                    img_nod = Image.fromarray(slice_i, mode="L")
                    img_nod = img_nod.resize(OUT_SHAPE_2D[:2])
                    img_nod.save(os.path.join(DATA_DIR, "nodules2d", "imgs", slice_id+".png"))
                    img_mask = Image.fromarray(mask_i, mode="L")
                    img_mask = img_mask.resize(OUT_SHAPE_2D[:2])
                    img_mask.save(os.path.join(DATA_DIR, "nodules2d", "masks", slice_id+".png"))

file.close()
