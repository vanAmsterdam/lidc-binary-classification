import os
import numpy as np
import pandas as pd
import pylidc as pl
import pydicom
import pickle
from sklearn.preprocessing import PowerTransformer

def annotation_to_dict(ann):
    '''
    Flattens annotations into a single row pandas DataFrame
    '''
    
    d = {
        'patient_id': ann.scan.patient_id,
        'nodule_id_lidc': ann._nodule_id,
        'annotation_id_lidc': ann.id,
        'scan_id': ann.scan_id,
        'volume': ann.volume,
        'surface_area': ann.surface_area,
        'diameter': ann.diameter
    }
    feature_names = ['sublety', 'internalstructure', 'calcification',
               'sphericity', 'margin', 'lobulation', 'spiculation',
               'texture', 'malignancy']
    for feature, value in zip(feature_names, ann.feature_vals()):
        d[feature] = value
    return d

def annotation_to_df(ann):
    try:
        d = annotation_to_dict(ann)
        d = {k: [v] for k, v in d.items()}
        df = pd.DataFrame.from_dict(d)
    except:
        df = None
    return df

def annotation_list_to_df(anns):
    assert isinstance(anns, list)
    dfs = []
    for ann in anns:
        dfs.append(annotation_to_df(ann))
    df = pd.concat(dfs, ignore_index=True)
    df["annotation_idx"] = range(1, df.shape[0]+1)
    return df


def get_intercept_and_slope(scan):
    ''' 
    scan is the results of a pydicom query
    returns the intercept and slope
    adapted from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    '''
    dcm_path = scan.get_path_to_dicom_files()
    dcm_files = [x for x in os.listdir(dcm_path) if x.endswith(".dcm")]
    slice0 = pydicom.read_file(os.path.join(dcm_path, dcm_files[0]), stop_before_pixels=True)
    intercept = slice0.RescaleIntercept
    slope = slice0.RescaleSlope
    return intercept, slope

def resample_and_crop_annotation(ann_id, ann, nodule_path, mask_path=None, scan=None, size_mm = 50, export_mask = True):
    '''
    take an annotation, crop and resample
    size is the length of the sides of the resulting cube in millimeters
    '''
    if scan is None:
        scan = ann.scan
    intercept, slope = get_intercept_and_slope(scan)
    try:
        vol, mask = ann.uniform_cubic_resample(side_length = size_mm, verbose = True)
        if slope != 1:
            vol = slope * vol.astype(np.float64)

        vol = vol.astype(np.int16)
        vol += np.int16(intercept)
        
        np.save(os.path.join(nodule_path, ann_id+".npy"), vol)
        if export_mask:
            assert mask_path != None
            np.save(os.path.join(mask_path, ann_id+".npy"), mask)
        print("")
    except:
        print("-failed")


def crop_nodule_tight_z(ann, volume=None, scan=None, scan_spacing=None, out_size_cm = 5):
    """
    Get nodule cropped tightly in z direction, but of minimum dimension in xy plane
    """
    # print(f"trying to crop")
    if volume is None:
        if scan is None:
            scan = ann.scan
        volume = scan.to_volume()
    if scan_spacing is None:
        scan_spacing = scan.pixel_spacing
    # print(f"scan_spacing: {scan_spacing}")
    padding = get_padding_tight_z(ann, scan_spacing=scan_spacing, out_size_cm=out_size_cm)
    # print(f"padding: {padding}")

    mask = ann.boolean_mask(pad=padding)
    bbox = ann.bbox(pad=padding)
    zvals= ann.contour_slice_indices
    arr  = volume[bbox]

    return arr, mask, zvals

def get_padding_tight_z(ann, scan=None, scan_spacing=None, out_size_cm = None):
    """
    Get bbox dimensions base on a minimal size, restricting to no padding in z direction
    """
    if scan_spacing is None:
        if scan is None:
            scan_spacing = ann.scan.pixel_spacing
        else:
            scan_spacing = scan.pixel_spacing
    # return tight bounding box
    if out_size_cm is None:
        padding = [(int(0), int(0))] * 3
    else:
        # if len(out_size_cm == 3):
        #     if out_size_cm[2] is None:
        #         padding_z = (0,0)        
        out_shape = (np.ceil((out_size_cm * 10) / scan_spacing) * np.ones((2,))).astype(int)

        bb_mat   = ann.bbox_matrix()
        bb_shape = bb_mat[:,1] - bb_mat[:,0]

        paddings = out_shape - bb_shape[:2]

        # print(f"paddings: {paddings}")

        padding_x = (int(np.ceil(paddings[0] / 2)), int(np.floor(paddings[0] / 2)))
        padding_y = (int(np.ceil(paddings[1] / 2)), int(np.floor(paddings[1] / 2)))

        padding = [padding_x, padding_y, (int(0),int(0))]

    return padding



def normalize(x, window = None, level = None, in_min = -1000.0, in_max = 600.0, center=0.0):
    """
    Normalize array to values between 0 and 1, possibly clipping
    """
    assert type(x) is np.ndarray

    if (not window is None) & (not level is None) :
        in_min = level - (window / 2)
        in_max = level + (window / 2)

    x = x - in_min                 # add zero point
    x = x / (in_max - in_min)      # scale to unit
    x = x + center                 # adjust white-balance
    x = np.clip(x, 0.0, 1.0)       # clip to (0,1)
    return x

def normalized_to_8bit(x):
    assert ((x.min() >= 0) & (x.max() <= 1))
    x = (255 * x)
    return x.astype(np.uint8)

def normalize_to_8bit(x, *args, **kwargs):
    return normalized_to_8bit(normalize(x, *args, **kwargs))

def pwr_transform(x, train_ids=None):
    x  = np.array(x).reshape(-1,1)
    pt = PowerTransformer(method="yeo-johnson")
    if train_ids is None:
        pt.fit(x)
    else:
        pt.fit(x[train_ids])
    y = pt.transform(x)
    return np.squeeze(y)
