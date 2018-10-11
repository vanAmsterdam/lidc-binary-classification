import os
import numpy as np
import pandas as pd
import pylidc as pl
import feather # for writing data frame to disk (works with R)


def flatten_annotation(ann):
    '''
    Flattens annotations into a single row that can be added to a pandas DataFrame
    '''
    
    id_vals = np.array([
        ann.scan.patient_id,
        ann._nodule_id,
        ann.id,
        ann.scan_id], 
        dtype = '<U14')
    feature_vals = ann.feature_vals()
    return(id_vals, feature_vals)


def flatten_annotations(annotations):
    '''
    Take a list of annotations, return a pandas DataFrame
    '''
    if not isinstance(annotations, list):
        # makes sure that anns is a list, even if it is of length 1
        annotations = [annotations]

    # instantiate empty arrays for the values
    id_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[0].shape[0]), dtype = "<U14")
    feature_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[1].shape[0]), dtype = "int64")
    
    # loop over list of annotations
    for i, ann in enumerate(annotations):
        id_vals, feature_vals = flatten_annotation(ann)
        id_values[i,:] = id_vals
        feature_values[i,:] = feature_vals
    
    # combine together in a pandas DataFrame
    df_ids = pd.DataFrame(id_values, columns = ["patient_id", "nodule_id", "annotation_id", "scan_id"])
    df_feat= pd.DataFrame(feature_values, columns = [
                                         'sublety', 'internalstructure', 'calcification',
                                         'sphericity', 'margin', 'lobulation', 'spiculation',
                                         'texture', 'malignancy'])
    df = pd.concat([df_ids, df_feat], axis = 1)
    return(df)


def flatten_annotations_by_nodule(scans):
    '''
    take a list of scans, return a pandas DataFrame
    '''
    
    # instantiate DataFrame
    df = flatten_annotations(scans[0].annotations[0]).iloc[0:0]
    df.assign(nodule_number = np.empty(0, dtype = "int32"))
    
    # loop over scans
    for scan in scans:
        # loop over nodules within a scan
        for i, nodule_annotations in enumerate(scan.cluster_annotations()):
            if not isinstance(nodule_annotations, list):
                # makes sure that anns is a list, even if it is of length 1
                nodule_annotations = [nodule_annotations]
            nodule_df = flatten_annotations(nodule_annotations)
            nodule_df = nodule_df.assign(nodule_number = i+1)
            df = pd.concat([df, nodule_df], axis = 0)
    return(df)

def get_intercept_and_slope(scan, verbose = False):
    ''' 
    scan is the results of a pydicom query
    returns the intercept and slope
    adapted from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    '''
    imgs = scan.load_all_dicom_images(verbose = verbose)
    slice0 = imgs[0]
    intercept = slice0.RescaleIntercept
    slope = slice0.RescaleSlope
    return(intercept, slope)

def resample_and_crop(scans, path, size_mm = 5, export_mask = False):
    '''
    take a list of scans, export a numpy array with the nodule, and the mask of the segmentation
    size is the length of the sides of the resulting cube in millimeters
    '''
    for scan in scans:
        patient_id = scan.patient_id
        patient_number = patient_id[-4:]
        print(patient_id, end = "")
        nodules = scan.cluster_annotations()
        intercept, slope = get_intercept_and_slope(scan)
        
        for i, nodule_annotations in enumerate(nodules):
            nodule_number = i+1
            nodule_idx = str(patient_number)+str("%02d" % nodule_number)
            print(" nodule " +str(nodule_number), end = "")

            if not isinstance(nodule_annotations, list):
                # makes sure that anns is a list, even if it is of length 1
                nodule_annotations = [nodule_annotations]
            
            ann = np.random.choice(nodule_annotations, size = 1)[0]
            
            try:
                vol, mask = ann.uniform_cubic_resample(side_length = size_mm*10, verbose = False)
                
                if slope != 1:
                    vol = slope * vol.astype(np.float64)
                    vol = vol.astype(np.int16)

                vol = vol.astype(np.int16)
                vol += np.int16(intercept)
                
                np.save(file = os.path.join(path, str(nodule_idx)+"_array.npy"), arr = vol)
                if export_mask:
                    np.save(file = os.path.join(path, str(nodule_idx)+"_mask.npy"), arr = mask)
                print("")
            except:
                print("-failed")
                
def flatten_multiindex_columns(df, sep = "_"):
    '''
    If a pandas DataFrame has a hierarchical index,
    flatten to single level
    '''
    col_vals = df.columns.values
    flattened = [sep.join(x) for x in col_vals]
    stripped = [x[:-1] if sep == x[-1] else x for x in flattened]
    df.columns = stripped
    return df
              