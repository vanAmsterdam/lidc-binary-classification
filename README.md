# lidc-binary-classification
This repository contains code to pre-process the LIDC-IDRI dataset of CT-scans with pulmonary nodules into a binary classification problem, easy to use for learning deep learning


## Overview

The workflow consists of a few steps

1. use the pylidc library to process image annotations and segmentations (identifying malignant vs benign and the locations of the nodules)
2. resample to 1mm x 1mm x 1mm and process HU values of different scanners
3. export cropped regions around the nodules in 2 ways: 3D cubes, 2D slices


## Download scans

Download the original scans using the steps from this website: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI



## Setup python environment

1. download anaconda 3
2. create a new environment (e.g. conda create --name lidc)
3. install some packages

(note we need scikit-image version 0.13 since replacement of measure.marching_cubes with measure.marching_cubes_lewiner in version 0.14 breaks compatibility with pylidc (as of yet)

`conda install jupyter numpy pandas feather-format scikit-image=0.13`

`pip install pylidc pypng`

4. configure pylidc to know where the scans are located, follow these steps: https://pylidc.github.io/install.html

## Follow the notebook



## Issues

Currently, the code uses the pylidc function 'cluster_annotations' twice: ones to create a DataFrame of annotations, a second time to export the images. Since this function takes some time, this could be made more efficient