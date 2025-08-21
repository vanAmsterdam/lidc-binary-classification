# lidc-binary-classification

This repository contains code to pre-process the LIDC-IDRI dataset of CT-scans
with pulmonary nodules into a binary classification problem, easy to use for learning deep learning.

Input:

- many .dcm files and annotations stored in XML database

Output:

- neat png files
- with labels in csv file

## Usage

[Download](@Donwloading) the Data

## Overview

The workflow consists of a few steps

### Downloading

0. download data from TCIA (133 Gb) from official repository [http://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX](http://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX)
Note: when using the `NBIA Data Retrieverz, be sure to select`Classic Directory Name`

### Extract regions, annotations and segmentations

Using the [pylidc library](https://github.com/notmatthancock/pylidc) to do the heavy lifting:
identifying malignant vs benign and the locations of the nodules

### format

2. resample to 1mm x 1mm x 1mm and process HU values of different scanners
3. export cropped regions around the nodules in 2 ways: 3D cubes, 2D slices

## Download scans

Download the original scans using the steps from this website:
[https://www.cancerimagingarchive.net/collection/lidc-idri/](https://www.cancerimagingarchive.net/collection/lidc-idri/)

## Setup python environment

1. download anaconda 3
2. create a new environment (e.g. conda create --name lidc)
3. install some packages

(note we need scikit-image version 0.13 since replacement of measure.marching_cubes with measure.marching_cubes_lewiner in version 0.14 breaks compatibility with pylidc (as of yet)

`conda install jupyter numpy pandas feather-format scikit-image=0.13`

`pip install pylidc pypng`

4. configure pylidc to know where the scans are located, follow these steps: <https://pylidc.github.io/install.html>

## Follow the notebook

Pre processing: lidc-preprocessing.jpynb

Modeling example:

- keras + tf CNN 3D: CNN_keras_3D.jpynb
- keras + tf CNN 2D: CNN_keras_3D.jpynb

## Issues

Currently, the code uses the pylidc function 'cluster_annotations' twice: ones to create a DataFrame of annotations, a second time to export the images. Since this function takes some time, this could be made more efficient

This is by no means an 'optimal' approach in the sense that I have not experimented with hyperparameters of the pre-processing like

- resampling size
- 'borderline malignancy' definition
- output size
- number of 2D slices
- extensive CNN alterations

But it is enough to get a model running as one can see from the provided examples. It should be able to get you up to speed for using deep learning on actual medical images!
