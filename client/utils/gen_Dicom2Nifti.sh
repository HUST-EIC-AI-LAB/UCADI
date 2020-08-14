#!/bin/bash

# modify data storage paths and gen_path() in preprocess.py based on your own
python preprocess.py \
    --input ../data/COVID-19/dataCT/scans_dicom/TongjiHospital/COVID/ \
    --output ../data/COVID-19/dataCT/scans_dicom/TongjiHospital/Converted_COVID_NIFTI/ ;

python preprocess.py \
    --input ../data/COVID-19/dataCT/scans_dicom/TongjiHospital/NonCOVID/ \
    --output ../data/COVID-19/dataCT/scans_dicom/TongjiHospital/Converted_NonCOVID_NIFTI/ ;
