#!/bin/bash

# Base directories
ZARR_BASE="/data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL_400k_filtered.zarr"
OUTPUT_DIR="/data/binaryData/Panda_Lab/SCN_DL_12AM_VL-1500"

# Voxel size (adjust as needed)
VOXEL_SIZE="15.6,15.6,30.0"

# Number of jobs (adjust as needed)
JOBS=16

# Iterate over each segmentation directory
for SEG_DIR in ${ZARR_BASE}/segmentation_*_xyz; do
  # Extract the segmentation name (e.g., segmentation_0.02_xyz)
  SEG_NAME=$(basename ${SEG_DIR})
  
  # Extract the numeric value (e.g., 0.02)
  SEG_VALUE=$(echo ${SEG_NAME} | grep -oP '(?<=segmentation_)[0-9]+\.[0-9]+')

  # Construct the layer name (e.g., seg_0.02)
  LAYER_NAME="seg_${SEG_VALUE}"
  
  # Run the webknossos convert-zarr command
  webknossos convert-zarr --jobs ${JOBS} --is-segmentation-layer --layer-name ${LAYER_NAME} --voxel-size ${VOXEL_SIZE} --data-format wkw --compress ${SEG_DIR} ${OUTPUT_DIR}
done


# /data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction