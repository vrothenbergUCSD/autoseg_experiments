import shutil
import os
from pathlib import Path

def copy_and_zip_datasets(input_zarr_path, output_zarr_path, zip_output_path):
    # Ensure the output directory exists
    Path(output_zarr_path).mkdir(parents=True, exist_ok=True)

    # Get list of directories in the input Zarr path
    for dataset_name in os.listdir(input_zarr_path):
        if dataset_name.startswith('segmentation_') and dataset_name.endswith('_xyz'):
            source_path = os.path.join(input_zarr_path, dataset_name)
            destination_path = os.path.join(output_zarr_path, dataset_name)
            print(f"Copying dataset: {dataset_name}")
            # Copy the directory
            shutil.copytree(source_path, destination_path)
    
    # Zip the output Zarr object
    print(f"Zipping the output Zarr object to {zip_output_path}.zip")
    shutil.make_archive(zip_output_path, 'zip', output_zarr_path)

    print("Copy and zip process completed.")

if __name__ == "__main__":
    input_zarr_path = '/data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL_400k.zarr'
    output_zarr_path = '/data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL_400k_filtered.zarr'
    zip_output_path = '/data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL_400k_filtered'

    copy_and_zip_datasets(input_zarr_path, output_zarr_path, zip_output_path)
