import subprocess
import os

def run_postprocess(input_zarr, num_cores, threshold):
    for i in range(1, 10):  # 0.1 to 1 with steps of 0.1
        segmentation_value = round(i * 0.1, 1)
        dataset_name = f"segmentation_{segmentation_value:.1f}"
        command = f"python postprocess.py --input_zarr {input_zarr} --dataset {dataset_name} --num_cores {num_cores} --threshold {threshold}"
        
        print(f"Running command: {command}")
        result = subprocess.run(command, shell=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command} with return code {result.returncode}")
        else:
            print(f"Completed: {command}")

if __name__ == "__main__":
    input_zarr = "/data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL_400k.zarr"
    num_cores = 32
    threshold = 25
    
    run_postprocess(input_zarr, num_cores, threshold)
