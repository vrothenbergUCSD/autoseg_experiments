import subprocess
import os
from datetime import datetime
import argparse 
from pathlib import Path
import time

def run_command(command):
    try: 
        result = subprocess.run(command, shell=True, text=True)
        if result.returncode != 0:  # Check if the command failed
            print(f"Error running command: {command} with return code {result.returncode}")
        elif result.stdout:  # Only print stdout if it's not empty
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.cmd} with return code {e.returncode}")

def generate_unique_filename(base, extension=".pkl"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base}_{timestamp}{extension}"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run post-processing pipeline scripts.")
    parser.add_argument('--input_zarr', required=True, help='Input Zarr directory path')
    parser.add_argument('--dataset', required=True, help='Dataset name for processing')
    # parser.add_argument('--output_zarr', required=True, help='Output Zarr directory path')
    parser.add_argument('--num_cores', default="32", help='Number of cores to use (default: 32)')
    parser.add_argument('--threshold', default=25, type=int, help='Threshold for filtering (default: 5)')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()

        output_dataset_filtered = f"{args.dataset}_filtered"
        output_dataset_remapped = f"{args.dataset}_remapped"
        output_dataset_xyz = f"{args.dataset}_xyz"
        log_path = os.path.join(os.getcwd(), "log", datetime.now().strftime("%Y%m%d%H%M%S"))
        counts_pickle_path = os.path.join(log_path, "combined_counts.pkl")
        sync_path = os.path.join(log_path, "data.sync")
        # output_zarr = str(args.input_zarr).replace(".zarr", "_filtered.zarr")
        output_zarr = args.input_zarr
        print(f"Output zarr path: {output_zarr}")

        Path(log_path).mkdir(parents=True, exist_ok=True)

        # Run segmentation_counts_chunkwise.py
        start = time.time()
        command = f"""python segmentation_counts_chunkwise.py --input_zarr {args.input_zarr} \
            --dataset {args.dataset} --num_cores {args.num_cores} \
            --output_pickle {counts_pickle_path} --log_path {log_path}"""
        run_command(command)

        elapsed = time.time() - start
        print(f'Elapsed: {round(elapsed,2)}s')

        # Run filter_counts.py
        start = time.time()
        command = f"""python filter_counts_chunkwise.py --input_zarr {args.input_zarr} \
    --input_dataset {args.dataset} \
    --output_zarr {output_zarr} \
    --output_dataset {output_dataset_filtered} \
    --sync_path {sync_path} \
    --counts_pickle_path {counts_pickle_path} \
    --threshold {args.threshold} \
    --num_cores {args.num_cores} \
    --log_path {log_path}
    """
        run_command(command)
        elapsed = time.time() - start
        print(f'Elapsed: {round(elapsed,2)}s')

        # Run remap_ids.py
        start = time.time()
        command = f"""python remap_ids_chunkwise.py --input_zarr {output_zarr} \
    --input_dataset {output_dataset_filtered} \
    --output_dataset {output_dataset_remapped} \
    --sync_path {sync_path} \
    --num_cores {args.num_cores} \
    --log_path {log_path}"""
        run_command(command)
        elapsed = time.time() - start
        print(f'Elapsed: {round(elapsed,2)}s')

        # Run transpose_zarr_optimized.py
        start = time.time()
        command = f"""python transpose_zarr_optimized.py --input_zarr {output_zarr} \
    --input_dataset {output_dataset_remapped} \
    --output_dataset {output_dataset_xyz} \
    --sync_path {sync_path} \
    --num_cores {args.num_cores} \
    --log_path {log_path}"""
        run_command(command)
        elapsed = time.time() - start
        print(f'Elapsed: {round(elapsed,2)}s')



    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")