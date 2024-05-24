# from multiprocessing import Pool, cpu_count, Manager
import numpy as np
import zarr
import pickle
import logging
# from tqdm import tqdm
import time
import argparse
import os
import numcodecs
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

# Blosc may share incorrect global state amongst processes causing programs to hang
#   https://zarr.readthedocs.io/en/stable/tutorial.html#tutorial-tips-blosc
numcodecs.blosc.use_threads = False


# Setup argument parser
parser = argparse.ArgumentParser(description='Filter objects from segmentation datasets below a threshold.')
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr.')
parser.add_argument('--input_dataset', type=str, required=True, help='Name of input Zarr dataset to filter IDs.')
parser.add_argument('--output_zarr', type=str, required=True, help='Path to output Zarr.')
parser.add_argument('--output_dataset', type=str, required=True, help='Name of new output Zarr dataset.')
parser.add_argument('--counts_pickle_path', type=str, required=True, help='Path to pickle file with segmentation counts.')
parser.add_argument('--threshold', type=int, default=5, help='Threshold for filtering objects.')
parser.add_argument('--num_cores', type=int, default=10, help='Number of cores to use for parallel multi-threading.')
parser.add_argument('--log_path', type=str, default='log', help='Directory for log files')
args = parser.parse_args()

# Initialize logging
logging.basicConfig(filename=f'{args.log_path}/filter_counts.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def filter_chunk(chunk_range, zarr_input_dataset, zarr_output_dataset, ids_to_remove):
    # Calculate slice objects for this chunk
    # slices = tuple(slice(start, min(start + size, end)) for start, size, end in zip(chunk_range, zarr_input_dataset.chunks, zarr_input_dataset.shape))
    slices = tuple(slice(start, start + chunk) for start, chunk in zip(chunk_range, zarr_input_dataset.chunks))
    logging.info(f"Chunk range: {chunk_range}")
    # Load the chunk, apply filtering, and save the result
    chunk_data = zarr_input_dataset[slices]
    # logging.info(f"Chunk data shape {chunk_data.shape}")
    mask = np.isin(chunk_data, list(ids_to_remove))
    chunk_data[mask] = 0  # Set filtered IDs to 0
    zarr_output_dataset[slices] = chunk_data
    # logging.info(f"zarr_output_dataset shape {zarr_output_dataset[slices].shape}")
    return np.count_nonzero(mask)

def get_chunk_ranges(shape, chunk_size):
    """
    Calculate the ranges of indices for each chunk.
    """
    ranges = [range(0, s, cs) for s, cs in zip(shape, chunk_size)]
    return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(shape), -1).T


if __name__ == '__main__':
    # Load segmentation counts
    with open(args.counts_pickle_path, 'rb') as f:
        segmentation_counts = pickle.load(f)
    
    # Identify IDs to remove
    ids_to_remove = {seg_id for seg_id, count in segmentation_counts.items() if count < args.threshold}
    logging.info(f"Removing {len(ids_to_remove)} IDs")
    print(f"Removing {len(ids_to_remove)} IDs")

    ids_to_remove_np = np.array(list(ids_to_remove))

    # Open input Zarr dataset
    zarr_input_dataset = zarr.open(args.input_zarr, mode='r')[args.input_dataset]

    # Create output Zarr dataset if it does not exist
    synchronizer = zarr.ThreadSynchronizer()
    zarr_output_group = zarr.open(args.output_zarr, mode='a', synchronizer=synchronizer)
    if args.output_dataset in zarr_output_group:
        del zarr_output_group[args.output_dataset]
    zarr_output_group.create_dataset(args.output_dataset, 
                                                   shape=zarr_input_dataset.shape, 
                                                   chunks=zarr_input_dataset.chunks, 
                                                   dtype=zarr_input_dataset.dtype, 
                                                   synchronizer=synchronizer,
                                                   overwrite=True)
    
    zarr_output_dataset = zarr_output_group[args.output_dataset]

    logging.info("Opened segmentation zarr array")

    # Process each chunk in parallel
    chunk_ranges = get_chunk_ranges(zarr_input_dataset.shape, zarr_input_dataset.chunks)
    logging.info(chunk_ranges)
    total_chunks = len(chunk_ranges)

    try: 
        with Progress() as progress:
            task = progress.add_task("Filtering...", total=total_chunks)
            with ProcessPoolExecutor(max_workers=args.num_cores) as executor:
                # futures = [executor.submit(filter_chunk, (i, j, k), zarr_input_dataset, zarr_output_dataset, ids_to_remove)
                #            for i in chunk_ranges[0] for j in chunk_ranges[1] for k in chunk_ranges[2]]
                futures = {executor.submit(filter_chunk, chunk_range, zarr_input_dataset, zarr_output_dataset, ids_to_remove_np): chunk_range
                    for chunk_range in chunk_ranges}
                
                for future in as_completed(futures):
                    chunk_range = futures[future]  # Retrieve the layer chunk for the completed task
                    try:
                        future.result()  # Optionally handle result or exceptions
                    except Exception as exc:
                        logging.error(f'Chunk {chunk_range} generated an exception: {exc}')
                    progress.advance(task)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        progress.stop()
        exit(1)
    logging.info('Completed filtering Zarr object.')
    print('Completed filtering Zarr object.')
    exit(0)