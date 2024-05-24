import zarr
import numpy as np
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from rich.progress import Progress
import numcodecs
import logging
import time 

# Blosc may share incorrect global state amongst processes causing programs to hang
numcodecs.blosc.use_threads = False


parser = argparse.ArgumentParser(description="Count segmentation objects in parallel and save the counts using a chunk-wise approach.")
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr file.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name within the Zarr file.')
parser.add_argument('--num_cores', type=int, default=32, help='Number of cores to use for processing.')
parser.add_argument('--output_pickle', type=str, default='combined_counts.pkl', help='Output pickle file name for combined counts.')
parser.add_argument('--log_path', type=str, default='log', help='Directory for log files')
args = parser.parse_args()

# Initialize logging
logging.basicConfig(filename=f'{args.log_path}/segmentation_counts_chunkwise.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# def count_segmentation_in_chunk(chunk_range, zarr_array):
#     # Calculate slice objects for this chunk
#     slices = tuple(slice(start, start + chunk) for start, chunk in zip(chunk_range, zarr_array.chunks))
    
#     # Load the chunk and filter out the background
#     chunk_data = zarr_array[slices].flatten()
#     chunk_data = chunk_data[chunk_data != 0]  # Exclude background
    
#     # Count unique segmentation IDs in this chunk
#     return Counter(chunk_data)

def count_segmentation_in_chunk(chunk_range, zarr_array):
    # Initialize an empty dictionary to hold the layer counts for segment IDs
    layer_counts = {}
    # Calculate slice objects for this chunk
    slices = tuple(slice(start, start + chunk) for start, chunk in zip(chunk_range, zarr_array.chunks))
    
    # Load the chunk. No need to flatten as we need the 3D structure
    chunk_data = zarr_array[slices]
    
    # Iterate over each layer in the chunk
    for layer_idx in range(chunk_data.shape[0]):
        layer_data = chunk_data[layer_idx].flatten()
        layer_data = layer_data[layer_data != 0]  # Exclude background
        unique_ids = np.unique(layer_data)
        
        # Update the layer_counts dictionary
        for seg_id in unique_ids:
            if seg_id not in layer_counts:
                layer_counts[seg_id] = {}
            layer_counts[seg_id][chunk_range[0] + layer_idx] = 1
    
    return layer_counts


def get_chunk_ranges(shape, chunk_sizes):
    """
    Calculate the ranges of indices for each chunk.
    """
    return np.array(np.meshgrid(*[range(0, s, cs) for s, cs in zip(shape, chunk_sizes)], indexing='ij')).reshape(len(shape), -1).T

if __name__ == '__main__':
    start = time.time()
    logging.info(f"Reading Zarr file from {args.input_zarr}")
    data = zarr.open(args.input_zarr, mode='r')
    if args.dataset not in data:
        logging.error(f"Dataset {args.dataset} not found in Zarr file.")
        exit(1)
    
    zarr_array = data[args.dataset]
    logging.info(f"Zarr array shape: {zarr_array.shape}, chunks: {zarr_array.chunks}")
    logging.info(f"Number of cores: {args.num_cores}")
    
    chunk_ranges = get_chunk_ranges(zarr_array.shape, zarr_array.chunks)

    combined_layer_counts = {}
    
    # Initialize progress bar
    with Progress() as progress:
        task = progress.add_task("Counting...", total=len(chunk_ranges))
        
        with ProcessPoolExecutor(max_workers=args.num_cores) as executor:
            futures = {executor.submit(count_segmentation_in_chunk, chunk_range, zarr_array): chunk_range for chunk_range in chunk_ranges}
            for future in as_completed(futures):
                progress.advance(task)
                chunk_result = future.result()
                for seg_id, layers in chunk_result.items():
                    if seg_id not in combined_layer_counts:
                        combined_layer_counts[seg_id] = {}
                    combined_layer_counts[seg_id].update(layers)

    final_counts = {seg_id: len(layers) for seg_id, layers in combined_layer_counts.items()}

    # Save the combined counts to a pickle file
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(dict(final_counts), f)

    message = f"Total unique segments counted: {len(final_counts)}"
    logging.info(message)
    print(message)
    exit(0)
