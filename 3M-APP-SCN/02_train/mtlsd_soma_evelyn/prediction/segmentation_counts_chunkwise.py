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

# Disable Blosc threading to avoid incorrect global state sharing amongst processes
numcodecs.blosc.use_threads = False

# Argument parser configuration
parser = argparse.ArgumentParser(description="Count segmentation objects in parallel and save the counts using a chunk-wise approach.")
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr file.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name within the Zarr file.')
parser.add_argument('--num_cores', type=int, default=64, help='Number of cores to use for processing.')
parser.add_argument('--output_pickle', type=str, default='combined_counts.pkl', help='Output pickle file name for combined counts.')
parser.add_argument('--log_path', type=str, default='log', help='Directory for log files')
args = parser.parse_args()

# Initialize logging
logging.basicConfig(
    filename=f'{args.log_path}/segmentation_counts_chunkwise.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def count_segmentation_in_chunk(chunk_range, zarr_array):
    """
    Count unique segmentation IDs in a given chunk of the Zarr array.
    
    Parameters:
    - chunk_range: Tuple containing the starting indices of the chunk.
    - zarr_array: The Zarr array from which to read data.
    
    Returns:
    - layer_counts: A dictionary with segmentation IDs as keys and a nested dictionary of layer indices as values.
    """
    layer_counts = {}
    slices = tuple(slice(start, start + chunk) for start, chunk in zip(chunk_range, zarr_array.chunks))
    chunk_data = zarr_array[slices]
    
    for layer_idx in range(chunk_data.shape[0]):
        layer_data = chunk_data[layer_idx].flatten()
        layer_data = layer_data[layer_data != 0]  # Exclude background
        unique_ids = np.unique(layer_data)
        
        for seg_id in unique_ids:
            if seg_id not in layer_counts:
                layer_counts[seg_id] = {}
            layer_counts[seg_id][chunk_range[0] + layer_idx] = 1
    
    return layer_counts

def get_chunk_ranges(shape, chunk_sizes):
    """
    Calculate the ranges of indices for each chunk.
    
    Parameters:
    - shape: Shape of the Zarr array.
    - chunk_sizes: Sizes of the chunks.
    
    Returns:
    - A numpy array of chunk starting indices.
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
                chunk_range = futures[future]
                logging.info(f"Processing chunk: {chunk_range}")
                progress.advance(task)
                try:
                    chunk_result = future.result()
                    logging.info(f"Finished processing chunk: {chunk_range}")
                    for seg_id, layers in chunk_result.items():
                        if seg_id not in combined_layer_counts:
                            combined_layer_counts[seg_id] = {}
                        combined_layer_counts[seg_id].update(layers)
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_range}: {e}")

    final_counts = {seg_id: len(layers) for seg_id, layers in combined_layer_counts.items()}

    # Save the combined counts to a pickle file
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(dict(final_counts), f)

    message = f"Total unique segments counted: {len(final_counts)}"
    logging.info(message)
    print(message)
    logging.info(f"Total processing time: {time.time() - start} seconds")
    exit(0)

''' Usage: 
python segmentation_counts_chunkwise.py \
--input_zarr /data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL-Soma.zarr \
--dataset segmentation_0.2 \
--num_cores 64 \
--output_pickle segmentation_0.2_counts.pkl \
--log_path log
'''
