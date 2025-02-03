import zarr
import os
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import time
import logging
import pickle
import argparse 
from collections import Counter

parser = argparse.ArgumentParser(description="Count segmentation objects in parallel and save the counts.")
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr file.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name within the Zarr file.')
parser.add_argument('--num_cores', type=int, default=min(cpu_count(), 15), help='Number of cores to use for processing. Default is min(system cores, 15).')
parser.add_argument('--output_pickle', type=str, default='combined_counts.pkl', help='Output pickle file name for combined counts.')
args = parser.parse_args()

logging.basicConfig(filename='segmentation_counts.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def count_segmentation_for_layers(args):
    zarr_array, start_layer, end_layer, progress_bar = args
    segmentation_counts = Counter()
    for layer_index in range(start_layer, end_layer):
        logging.info(f"Processing layer {layer_index} in range {start_layer}-{end_layer}")
        layer = zarr_array[layer_index, :, :].flatten()
        layer = layer[layer != 0] # Exclude background
        # unique_segmentation_ids, counts = np.unique(filtered_layer, return_counts=True)
        unique_segmentation_ids = np.unique(layer)
        segmentation_counts.update(unique_segmentation_ids)
        # for seg_id in unique_segmentation_ids:
        #     if seg_id != 0:
        #         segmentation_counts[seg_id] = segmentation_counts.get(seg_id, 0) + 1
        progress_bar.value += 1  # Update the shared counter directly
    return dict(segmentation_counts)

def get_segmentation_counts_parallel(zarr_array, num_cores):
    total_layers = zarr_array.shape[0]
    layers_per_core = total_layers // num_cores
    remainder = total_layers % num_cores

    with Manager() as manager:
        progress_bar = manager.Value('i', 0)  # shared counter

        # Distribute layers among cores
        args = []
        start_layer = 0
        for i in range(num_cores):
            end_layer = start_layer + layers_per_core
            if i < remainder:
                end_layer += 1  # give one extra layer to this core
            args.append((zarr_array, start_layer, end_layer, progress_bar))
            start_layer = end_layer

        logging.info(f'Layers per core: {layers_per_core}, with {remainder} cores processing an extra layer.')

        # Process in parallel
        with Pool(num_cores) as pool:
            async_result = pool.map_async(count_segmentation_for_layers, args)

            # Update tqdm while the processes are running
            with tqdm(total=zarr_array.shape[0], position=0, leave=True) as pbar:
                last_count = 0
                while not async_result.ready():
                    current_count = progress_bar.value
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    time.sleep(0.5)
                results = async_result.get()

    # print('Saving pool results to pickle.')
    # # Save to pickle file
    # with open('pool_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # Combine results
    logging.info('Combining results')
    # combined_counts = {}
    # for segmentation_counts in results:
    #     for seg_id, count in segmentation_counts.items():
    #         combined_counts[seg_id] = combined_counts.get(seg_id, 0) + count

    combined_counts = Counter()
    for segmentation_counts in results:
        combined_counts.update(segmentation_counts)

    logging.info(f'Total count: {len(combined_counts)}')


    # Save to pickle file
    with open('combined_counts.pkl', 'wb') as f:
        pickle.dump(combined_counts, f)
    return combined_counts



if __name__ == '__main__':
    logging.info(f"Reading Zarr file from {args.input_zarr}")
    data = zarr.open(args.input_zarr, mode='r')
    if args.dataset not in data:
        logging.error(f"Dataset {args.dataset} not found in Zarr file.")
        exit(1)
    zarr_array = data[args.dataset]
    logging.info(f"Zarr array shape: {zarr_array.shape}")
    logging.info(f"Number of cores: {args.num_cores}")
    segmentation_counts = get_segmentation_counts_parallel(zarr_array, args.num_cores)
