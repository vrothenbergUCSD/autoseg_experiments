# Probably more efficient with compute-optimized instance (64 cores)
# Currently using AWS_LSD_m5a.8xlarge 32 vCPU, only 14/128 GB RAM used

from multiprocessing import Pool, cpu_count, Manager
import numpy as np
import zarr
import pickle
import logging
from tqdm import tqdm
import time
import argparse 
import os
from rich.progress import Progress
from concurrent.futures import ProcessPoolExecutor, as_completed



# Setup argument parser
parser = argparse.ArgumentParser(description='Remap segmentation IDs in a Zarr dataset.')
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr file.')
parser.add_argument('--input_dataset', type=str, required=True, help='Input dataset name within the Zarr file.')
# parser.add_argument('--output_zarr', type=str, required=True, help='Path to output Zarr file.')
parser.add_argument('--output_dataset', type=str, required=True, help='Output dataset name within the Zarr file.')
parser.add_argument('--num_cores', type=int, default=16, help='Number of cores to use for processing.')
parser.add_argument('--pickle_file', type=str, default='combined_counts.pkl', help='Pickle file name for combined counts.')
parser.add_argument('--log_path', type=str, default='log', help='Directory for log files')
args = parser.parse_args()

# Initialize logging
logging.basicConfig(filename=f'{args.log_path}/remap_ids.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


        
def find_unique_ids_in_layer(zarr_array, layer_index):
    layer = zarr_array[layer_index, :, :]
    unique_ids = np.unique(layer)
    logging.info(f"Found {len(unique_ids)} unique IDs in layer {layer_index}")
    return set(unique_ids)


def parallelize_unique_id_search(zarr_array, num_cores):
    # pickle_path = f'unique_IDs.pkl'
    # if os.path.exists(pickle_path):
    #     logging.info("Opening pickle of unique IDs.")
    #     with open(pickle_path, 'rb') as f:
    #         return pickle.load(f)
        
    total_layers = zarr_array.shape[0]
    all_unique_ids = set()

    # Setup the rich progress bar
    with Progress() as progress:
        task = progress.add_task("[green]Counting Unique IDs...", total=total_layers)
        
        # Use ProcessPoolExecutor to manage the pool of worker processes
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all layers as separate tasks
            futures = [executor.submit(find_unique_ids_in_layer, zarr_array, layer_idx) for layer_idx in range(total_layers)]
            
            # Monitor for task completion and update progress accordingly
            for future in as_completed(futures):
                unique_ids = future.result()  # Collect result from each completed task
                progress.advance(task)  # Update progress bar per layer processed
                all_unique_ids.update(unique_ids)
    

    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(all_unique_ids, f)
    return all_unique_ids

# def vectorized_remap(layer, id_map):
#     all_old_ids = np.array(list(id_map.keys()))
#     new_ids = np.array(list(id_map.values()))
    
#     # Sort old IDs and corresponding new IDs
#     sorted_indices = np.argsort(all_old_ids)
#     sorted_old_ids = all_old_ids[sorted_indices]
#     sorted_new_ids = new_ids[sorted_indices]
    
#     # Flatten layer for processing
#     flat_layer = layer.ravel()
    
#     # Find positions of layer IDs in the sorted list of old IDs
#     idx = np.searchsorted(sorted_old_ids, flat_layer)
    
#     # Ensure only valid indices are mapped
#     valid_idx = idx < len(sorted_old_ids)
#     valid_flat_layer = flat_layer[valid_idx]
#     valid_idx = idx[valid_idx]
    
#     # Use np.take to map each old ID to its new ID, ensuring only valid mappings are applied
#     remapped_flat_layer = np.copy(flat_layer)  # Start with a copy to preserve the original data
#     remapped_flat_layer[valid_idx] = np.take(sorted_new_ids, valid_idx)
    
#     # Reshape back to the original layer shape
#     remapped_layer = remapped_flat_layer.reshape(layer.shape)
    
#     return remapped_layer

def vectorized_remap(layer, id_map):
    all_old_ids = np.array(list(id_map.keys()))
    new_ids = np.array(list(id_map.values()))
    
    # Sort old IDs and corresponding new IDs
    sorted_indices = np.argsort(all_old_ids)
    sorted_old_ids = all_old_ids[sorted_indices]
    sorted_new_ids = new_ids[sorted_indices]
    
    # Flatten layer for processing
    flat_layer = layer.ravel()
    
    # Find positions of layer IDs in the sorted list of old IDs
    idx = np.searchsorted(sorted_old_ids, flat_layer, side='left')
    
    # Ensure only valid indices are mapped
    valid_mask = idx < len(sorted_old_ids)  # Ensure indices are within bounds
    valid_flat_layer = flat_layer[valid_mask]
    valid_idx = idx[valid_mask]
    
    # Further ensure that the IDs match exactly (to handle IDs not in old_ids)
    actual_matches = sorted_old_ids[valid_idx] == valid_flat_layer
    final_valid_idx = valid_idx[actual_matches]
    final_valid_flat_layer = valid_flat_layer[actual_matches]
    
    # Map to new IDs
    remapped_flat_layer = np.copy(flat_layer)  # Start with a copy to preserve original data
    remapped_flat_layer[valid_mask][actual_matches] = sorted_new_ids[final_valid_idx]
    
    # Reshape back to the original layer shape
    remapped_layer = remapped_flat_layer.reshape(layer.shape)
    
    return remapped_layer


def remap_chunk(chunk_range, zarr_input, zarr_output, id_map):
    """
    Process a given chunk range, applying the remapping function.
    """
    start = time.time()
    # Calculate slice objects for this chunk
    slices = tuple(slice(start, min(start + size, end)) for start, size, end in zip(chunk_range, zarr_input.chunks, zarr_input.shape))

    # Load the chunk, process it, and save the result
    chunk_data = zarr_input[slices]
    remapped_chunk = vectorized_remap(chunk_data, id_map)  # Your remap function here
    zarr_output[slices] = remapped_chunk
    elapsed = round(time.time() - start,2)
    logging.info(f"Completed remapping for chunk {tuple(chunk_range)}. Elapsed: {elapsed}s")

def get_chunk_ranges(shape, chunk_size):
    """
    Calculate the ranges of indices for each chunk.
    """
    ranges = [range(0, s, cs) for s, cs in zip(shape, chunk_size)]
    return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(shape), -1).T

def parallelize_remap_layers(zarr_array, id_map, new_zarr_array, num_cores):
    total_layers = zarr_array.shape[0]
    chunk_ranges = get_chunk_ranges(zarr_array.shape, zarr_array.chunks)
    total_chunks = len(chunk_ranges)

    with Progress() as progress:
        task = progress.add_task("[green]Remapping Layers...", total=total_chunks)
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(remap_chunk, chunk_range, zarr_array, new_zarr_array, id_map): chunk_range
                  for chunk_range in chunk_ranges}
            
            for future in as_completed(futures):
                chunk_range = futures[future]  # Retrieve the layer chunk for the completed task
                try:
                    future.result()  # Optionally handle result or exceptions
                except Exception as exc:
                    logging.error(f'Chunk {chunk_range} generated an exception: {exc}')
                progress.advance(task)


if __name__ == '__main__':
    logging.info(f"Reading Zarr file from {args.input_zarr}")
    # sync_path = os.path.join(args.output_zarr, args.output_dataset + '.sync')
    # # sync_path = args.output_zarr + '.sync'
    # logging.info(f"Sync path: {sync_path}")
    # if os.path.exists(sync_path): os.remove(sync_path)
    # synchronizer = zarr.ProcessSynchronizer(sync_path)
    synchronizer = zarr.ThreadSynchronizer()
    
    zarr_object = zarr.open(args.input_zarr, mode='a', synchronizer=synchronizer)
    if args.input_dataset not in zarr_object:
        logging.error(f"Dataset {args.input_dataset} not found in Zarr file.")
        exit(1)
    zarr_array = zarr_object[args.input_dataset]

    logging.info("Opened segmentation zarr array")

    

    # new_zarr_group = zarr.open(args.output_zarr, mode='r+', synchronizer=synchronizer)

    # Delete if already exists
    if args.output_dataset in zarr_object:
        logging.info(f'Deleting existing dataset {args.output_dataset}')
        del zarr_object[args.output_dataset]
    new_zarr_array = zarr_object.create_dataset(args.output_dataset, 
                                                    shape=zarr_array.shape, 
                                                    chunks=zarr_array.chunks, 
                                                    dtype=zarr_array.dtype,
                                                    synchronizer=synchronizer)
    logging.info(f'Initialized {args.output_dataset} in {args.input_zarr} with shape {new_zarr_array.shape} and dtype {new_zarr_array.dtype}')

    num_cores = args.num_cores
    logging.info(f"Number of cores: {num_cores}")

    unique_ids = parallelize_unique_id_search(zarr_array, num_cores)
    logging.info(f"Total number of unique IDs: {len(unique_ids)}")
    print(f"Total number of unique IDs: {len(unique_ids)}")
    id_map = {id: idx for idx, id in enumerate(unique_ids)}

    parallelize_remap_layers(zarr_array, id_map, new_zarr_array, num_cores)
    logging.info('Completed remapping of IDs')
    print('Completed remapping of IDs')
    exit(0)
