import zarr
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress
from zarr import ProcessSynchronizer
import argparse
from multiprocessing import shared_memory
import os
import signal
import sys

# Function to process a chunk
def process_chunk(args):
    chunk_range, dataset_name, output_dataset_name, data_path, sync_path, shared_ids_name, shared_ids_shape = args
    try:
        synchronizer = ProcessSynchronizer(sync_path)
        data = zarr.open(data_path, mode='r+', synchronizer=synchronizer)
        dataset = data[dataset_name]

        # Access the shared memory for unique_ids
        existing_shm = shared_memory.SharedMemory(name=shared_ids_name)
        unique_ids = np.ndarray(shared_ids_shape, dtype=np.int64, buffer=existing_shm.buf)

        # Build the id_map
        id_map = {id_: idx for idx, id_ in enumerate(unique_ids)}

        # Calculate slice objects for this chunk
        slices = tuple(slice(start, min(start + size, end)) for start, size, end in zip(chunk_range, dataset.chunks, dataset.shape))

        # Read the chunk
        chunk = dataset[slices]

        # Remap the chunk
        remapped_chunk = vectorized_remap(chunk, id_map)

        # Write the remapped chunk to the new dataset
        output_dataset = data[output_dataset_name]
        output_dataset[slices] = remapped_chunk

        return True
    except Exception as e:
        logging.error(f'Error processing chunk {chunk_range} of dataset {dataset_name}: {str(e)}')
        return False

# Function to find unique IDs in a chunk
def find_unique_ids_in_chunk(args):
    chunk_range, dataset_name, data_path, sync_path = args
    try:
        synchronizer = ProcessSynchronizer(sync_path)
        data = zarr.open(data_path, mode='r+', synchronizer=synchronizer)
        dataset = data[dataset_name]

        # Calculate slice objects for this chunk
        slices = tuple(slice(start, min(start + size, end)) for start, size, end in zip(chunk_range, dataset.chunks, dataset.shape))

        # Read the chunk
        chunk = dataset[slices]

        # Find unique IDs in the chunk
        unique_ids = np.unique(chunk)
        
        return unique_ids
    except Exception as e:
        logging.error(f'Error finding unique IDs in chunk {chunk_range} of dataset {dataset_name}: {str(e)}')
        return np.array([], dtype=np.int64)

# Function to remap a given dataset in a Zarr store
def remap_dataset(data_path, dataset_name, output_dataset_name, sync_path, num_cores, unique_ids):
    data = zarr.open(data_path, mode='r+')
    dataset = data[dataset_name]

    if output_dataset_name in data:
        logging.info(f'Deleting existing {output_dataset_name} dataset')
        del data[output_dataset_name]

    # Create a new dataset for the remapped data
    output_dataset = data.create_dataset(output_dataset_name, 
                                         shape=dataset.shape, 
                                         dtype=dataset.dtype, 
                                         chunks=dataset.chunks,
                                         compressor=dataset.compressor,
                                         synchronizer=ProcessSynchronizer(sync_path))

    # Create shared memory for unique_ids
    shm = shared_memory.SharedMemory(create=True, size=unique_ids.nbytes)
    shared_ids = np.ndarray(unique_ids.shape, dtype=unique_ids.dtype, buffer=shm.buf)
    np.copyto(shared_ids, unique_ids)

    # Prepare the arguments for parallel processing
    chunk_ranges = get_chunk_ranges(dataset.shape, dataset.chunks)
    total_chunks = len(chunk_ranges)

    logging.info(f'Total number of chunks to process: {total_chunks}')

    # Signal handler for graceful termination
    def signal_handler(sig, frame):
        logging.info("KeyboardInterrupt received. Cleaning up...")
        shm.close()
        shm.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parallel processing of the chunks
    logging.info(f'Starting parallel processing of chunks for dataset: {dataset_name}')
    
    with Progress() as progress:
        task = progress.add_task("Remapping Dataset...", total=total_chunks)
        try:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                args_list = [
                    (chunk_range, dataset_name, output_dataset_name, data_path, sync_path, shm.name, unique_ids.shape)
                    for chunk_range in chunk_ranges
                ]

                futures = {executor.submit(process_chunk, arg): arg for arg in args_list}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            progress.advance(task)
                        else:
                            logging.error(f"Chunk failed: {future}")
                    except Exception as e:
                        logging.error(f'Error processing a chunk: {str(e)}')
        except Exception as e:
            logging.error(f'Error during parallel processing of dataset: {dataset_name}, Error: {str(e)}')

    # Clean up shared memory
    shm.close()
    shm.unlink()

    logging.info(f'Completed parallel processing of all chunks for dataset: {dataset_name}')
    # Verify the shape of the new dataset
    logging.info(f"Remapped shape of {output_dataset_name}: {output_dataset.shape}")
    logging.info(f"Data type of {output_dataset_name}: {output_dataset.dtype}")

    print(f"Remapped shape of {output_dataset_name}:", output_dataset.shape)
    print(f"Data type of {output_dataset_name}:", output_dataset.dtype)

# Function to calculate chunk ranges
def get_chunk_ranges(shape, chunk_size):
    ranges = [range(0, s, cs) for s, cs in zip(shape, chunk_size)]
    return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(shape), -1).T

# Function to parallelize unique ID search
def parallelize_unique_id_search(data_path, dataset_name, sync_path, num_cores):
    data = zarr.open(data_path, mode='r+')
    dataset = data[dataset_name]
    chunk_ranges = get_chunk_ranges(dataset.shape, dataset.chunks)
    total_chunks = len(chunk_ranges)

    all_unique_ids = set()

    with Progress() as progress:
        task = progress.add_task("Finding Unique IDs...", total=total_chunks)
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            args_list = [
                (chunk_range, dataset_name, data_path, sync_path)
                for chunk_range in chunk_ranges
            ]

            futures = {executor.submit(find_unique_ids_in_chunk, arg): arg for arg in args_list}

            for future in as_completed(futures):
                try:
                    unique_ids = future.result()
                    all_unique_ids.update(unique_ids)
                    progress.advance(task)
                except Exception as e:
                    logging.error(f'Error finding unique IDs in a chunk: {str(e)}')

    return np.array(sorted(all_unique_ids), dtype=np.int64)

def vectorized_remap(layer, id_map):
    all_old_ids = np.array(list(id_map.keys()))
    new_ids = np.array(list(id_map.values()))

    if len(all_old_ids) == 0:  # No IDs to remap
        return layer

    sorted_indices = np.argsort(all_old_ids)
    sorted_old_ids = all_old_ids[sorted_indices]
    sorted_new_ids = new_ids[sorted_indices]

    flat_layer = layer.ravel()
    idx = np.searchsorted(sorted_old_ids, flat_layer, side='left')

    # Use boolean masking to filter valid indices and matches
    valid_mask = (idx < len(sorted_old_ids)) & (sorted_old_ids[idx] == flat_layer)

    # Directly modify the flattened layer
    flat_layer[valid_mask] = sorted_new_ids[idx[valid_mask]]

    return flat_layer.reshape(layer.shape)

# Setup argument parser
parser = argparse.ArgumentParser(description='Remap segmentation IDs in a Zarr dataset.')
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr file.')
parser.add_argument('--input_dataset', type=str, required=True, help='Input dataset name within the Zarr file.')
parser.add_argument('--output_dataset', type=str, required=True, help='Output dataset name within the Zarr file.')
parser.add_argument('--sync_path', type=str, required=True, help='Path to synchronization file.')
parser.add_argument('--num_cores', type=int, default=16, help='Number of cores to use for processing.')
parser.add_argument('--log_path', type=str, default='log', help='Directory for log files')
args = parser.parse_args()

# Initialize logging
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

logging.basicConfig(filename=f'{args.log_path}/remap_ids_chunkwise.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    logging.info(f"Reading Zarr file from {args.input_zarr}")

    sync_path = args.sync_path
    
    zarr_object = zarr.open(args.input_zarr, mode='a', synchronizer=ProcessSynchronizer(sync_path))
    if args.input_dataset not in zarr_object:
        logging.error(f"Dataset {args.input_dataset} not found in Zarr file.")
        exit(1)
    zarr_array = zarr_object[args.input_dataset]

    logging.info("Opened segmentation zarr array")

    if args.output_dataset in zarr_object:
        logging.info(f'Deleting existing dataset {args.output_dataset}')
        del zarr_object[args.output_dataset]
        
    new_zarr_array = zarr_object.create_dataset(args.output_dataset, 
                                                shape=zarr_array.shape, 
                                                chunks=zarr_array.chunks, 
                                                dtype=zarr_array.dtype,
                                                synchronizer=ProcessSynchronizer(sync_path))
    logging.info(f'Initialized {args.output_dataset} in {args.input_zarr} with shape {new_zarr_array.shape} and dtype {new_zarr_array.dtype}')

    num_cores = args.num_cores
    logging.info(f"Number of cores: {num_cores}")

    unique_ids = parallelize_unique_id_search(args.input_zarr, args.input_dataset, sync_path, num_cores)
    logging.info(f"Total number of unique IDs: {len(unique_ids)}")
    print(f"Total number of unique IDs: {len(unique_ids)}")

    remap_dataset(args.input_zarr, args.input_dataset, args.output_dataset, sync_path, num_cores, unique_ids)
    logging.info('Completed remapping of IDs')
    print('Completed remapping of IDs')
    exit(0)

'''
Usage:
python remap_ids_chunkwise.py \
--input_zarr /data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL-Soma.zarr \
--input_dataset segmentation_0.2_filtered \
--output_dataset segmentation_0.2_remapped \
--sync_path data.sync \
--num_cores 64 \
--log_path log
'''
