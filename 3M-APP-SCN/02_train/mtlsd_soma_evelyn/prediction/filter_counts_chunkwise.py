import os
import zarr
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress
from zarr import ProcessSynchronizer
import pickle
import argparse
from multiprocessing import shared_memory, Array


# Function to process a chunk
def process_chunk(args):
    z_start, z_end, y_start, y_end, x_start, x_end, dataset_name, output_dataset_name, data_path, out_data_path, sync_path, shared_ids_name, shared_ids_shape = args
    try:
        synchronizer = ProcessSynchronizer(sync_path)
        data = zarr.open(data_path, mode='r+', synchronizer=synchronizer)
        dataset = data[dataset_name]

        # Access the shared memory for ids_to_remove
        existing_shm = shared_memory.SharedMemory(name=shared_ids_name)
        ids_to_remove = np.ndarray(shared_ids_shape, dtype=np.int64, buffer=existing_shm.buf)

        # Read the chunk
        chunk = dataset[z_start:z_end, y_start:y_end, x_start:x_end]

        # Filter the chunk
        filtered_chunk = np.where(np.isin(chunk, ids_to_remove), 0, chunk)

        # Write the filtered chunk to the new dataset
        output_dataset = data[output_dataset_name]
        output_dataset[z_start:z_end, y_start:y_end, x_start:x_end] = filtered_chunk

        return True
    except Exception as e:
        logging.error(f'Error processing chunk: Z({z_start}:{z_end}), Y({y_start}:{y_end}), X({x_start}:{x_end}) of dataset: {dataset_name}, Error: {str(e)}')
        return False

# Function to filter a given dataset in a Zarr store
def filter_dataset(data_path, dataset_name, output_zarr, output_dataset_name, sync_path, counts_pickle_path, threshold, num_cores):
    data = zarr.open(data_path, mode='r+')
    out_data = zarr.open(output_zarr, mode='a')
    dataset = data[dataset_name]

    if output_dataset_name in out_data:
        logging.info(f'Deleting existing {output_dataset_name} dataset')
        del out_data[output_dataset_name]

    # Create a new dataset for the filtered data
    output_dataset = out_data.create_dataset(output_dataset_name, 
                                             shape=dataset.shape, 
                                             dtype=dataset.dtype, 
                                             chunks=dataset.chunks,
                                             compressor=dataset.compressor,
                                             synchronizer=ProcessSynchronizer(sync_path))

    # Load segmentation counts
    with open(counts_pickle_path, 'rb') as f:
        segmentation_counts = pickle.load(f)

    # Identify IDs to remove and create shared memory for efficient lookup
    ids_to_remove = np.array([seg_id for seg_id, count in segmentation_counts.items() if count < threshold], dtype=np.int64)
    logging.info(f"Identified {len(ids_to_remove)} IDs to remove based on the threshold {threshold}")
    print(f"Removing {len(ids_to_remove)} IDs")

    # Create a shared memory block for ids_to_remove
    shm = shared_memory.SharedMemory(create=True, size=ids_to_remove.nbytes)
    shared_ids_to_remove = np.ndarray(ids_to_remove.shape, dtype=ids_to_remove.dtype, buffer=shm.buf)
    np.copyto(shared_ids_to_remove, ids_to_remove)

    # Prepare the arguments for parallel processing
    z_chunk_size, y_chunk_size, x_chunk_size = dataset.chunks
    args = [(z, min(z + z_chunk_size, dataset.shape[0]), 
             y, min(y + y_chunk_size, dataset.shape[1]), 
             x, min(x + x_chunk_size, dataset.shape[2]), 
             dataset_name, output_dataset_name, data_path, output_zarr, sync_path, shm.name, ids_to_remove.shape)
            for z in range(0, dataset.shape[0], z_chunk_size)
            for y in range(0, dataset.shape[1], y_chunk_size)
            for x in range(0, dataset.shape[2], x_chunk_size)]

    total_chunks = len(args)

    logging.info(f'Total number of chunks to process: {total_chunks}')

    # Parallel processing of the chunks
    logging.info(f'Starting parallel processing of chunks for dataset: {dataset_name}')
    
    with Progress() as progress:
        task = progress.add_task("Filtering...", total=total_chunks)
        try:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {executor.submit(process_chunk, arg): arg for arg in args}
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
    logging.info(f"Filtered shape of {output_dataset_name}: {output_dataset.shape}")
    logging.info(f"Data type of {output_dataset_name}: {output_dataset.dtype}")

    print(f"Filtered shape of {output_dataset_name}:", output_dataset.shape)
    print(f"Data type of {output_dataset_name}:", output_dataset.dtype)

# Setup argument parser
parser = argparse.ArgumentParser(description='Filter objects from segmentation datasets below a threshold.')
parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr.')
parser.add_argument('--input_dataset', type=str, required=True, help='Name of input Zarr dataset to filter IDs.')
parser.add_argument('--output_zarr', type=str, required=True, help='Path to output Zarr.')
parser.add_argument('--output_dataset', type=str, required=True, help='Name of new output Zarr dataset.')
parser.add_argument('--sync_path', type=str, required=True, help='Path to synchronization Zarr.')
parser.add_argument('--counts_pickle_path', type=str, required=True, help='Path to pickle file with segmentation counts.')
parser.add_argument('--threshold', type=int, default=5, help='Threshold for filtering objects.')
parser.add_argument('--num_cores', type=int, default=10, help='Number of cores to use for parallel multi-threading.')
parser.add_argument('--log_path', type=str, default='log', help='Directory for log files')
args = parser.parse_args()

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

# Set up logging
logging.basicConfig(filename=f'{args.log_path}/filter_counts_chunkwise.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


logging.info(args)

# Call the function to filter the dataset
filter_dataset(args.input_zarr, 
               args.input_dataset, 
               args.output_zarr,
               args.output_dataset, 
               args.sync_path, 
               args.counts_pickle_path, 
               args.threshold, 
               args.num_cores)
