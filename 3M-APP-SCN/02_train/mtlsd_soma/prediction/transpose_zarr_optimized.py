import os
import zarr
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from zarr import ProcessSynchronizer
import argparse


# Function to process a chunk
def process_chunk(args):
    z_start, z_end, y_start, y_end, x_start, x_end, dataset_name, output_dataset_name, data_path, sync_path = args
    try:
        logging.info(f'Starting processing for chunk: Z({z_start}:{z_end}), Y({y_start}:{y_end}), X({x_start}:{x_end}) of dataset: {dataset_name}')
        synchronizer = ProcessSynchronizer(sync_path)
        data = zarr.open(data_path, mode='r+', synchronizer=synchronizer)
        dataset = data[dataset_name]

        # Read the chunk
        chunk = dataset[z_start:z_end, y_start:y_end, x_start:x_end]

        # Transpose the chunk
        transposed_chunk = chunk.transpose(2, 1, 0)

        # Write the transposed chunk to the new dataset
        output_dataset = data[output_dataset_name]
        output_dataset[x_start:x_end, y_start:y_end, z_start:z_end] = transposed_chunk

        logging.info(f'Finished processing for chunk: Z({z_start}:{z_end}), Y({y_start}:{y_end}), X({x_start}:{x_end}) of dataset: {dataset_name}')
        return True
    except Exception as e:
        logging.error(f'Error processing chunk: Z({z_start}:{z_end}), Y({y_start}:{y_end}), X({x_start}:{x_end}) of dataset: {dataset_name}, Error: {str(e)}')
        return False

# Function to transpose a given dataset in a Zarr store
def transpose_dataset(data_path, dataset_name, output_dataset_name, sync_path, num_cores):
    data = zarr.open(data_path, mode='r+')
    dataset = data[dataset_name]

    if output_dataset_name in data:
        logging.info(f'Deleting existing {output_dataset_name} dataset')
        del data[output_dataset_name]

    # Create a new dataset for the transposed data
    output_dataset = data.create_dataset(output_dataset_name, 
                                         shape=(dataset.shape[2], dataset.shape[1], dataset.shape[0]), 
                                         dtype=dataset.dtype, 
                                         chunks=(dataset.chunks[2], dataset.chunks[1], dataset.chunks[0]),
                                         synchronizer=ProcessSynchronizer(sync_path))

    # Prepare the arguments for parallel processing
    z_chunk_size, y_chunk_size, x_chunk_size = dataset.chunks
    args = [(z, min(z + z_chunk_size, dataset.shape[0]), 
             y, min(y + y_chunk_size, dataset.shape[1]), 
             x, min(x + x_chunk_size, dataset.shape[2]), 
             dataset_name, output_dataset_name, data_path, sync_path)
            for z in range(0, dataset.shape[0], z_chunk_size)
            for y in range(0, dataset.shape[1], y_chunk_size)
            for x in range(0, dataset.shape[2], x_chunk_size)]

    total_chunks = len(args)

    # Parallel processing of the chunks
    logging.info(f'Starting parallel processing of chunks for dataset: {dataset_name}')
    
    with tqdm(total=total_chunks, desc="Transposing") as pbar:
        try:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {executor.submit(process_chunk, arg): arg for arg in args}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            pbar.update(1)
                    except Exception as e:
                        logging.error(f'Error processing a chunk: {str(e)}')
        except Exception as e:
            logging.error(f'Error during parallel processing of dataset: {dataset_name}, Error: {str(e)}')

    logging.info(f'Completed parallel processing of all chunks for dataset: {dataset_name}')
    # Verify the shape of the new dataset
    logging.info(f"Transposed shape of {output_dataset_name}: {output_dataset.shape}")
    logging.info(f"Data type of {output_dataset_name}: {output_dataset.dtype}")

    print(f"Transposed shape of {output_dataset_name}:", output_dataset.shape)
    print(f"Data type of {output_dataset_name}:", output_dataset.dtype)




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

logging.basicConfig(filename=f'{args.log_path}/transpose_zarr_optimized.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

# # Example usage
# data_path = '/data/base/3M-APP-SCN/02_train/mtlsd_soma/prediction/SCN_DL_12AM_VL-Soma.zarr'
# sync_path = 'data.sync'
# dataset_name = 'segmentation_0.2_remapped'  # Replace with any dataset name you want to process
# output_dataset_name = 'segmentation_0.2_xyz'  # Specify the output dataset name

transpose_dataset(args.input_zarr, 
                  args.input_dataset, 
                  args.output_dataset, 
                  args.sync_path, 
                  args.num_cores)
