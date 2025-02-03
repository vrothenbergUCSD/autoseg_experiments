import numpy as np
import zarr
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Manager

# Set up logging
logging.basicConfig(filename='create_masks.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to process a chunk
def process_chunk(args):
    z_start, z_end, y_start, y_end, x_start, x_end, labels_name, labels_mask_name, data_path = args
    try:
        logging.info(f'Starting processing for chunk: Z({z_start}:{z_end}), Y({y_start}:{y_end}), X({x_start}:{x_end})')
        data = zarr.open(data_path, mode='a')
        labels = data[labels_name]
        labels_mask = data[labels_mask_name]

        for z in range(z_start, z_end):
            layer = labels[z, y_start:y_end, x_start:x_end]
            mask = np.where(layer > 0, 1, 0).astype(np.uint8)
            labels_mask[z, y_start:y_end, x_start:x_end] = mask

        return True
    except Exception as e:
        logging.error(f'Error processing chunk: Z({z_start}:{z_end}), Y({y_start}:{y_end}), X({x_start}:{x_end}), Error: {str(e)}')
        return False

# Function to create masks for large datasets in parallel
def create_masks(data_path, labels_name, labels_mask_name):
    data = zarr.open(data_path, mode='a')
    labels = data[labels_name]

    with Manager() as manager:
        # Prepare the arguments for parallel processing
        z_chunk_size, y_chunk_size, x_chunk_size = labels.chunks
        shape = labels.shape
        
        # Create an empty labels_mask dataset in the zarr container with the same shape as labels but with dtype=np.uint8
        if labels_mask_name not in data:
            data.create_dataset(labels_mask_name, shape=shape, dtype=np.uint8, chunks=labels.chunks)
        
        labels_mask = data[labels_mask_name]
        labels_mask.attrs["offset"] = labels.attrs["offset"]
        labels_mask.attrs["resolution"] = labels.attrs["resolution"]

        args = [(z, min(z + z_chunk_size, shape[0]), 
                 y, min(y + y_chunk_size, shape[1]), 
                 x, min(x + x_chunk_size, shape[2]), 
                 labels_name, labels_mask_name, data_path)
                for z in range(0, shape[0], z_chunk_size)
                for y in range(0, shape[1], y_chunk_size)
                for x in range(0, shape[2], x_chunk_size)]

        total_chunks = len(args)

        # Parallel processing of the chunks
        logging.info(f'Starting parallel processing of chunks for dataset: {labels_name}')
        
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_chunk, arg): arg for arg in args}
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                pbar.update(1)
                        except Exception as e:
                            logging.error(f'Error processing a chunk: {str(e)}')
            except Exception as e:
                logging.error(f'Error during parallel processing of dataset: {labels_name}, Error: {str(e)}')

        logging.info(f'Completed parallel processing of all chunks for dataset: {labels_name}')

    print("Mask creation complete. See create_masks.log for details.")

# Example usage
data_path = 'CompImBio800M.zarr'
labels_name = 'labels'
labels_mask_name = 'labels_mask'

workers = 8

logging.info(f"Opening: {data_path}")
logging.info(f"Dataset: {labels_name}")

create_masks(data_path, labels_name, labels_mask_name)
