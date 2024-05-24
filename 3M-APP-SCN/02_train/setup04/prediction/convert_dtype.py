from multiprocessing import Pool, cpu_count, Manager
import numpy as np
import zarr
import logging
from tqdm import tqdm
import time

# Initialize logging
logging.basicConfig(filename='convert_dtype.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def convert_dtype(args):
  start_layer, end_layer, source_array, dest_array, progress_bar = args
  try:
    for layer_idx in range(start_layer, end_layer):
      logging.info(f"Processing layer {layer_idx} in range {start_layer}-{end_layer}")
      layer = source_array[layer_idx, :, :]
      dest_array[layer_idx, :, :] = layer.astype(np.uint32)
      progress_bar.value += 1
  except Exception as e:
    logging.error("Exception occurred", exc_info=True)

# Open source Zarr array
source_group = zarr.open('3M-APP-SCN_remapped.zarr', mode='r')
source_array = source_group['segmentation_0.1']

# Create destination Zarr array with dtype np.uint32
dest_group = zarr.open('3M-APP-SCN_remapped_32bit.zarr', mode='a')
dest_array = dest_group.create_dataset('segmentation_0.1', shape=source_array.shape, chunks=source_array.chunks, dtype=np.uint32, compressor=source_array.compressor)

# Number of cores
num_cores = min(cpu_count(), 15)
logging.info(f"Number of cores: {num_cores}")

# Calculate layers per core
total_layers = source_array.shape[0]
layers_per_core = total_layers // num_cores
remainder = total_layers % num_cores

with Manager() as manager:
  progress_bar = manager.Value('i', 0)  # shared counter

  args = []
  start_layer = 0
  for i in range(num_cores):
    end_layer = start_layer + layers_per_core
    if i < remainder:
      end_layer += 1
    args.append((start_layer, end_layer, source_array, dest_array, progress_bar))
    start_layer = end_layer

  # Create ThreadPool and run
  try:
    with Pool(num_cores) as pool:
      async_result = pool.map_async(convert_dtype, args)
      # Initialize tqdm progress bar
      with tqdm(total=total_layers, position=0, leave=True) as pbar:
        last_count = 0
        # Update tqdm while the processes are running
        while not async_result.ready():
          current_count = progress_bar.value
          pbar.update(current_count - last_count)
          last_count = current_count
          time.sleep(0.5)
        # Retrieve results when done
        results = async_result.get()
  except Exception as e:
    logging.error("Exception occurred in ThreadPool", exc_info=True)

logging.info('Completed dtype conversion of 3M-APP-SCN_remapped.zarr')
