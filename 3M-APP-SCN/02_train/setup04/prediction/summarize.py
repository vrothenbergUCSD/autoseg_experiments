from multiprocessing import Pool, cpu_count, Manager, Lock
import numpy as np
import zarr
import pickle
import logging
from tqdm import tqdm
import time

# Initialize logging
logging.basicConfig(filename='summarize.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def count_pixels(args):
  start_layer, end_layer, zarr_array, shared_dict, lock, progress_bar = args
  local_dict = {}
  try:
    for layer_idx in range(start_layer, end_layer):
      logging.info(f"Processing layer {layer_idx} in range {start_layer}-{end_layer}")
      layer = zarr_array[layer_idx, :, :]
      unique, counts = np.unique(layer, return_counts=True)
      for u, c in zip(unique, counts):
        local_dict[u] = local_dict.get(u, 0) + c
      progress_bar.value += 1
  except Exception as e:
    logging.error("Exception occurred", exc_info=True)
  
  with lock:
    for k, v in local_dict.items():
      shared_dict[k] = shared_dict.get(k, 0) + v

# Open Zarr array with thread synchronizer
synchronizer = zarr.ThreadSynchronizer()
zarr_array = zarr.open('3M-APP-SCN_remapped.zarr', mode='r', synchronizer=synchronizer)['segmentation_0.1']
logging.info("Opened segmentation zarr array")

# Number of cores
num_cores = min(cpu_count(), 15)
logging.info(f"Number of cores: {num_cores}")

# Calculate layers per core
total_layers = zarr_array.shape[0]
layers_per_core = total_layers // num_cores
remainder = total_layers % num_cores

with Manager() as manager:
  shared_dict = manager.dict()
  lock = manager.Lock()
  progress_bar = manager.Value('i', 0)  # shared counter

  args = []
  start_layer = 0
  for i in range(num_cores):
    end_layer = start_layer + layers_per_core
    if i < remainder:
      end_layer += 1
    args.append((start_layer, end_layer, zarr_array, shared_dict, lock, progress_bar))
    start_layer = end_layer

  # Create ThreadPool and run
  try:
    with Pool(num_cores) as pool:
      async_result = pool.map_async(count_pixels, args)
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

  # Save the dictionary to a pickle file
  with open('pixel_counts.pkl', 'wb') as f:
    pickle.dump(dict(shared_dict), f)

  # Find the max object ID value
  max_id = max(shared_dict.keys())
  logging.info(f"Max object ID: {max_id}")

logging.info('Completed summarization of 3M-APP-SCN_remapped.zarr')