#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import time

import numcodecs
import numpy as np
import zarr

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

# Disable Blosc threading in multi-process
numcodecs.blosc.use_threads = False


###############################################################################
#                            Argument Parsing                                 #
###############################################################################

parser = argparse.ArgumentParser(description='Remap segmentation IDs in a Zarr dataset.')
parser.add_argument('--input_zarr', type=str, required=True,
                    help='Path to input Zarr file.')
parser.add_argument('--input_dataset', type=str, required=True,
                    help='Input dataset name within the Zarr file.')
parser.add_argument('--output_dataset', type=str, required=True,
                    help='Output dataset name within the same Zarr file.')
parser.add_argument('--num_cores', type=int, default=16,
                    help='Number of cores to use for processing.')
parser.add_argument('--pickle_file', type=str, default='combined_counts.pkl',
                    help='Pickle file name for combined counts (optional; not strictly needed here).')
parser.add_argument('--log_path', type=str, default='log',
                    help='Directory for log files')

args = parser.parse_args()

# Initialize logging
os.makedirs(args.log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(args.log_path, 'remap_ids.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


###############################################################################
#                            Utility Functions                                #
###############################################################################

def get_chunk_ranges(shape, chunk_sizes):
    """
    Calculate the (start indices) for each chunk in 'shape' dimension,
    given the chunk sizes for each dimension.
    Returns a list of tuples, each tuple is the start index along each dimension.
    """
    # For each dimension, we create a range(0, shape[d], chunk_sizes[d])
    # Then we create the Cartesian product of those starts across all dims.
    grids = [range(0, s, cs) for s, cs in zip(shape, chunk_sizes)]
    # Mesh them together and reshape
    mesh = np.array(np.meshgrid(*grids, indexing='ij'))
    # Each column of mesh.reshape(len(shape), -1).T is a start coordinate
    return mesh.reshape(len(shape), -1).T


def find_unique_ids_in_chunk(chunk_start, zarr_array):
    """
    Reads one chunk (or partial chunk) from zarr_array given the chunk_start,
    then returns a set of all unique IDs within that chunk.
    """
    # Build actual slices for each dimension
    slices = []
    for start_idx, csize, dim_size in zip(chunk_start, zarr_array.chunks, zarr_array.shape):
        end_idx = min(start_idx + csize, dim_size)
        slices.append(slice(start_idx, end_idx))
    slices = tuple(slices)

    chunk_data = zarr_array[slices]
    return set(np.unique(chunk_data))


def parallelize_unique_id_search(zarr_array, num_cores):
    """
    Finds all unique IDs across the entire zarr_array by iterating over each chunk in parallel.
    """
    logging.info("Starting parallel unique ID search (chunk-based).")

    chunk_starts = get_chunk_ranges(zarr_array.shape, zarr_array.chunks)
    all_unique = set()

    with Progress() as progress:
        task = progress.add_task("[green]Counting Unique IDs...", total=len(chunk_starts))

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            future_to_chunk = {
                executor.submit(find_unique_ids_in_chunk, cs, zarr_array): cs
                for cs in chunk_starts
            }

            for future in as_completed(future_to_chunk):
                chunk_unique = future.result()
                all_unique.update(chunk_unique)
                progress.advance(task)

    logging.info(f"Unique ID search complete. Found {len(all_unique)} unique IDs.")
    return all_unique


def vectorized_remap(layer, id_map):
    """
    Vectorized remapping of IDs in 'layer' using the dictionary 'id_map'.
    Unknown IDs (not in old->new dict) remain unchanged.
    """
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
    valid_mask = (idx >= 0) & (idx < len(sorted_old_ids))
    valid_flat_layer = flat_layer[valid_mask]
    valid_idx = idx[valid_mask]

    # Further ensure that the IDs match exactly (handle IDs not in old_ids)
    actual_matches = (sorted_old_ids[valid_idx] == valid_flat_layer)
    final_valid_idx = valid_idx[actual_matches]

    # Map to new IDs
    remapped_flat_layer = flat_layer.copy()
    remapped_flat_layer[valid_mask][actual_matches] = sorted_new_ids[final_valid_idx]

    # Reshape back to the original shape
    return remapped_flat_layer.reshape(layer.shape)


def remap_chunk(chunk_start, zarr_input, zarr_output, id_map):
    """
    Reads the chunk from zarr_input at chunk_start, remaps IDs, and writes to zarr_output.
    """
    t0 = time.time()

    # Build the slices for each dimension
    slices = []
    for start_idx, csize, dim_size in zip(chunk_start, zarr_input.chunks, zarr_input.shape):
        end_idx = min(start_idx + csize, dim_size)
        slices.append(slice(start_idx, end_idx))
    slices = tuple(slices)

    # Load data
    chunk_data = zarr_input[slices]
    # Remap
    remapped = vectorized_remap(chunk_data, id_map)
    # Write out
    zarr_output[slices] = remapped

    elapsed = time.time() - t0
    logging.info(f"Completed remapping for chunk {chunk_start}, time={elapsed:.2f}s")


def parallelize_remap(zarr_input, zarr_output, id_map, num_cores):
    """
    Reads the entire zarr_input chunk-by-chunk in parallel, remaps using id_map,
    and writes into zarr_output.
    """
    chunk_starts = get_chunk_ranges(zarr_input.shape, zarr_input.chunks)

    with Progress() as progress:
        task = progress.add_task("[green]Remapping IDs...", total=len(chunk_starts))

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            future_to_chunk = {
                executor.submit(remap_chunk, cs, zarr_input, zarr_output, id_map): cs
                for cs in chunk_starts
            }

            for future in as_completed(future_to_chunk):
                # If an exception is raised inside remap_chunk, it will bubble up here
                # You can catch/log if desired
                future.result()
                progress.advance(task)


###############################################################################
#                                Main                                         #
###############################################################################

if __name__ == '__main__':
    logging.info(f"Opening Zarr file: {args.input_zarr}")
    synchronizer = zarr.ThreadSynchronizer()

    zarr_object = zarr.open(args.input_zarr, mode='a', synchronizer=synchronizer)
    if args.input_dataset not in zarr_object:
        logging.error(f"Dataset {args.input_dataset} not found in Zarr file.")
        raise ValueError(f"Dataset {args.input_dataset} not found.")

    zarr_array = zarr_object[args.input_dataset]
    logging.info(f"Opened dataset '{args.input_dataset}' with shape {zarr_array.shape} and chunks {zarr_array.chunks}")

    # Create the output dataset within the same Zarr file if it does not exist
    # Delete it first if it already exists
    if args.output_dataset in zarr_object:
        logging.info(f"Deleting existing dataset '{args.output_dataset}' to overwrite.")
        del zarr_object[args.output_dataset]

    # Create a new dataset for the remapped data
    new_zarr_array = zarr_object.create_dataset(
        args.output_dataset,
        shape=zarr_array.shape,
        chunks=zarr_array.chunks,
        dtype=zarr_array.dtype,
        synchronizer=synchronizer
    )
    logging.info(f"Created new dataset '{args.output_dataset}' in the same Zarr store.")

    # Determine how many workers to use
    num_cores = args.num_cores
    logging.info(f"Number of cores to use: {num_cores}")

    # 1) Find all unique IDs in the input zarr
    unique_ids = parallelize_unique_id_search(zarr_array, num_cores)
    logging.info(f"Total unique IDs found: {len(unique_ids)}")
    print(f"[INFO] Total unique IDs found: {len(unique_ids)}")

    # 2) Build ID map from old -> new. You can adjust numbering logic if desired.
    #    For example, skip 0 if you want zero to remain zero, etc.
    #    Currently, zero remains zero in the dataset but won't appear in unique_ids.
    #    We do not typically remap background, so it is safe to skip or ignore it.
    id_map = {}
    # Let's skip background if 0 is in unique_ids. Start new IDs at 1, for instance.
    # But if you want a contiguous range from 0, you can just do enumerate(unique_ids).
    # We'll do contiguous from 1 in this example:
    next_id = 1
    for old_id in unique_ids:
        if old_id == 0:
            # Keep background as 0 if present, do not remap
            id_map[old_id] = 0
        else:
            id_map[old_id] = next_id
            next_id += 1

    # 3) Perform chunk-wise parallel remapping
    parallelize_remap(zarr_array, new_zarr_array, id_map, num_cores)
    logging.info("Completed parallel ID remapping.")
    print("[INFO] Completed parallel ID remapping.")

    exit(0)
