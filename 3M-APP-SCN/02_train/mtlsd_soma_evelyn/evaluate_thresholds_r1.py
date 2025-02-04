import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import daisy
import json
import logging
import numpy as np
import numba as nb
import time
import os
import sys
from funlib.evaluate import rand_voi
from funlib.segment.arrays import replace_values
from multiprocessing import Pool, shared_memory
from funlib.persistence.arrays import open_ds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@nb.njit(nogil=True)
def replace_where_not(arr, needle, replace):
    """Numba-optimized array value replacement."""
    for idx in nb.prange(arr.size):
        if arr.flat[idx] not in needle:
            arr.flat[idx] = replace

def evaluate_thresholds(
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        crops,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        num_workers):
    
    results = {}
    results_file = os.path.join(fragments_file, "results.json")

    for crop in crops:
        start_time = time.time()
        crop_name, crop_roi = parse_crop(crop)
        
        logger.info(f"Processing crop: {crop_name}")
        fragments, gt = load_datasets(
            fragments_file, fragments_dataset,
            gt_file, gt_dataset, crop_roi
        )

        thresholds = generate_thresholds(thresholds_minmax, thresholds_step)
        logger.info(f"Evaluating {len(thresholds)} thresholds with {num_workers} workers")

        # Convert arrays to shared memory
        with SharedMemoryManager() as smm:
            shm_fragments = smm.SharedMemory(size=fragments.nbytes)
            np_fragments = np.ndarray(
                fragments.shape, dtype=fragments.dtype, buffer=shm_fragments.buf)
            np_fragments[:] = fragments[:]

            shm_gt = smm.SharedMemory(size=gt.nbytes)
            np_gt = np.ndarray(gt.shape, dtype=gt.dtype, buffer=shm_gt.buf)
            np_gt[:] = gt[:]

            # Pass shared memory names to workers
            args = [
                (shm_fragments.name, fragments.shape, fragments.dtype.str,
                 shm_gt.name, gt.shape, gt.dtype.str,
                 threshold, fragments_file, edges_collection, crop_name)
                for threshold in thresholds
            ]

            with Pool(num_workers) as pool:
                metrics_list = pool.starmap(process_threshold, args)

        # Aggregate results
        metrics_dict = {m['threshold']: m for m in metrics_list}
        best_voi = min(metrics_dict.values(), key=lambda x: x['voi_sum'])
        results[crop_name] = {
            'metrics': metrics_dict,
            'best_voi': best_voi
        }

        logger.info(f"Crop {crop_name} completed in {time.time()-start_time:.2f}s")
        logger.info(f"Best VOI: {best_voi['voi_sum']} @ threshold {best_voi['threshold']}")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

def parse_crop(crop):
    """Parse crop configuration."""
    assert crop is not None
    
    with open(os.path.join(crop['file']), "r") as f:
        crop_config = json.load(f)
    return crop_config['name'], daisy.Roi(crop_config['offset'], crop_config['shape'])

def load_datasets(fragments_file, fragments_ds, gt_file, gt_ds, roi):
    """Load and align datasets."""
    fragments = open_ds(fragments_file, fragments_ds)
    gt = open_ds(gt_file, gt_ds)
    
    if roi:
        common_roi = roi.intersect(fragments.roi.intersect(gt.roi))
    else:
        common_roi = fragments.roi.intersect(gt.roi)
    
    logger.info(f"Processing ROI: {common_roi}")
    return (
        fragments[common_roi].to_ndarray().astype(np.uint64),
        gt[common_roi].to_ndarray().astype(np.uint64)
    )

def generate_thresholds(minmax, step):
    return np.round(np.arange(minmax[0], minmax[1], step), decimals=2).tolist()

def process_threshold(shm_frag_name, frag_shape, frag_dtype,
                      shm_gt_name, gt_shape, gt_dtype,
                      threshold, fragments_file, edges_collection, crop_name):
    """Worker process for threshold evaluation."""
    try:
        # Attach to shared memory
        existing_shm_frag = shared_memory.SharedMemory(name=shm_frag_name)
        fragments = np.ndarray(frag_shape, dtype=frag_dtype, buffer=existing_shm_frag.buf)
        
        existing_shm_gt = shared_memory.SharedMemory(name=shm_gt_name)
        gt = np.ndarray(gt_shape, dtype=gt_dtype, buffer=existing_shm_gt.buf)

        # Get segmentation
        segment_ids = get_segmentation(fragments.copy(), fragments_file, edges_collection, threshold)
        
        # Calculate metrics
        return evaluate_segmentation(segment_ids, gt.copy(), threshold, crop_name)
    finally:
        existing_shm_frag.close()
        existing_shm_gt.close()

def get_segmentation(fragments, fragments_file, edges_collection, threshold):
    """Load LUT and relabel fragments."""
    lut_path = os.path.join(
        fragments_file, 'luts', 'fragment_segment',
        f'seg_{edges_collection}_{int(threshold*100)}.npz'
    )
    
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"LUT not found: {lut_path}")
    
    lut = np.load(lut_path)['fragment_segment_lut']
    return replace_values(fragments, lut[0], lut[1])

def evaluate_segmentation(segment_ids, gt, threshold, crop_name):
    """Calculate evaluation metrics."""
    if not crop_name.startswith('crop'):
        valid_ids = np.unique(segment_ids[gt != 0])
        replace_where_not(segment_ids, valid_ids, 0)

    metrics = rand_voi(gt, segment_ids, return_cluster_scores=False)
    metrics['voi_sum'] = metrics['voi_split'] + metrics['voi_merge']
    metrics['threshold'] = float(threshold)
    
    return metrics

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    evaluate_thresholds(**config)