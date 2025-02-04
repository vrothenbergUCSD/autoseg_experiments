import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import daisy
import json
import numpy as np
import numba as nb
import time
import os
import sys

from multiprocessing import Pool
from funlib.evaluate import rand_voi  # detection_scores
from funlib.segment.arrays import replace_values
from funlib.persistence.arrays import open_ds

# -----------------------------------------------------------------------------
# Optional: keep numba if you want parallel speed. Otherwise, can use numpy.isin
@nb.njit(parallel=False, fastmath=True)
def replace_where_not(arr, keep_ids, replace_with=0):
    """In-place set arr[...] = replace_with if not in keep_ids."""
    keep_ids_set = set(keep_ids)
    for i in range(arr.size):
        if arr[i] not in keep_ids_set:
            arr[i] = replace_with
# -----------------------------------------------------------------------------

def ds_wrapper(in_file, in_ds):
    """
    Safely open a dataset from `in_file` at dataset `in_ds`.
    If `in_ds` fails, try `in_ds + "/s0"`.
    """
    try:
        ds = open_ds(in_file, in_ds)
    except Exception:
        ds = open_ds(in_file, in_ds + '/s0')
    return ds

def get_segmentation(fragments, fragments_file, edges_collection, threshold):
    """
    Given fragments, load the LUT at the appropriate threshold and relabel
    the fragments to produce a segmentation.
    """
    # Carefully round threshold to int
    lut_threshold = int(round(threshold * 100))

    fragment_segment_lut_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment'
    )
    fragment_segment_lut_file = os.path.join(
        fragment_segment_lut_dir,
        f'seg_{edges_collection}_{lut_threshold}.npz'
    )

    if not os.path.exists(fragment_segment_lut_file):
        raise FileNotFoundError(
            f"Could not find LUT file: {fragment_segment_lut_file}"
        )

    fragment_segment_lut = np.load(fragment_segment_lut_file)['fragment_segment_lut']
    # This LUT is assumed to have shape (2, N), first row = fragment_ids, second = segment_ids
    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])
    return segment_ids

def evaluate_threshold(edges_collection, crop_name, seg, gt, threshold):
    """
    Compute VOI and Rand given a segmentation and ground truth.
    Optionally mask out anything outside ground truth if this is not a 'crop' ROI.
    """
    seg = seg.copy().astype(np.uint64)
    gt = gt.copy().astype(np.uint64)

    assert seg.shape == gt.shape, "Segmentation and GT shapes do not match!"

    # If crop is a 'run' or training area (just an example condition)
    if not crop_name.startswith('crop'):
        # Keep only segmentation IDs where GT != 0
        keep_ids = np.unique(seg[gt != 0])
        # in-place zero out everything else
        replace_where_not(seg, keep_ids, 0)

    # Rand/VOI from funlib.evaluate
    rand_voi_report = rand_voi(gt, seg, return_cluster_scores=False)
    # Potential detection scores - commented out
    # detection = detection_scores(gt, seg, voxel_size=[50,2,2])

    # Remove extra keys if present
    for k in ('voi_split_i', 'voi_merge_j'):
        if k in rand_voi_report:
            del rand_voi_report[k]

    metrics = rand_voi_report.copy()
    metrics['voi_sum'] = metrics['voi_split'] + metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split'] + metrics['nvi_merge']
    metrics['merge_function'] = edges_collection.replace('edges_', '')
    metrics['threshold'] = float(threshold)

    return metrics

def _evaluate_single_threshold(args):
    """
    Helper function to unpack arguments for parallel processing.
    """
    (threshold, fragments, gt, fragments_file, crop_name, edges_collection) = args
    seg = get_segmentation(fragments, fragments_file, edges_collection, threshold)
    return threshold, evaluate_threshold(edges_collection, crop_name, seg, gt, threshold)

def evaluate_thresholds(
    gt_file,
    gt_dataset,
    fragments_file,
    fragments_dataset,
    crops,
    edges_collection,
    thresholds_minmax,
    thresholds_step,
    num_workers=1
):
    """
    Evaluate a range of thresholds on a (fragments) dataset vs. ground truth.
    For each crop, compute VOI metrics across thresholds and pick the best.
    """
    results_file = os.path.join(fragments_file, "results.json")
    results = {}

    for crop in crops:
        start_time = time.time()

        crop_name = crop.get("name", "unknown_crop")
        offset = crop.get("offset", [0,0,0])
        shape = crop.get("shape", [0,0,0])
        crop_roi = daisy.Roi(offset, shape) if shape and offset else None

        print(f"\n=== Processing crop '{crop_name}' ===")
        print(f"Reading fragments from: {fragments_file}")
        fragments_ds = ds_wrapper(fragments_file, fragments_dataset)

        print(f"Reading GT from: {gt_file}")
        gt_ds = ds_wrapper(gt_file, gt_dataset)

        common_roi = fragments_ds.roi.intersect(gt_ds.roi)
        if crop_roi is not None:
            common_roi = common_roi.intersect(crop_roi)

        print(f"Common ROI: {common_roi}")
        fragments_ds = fragments_ds[common_roi]
        gt_ds = gt_ds[common_roi]

        # Convert once to numpy arrays (expensive, do it outside parallel loop!)
        print("Converting fragments to ndarray...")
        fragments_arr = fragments_ds.to_ndarray()

        print("Converting GT to ndarray...")
        gt_arr = gt_ds.to_ndarray()

        # Build list of thresholds
        t0, t1 = thresholds_minmax
        thresholds = np.arange(t0, t1, thresholds_step)
        thresholds = thresholds.tolist()  # for easier iteration

        print(f"Evaluating thresholds in [{t0}, {t1}) with step {thresholds_step} ...")

        # Prepare args for parallel pool
        pool_args = [
            (thr, fragments_arr, gt_arr, fragments_file, crop_name, edges_collection)
            for thr in thresholds
        ]

        # Evaluate in parallel
        with Pool(num_workers) as pool:
            parallel_results = pool.map(_evaluate_single_threshold, pool_args)

        # Convert list of (threshold, metrics_dict) into a dict
        threshold_metrics = {}
        for thr, m in parallel_results:
            threshold_metrics[thr] = m

        # Find best threshold by minimal VOI
        best_thr = None
        best_voi = float('inf')
        for thr, m in threshold_metrics.items():
            if m['voi_sum'] < best_voi:
                best_voi = m['voi_sum']
                best_thr = thr

        # You can do likewise for nvi_sum, etc., if needed

        # Attach results to the final dictionary
        threshold_metrics['best_voi'] = threshold_metrics[best_thr]
        results[crop_name] = threshold_metrics

        elapsed = time.time() - start_time
        print(f"Crop '{crop_name}': best VOI at threshold={best_thr}, "
              f"voi_sum={threshold_metrics[best_thr]['voi_sum']}")
        print(f"Time to evaluate = {elapsed:.2f}s")

    # Finally, dump results to a JSON
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll results saved to: {results_file}")

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_thresholds(**config)
