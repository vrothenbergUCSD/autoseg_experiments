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
import concurrent.futures
from multiprocessing import shared_memory, Manager
from funlib.evaluate import rand_voi, detection_scores
from funlib.segment.arrays import replace_values
from funlib.persistence.arrays import open_ds, prepare_ds

# Initialize logging
logging.basicConfig(filename='evaluate_thresholds.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Redirect stdout and stderr to the log file
log_file = open('evaluate_thresholds.log', 'a')
sys.stdout = log_file
sys.stderr = log_file

""" Script to evaluate VOI, NVI, NID against ground truth for a fragments dataset at different 
agglomeration thresholds and find the best threshold. """


@nb.njit
def replace_where_not(arr, needle, replace):
    arr = arr.ravel()
    needles = set(needle)
    for idx in range(arr.size):
        if arr[idx] not in needles:
            arr[idx] = replace


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

    results_file = os.path.join(fragments_file, "results.json") 
    results = {}

    for crop in crops:

        start = time.time()

        if crop:  # crop must be absolute path
            crop_name = crop["name"]
            crop_roi = daisy.Roi(crop["offset"], crop["shape"])
            # fragments_file = os.path.join(fragments_file, crop_name + '.zarr')

        else:
            crop_name = "wtf"
            crop_roi = None

        logging.info("Crop %s", crop_name)
        logging.info(crop_roi)

        # open fragments
        logging.info("Reading fragments from %s", fragments_file)

        fragments = ds_wrapper(fragments_file, fragments_dataset)

        logging.info("Reading gt dataset %s from %s", gt_dataset, gt_file)

        gt = ds_wrapper(gt_file, gt_dataset)

        logging.info("fragments ROI is %s", fragments.roi)
        logging.info("gt roi is %s", gt.roi)

        vs = gt.voxel_size

        if crop_roi:
            common_roi = crop_roi
        else:
            common_roi = fragments.roi.intersect(gt.roi)

        logging.info("common roi is %s", common_roi)
        # evaluate only where we have both fragments and GT
        logging.info("Cropping fragments, mask, and GT to common ROI %s", common_roi)
        fragments = fragments[common_roi]
        gt = gt[common_roi]

        logging.info("Converting fragments to nd array...")
        fragments = fragments.to_ndarray()

        logging.info("Converting gt to nd array...")
        gt = gt.to_ndarray()

        # # Create shared memory blocks for fragments and gt
        # shm_fragments = shared_memory.SharedMemory(create=True, size=fragments.nbytes)
        # shm_gt = shared_memory.SharedMemory(create=True, size=gt.nbytes)

        # shared_fragments = np.ndarray(fragments.shape, dtype=fragments.dtype, buffer=shm_fragments.buf)
        # shared_gt = np.ndarray(gt.shape, dtype=gt.dtype, buffer=shm_gt.buf)

        # np.copyto(shared_fragments, fragments)
        # np.copyto(shared_gt, gt)

        thresholds = list(np.round(np.arange(
            thresholds_minmax[0],
            thresholds_minmax[1],
            thresholds_step), 4))

        logging.info("Evaluating thresholds...")

        # parallel process
        manager = Manager()
        metrics = manager.dict()

        for threshold in thresholds:
            metrics[threshold] = manager.dict()

        for t in thresholds:
            evaluate(t, fragments, gt, fragments_file, crop_name, edges_collection, metrics)
            


        # try:
        #     run_evaluation(thresholds, lambda t: (t, shm_fragments.name, fragments.shape, fragments.dtype, shm_gt.name, gt.shape, gt.dtype, fragments_file, crop_name, edges_collection, metrics), num_workers)
        # finally:
        #     # Clean up shared memory
        #     shm_fragments.close()
        #     shm_fragments.unlink()
        #     shm_gt.close()
        #     shm_gt.unlink()

        voi_sums = {}
        for t in thresholds:
            if 'voi_sum' in metrics[t]:
                voi_sums[metrics[t]['voi_sum']] = t

        if voi_sums:
            voi_thresh = voi_sums[sorted(voi_sums.keys())[0]]

            metrics = dict(metrics)
            metrics['best_voi'] = metrics[voi_thresh]

            results[crop_name] = metrics

            logging.info("best VOI for %s is at threshold= %s , VOI= %s, VOI_split= %s , VOI_merge= %s", crop_name, voi_thresh, metrics[voi_thresh]['voi_sum'], metrics[voi_thresh]['voi_split'], metrics[voi_thresh]['voi_merge'])
        else:
            logging.warning("No valid VOI results found for crop %s", crop_name)

        logging.info("Time to evaluate thresholds = %s", time.time() - start)

        fragments_file = os.path.dirname(fragments_file)  # reset

    # finally, dump all results to json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


def ds_wrapper(in_file, in_ds):
    try:
        ds = open_ds(in_file, in_ds)
    except:
        ds = open_ds(in_file, in_ds + '/s0')

    return ds

def evaluate(
        threshold,
        fragments,
        gt,
        fragments_file,
        crop_name,
        edges_collection,
        metrics):
    
    segment_ids = get_segmentation(
            fragments,
            fragments_file,
            edges_collection,
            threshold)

    results = evaluate_threshold(
            edges_collection,
            crop_name,
            segment_ids,
            gt,
            threshold)

    metrics[threshold] = results


def evaluate_multi(
        threshold,
        shm_fragments_name,
        fragments_shape,
        fragments_dtype,
        shm_gt_name,
        gt_shape,
        gt_dtype,
        fragments_file,
        crop_name,
        edges_collection,
        metrics):

    logging.info("Evaluating threshold %s", threshold)

    # Access shared memory
    existing_shm_fragments = shared_memory.SharedMemory(name=shm_fragments_name)
    existing_shm_gt = shared_memory.SharedMemory(name=shm_gt_name)

    fragments = np.ndarray(fragments_shape, dtype=fragments_dtype, buffer=existing_shm_fragments.buf)
    gt = np.ndarray(gt_shape, dtype=gt_dtype, buffer=existing_shm_gt.buf)

    try:
        segment_ids = get_segmentation(fragments, fragments_file, edges_collection, threshold)
        results = evaluate_threshold(edges_collection, crop_name, segment_ids, gt, threshold)
        metrics[threshold] = results
        logging.info("Successfully evaluated threshold %s", threshold)
    except Exception as exc:
        logging.error(f'Error evaluating threshold {threshold}: {exc}')
        raise
    finally:
        existing_shm_fragments.close()
        existing_shm_gt.close()


def get_segmentation(fragments, fragments_file, edges_collection, threshold):

    logging.info("Getting segmentation for threshold %s", threshold)

    # Construct the filename for the LUT file
    lut_dir = os.path.join(fragments_file, 'luts', 'fragment_segment')
    lut_filename = f'seg_{edges_collection}_{int(threshold * 10000)}.npz'
    lut_path = os.path.join(lut_dir, lut_filename)

    # Load the LUT file
    if not os.path.exists(lut_path):
        logging.error("No LUT file found for threshold %s", threshold)
        raise ValueError(f"No LUT file found for threshold {threshold}")

    lut = np.load(lut_path)['fragment_segment_lut']

    fragment_ids = lut[0]
    segment_ids = lut[1]

    # Use replace_values to create the segmented array
    segmented_array = replace_values(fragments, fragment_ids, segment_ids)

    logging.info("Segmentation for threshold %s obtained", threshold)

    return segmented_array


def evaluate_threshold(
        edges_collection,
        crop_name,
        test,
        truth,
        threshold):

    logging.info("Evaluating VOI and RAND for threshold %s", threshold)

    gt = truth.copy().astype(np.uint64)
    segment_ids = test.copy().astype(np.uint64)

    assert gt.shape == segment_ids.shape

    # if crop is a training crop (i.e a "run")
    if not crop_name.startswith('crop'):
        name = 'run' + crop_name[-2:]

        # return IDs of segment_ids where gt > 0
        chosen_ids = np.unique(segment_ids[gt != 0])

        # mask out all but remaining ids in segment_ids
        replace_where_not(segment_ids, chosen_ids, 0)

    else:
        name = crop_name

    # get VOI and RAND
    rand_voi_report = rand_voi(gt, segment_ids, return_cluster_scores=False)

    metrics = rand_voi_report.copy()
    out = {}

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]

    metrics['voi_sum'] = metrics['voi_split'] + metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split'] + metrics['nvi_merge']
    metrics['merge_function'] = edges_collection.strip('edges_')
    metrics['threshold'] = threshold

    logging.info("VOI and RAND for threshold %s evaluated", threshold)

    return metrics


def evaluate_with_handling(*args):
    try:
        evaluate(*args)
    except Exception as exc:
        logging.error(f'An exception occurred during processing: {exc}')
        raise


def run_evaluation(thresholds, evaluate_args, num_workers):
    logging.info("Running evaluation with %d workers", num_workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_with_handling, *evaluate_args(t)): t for t in thresholds}
        try:
            for future in concurrent.futures.as_completed(futures):
                t = futures[future]
                try:
                    future.result()
                    logging.info("Successfully processed threshold %s", t)
                except Exception as exc:
                    logging.error(f'Error evaluating threshold {t}: {exc}')
                    raise
        
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt detected, terminating all processes...")
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
            executor.shutdown(wait=False, cancel_futures=True)
            raise


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_thresholds(**config)
