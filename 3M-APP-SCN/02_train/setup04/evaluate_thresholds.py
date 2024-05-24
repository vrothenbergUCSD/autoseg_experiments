import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import daisy
import json
import logging
import numpy as np
import numba as nb
import time
import os
import sys
from funlib.evaluate import rand_voi,detection_scores
from funlib.segment.arrays import replace_values
from funlib.persistence.arrays import open_ds, prepare_ds

from multiprocessing import Process,Manager,Pool
from multiprocessing.managers import SharedMemoryManager

# from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

""" Script to evaluate VOI,NVI,NID against ground truth for a fragments dataset at different 
agglomeration thresholds and find the best threshold. """

log_directory = 'log'
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, 'evaluation.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        roi_offset,
        roi_shape,
        num_workers):

    results_file = os.path.join(fragments_file,"results.json") 
    results = {}
    crop = ""

    start = time.time()
    
    if crop != "": #crop must be absolute path
        with open(os.path.join(gt_file,crop),"r") as f:
            crop = json.load(f)
        
        crop_name = crop["name"]
        crop_roi = daisy.Roi(crop["offset"],crop["shape"])

        fragments_file = os.path.join(fragments_file,crop_name+'.zarr')

    else:
        crop_name = "evaluate_crop"
        crop_roi = daisy.Roi(roi_offset, roi_shape)
        # "roi_offset": [2500,5000,5000]
        # "roi_shape": [5000,10000,10000]

    # open fragments
    logging.info(f"Reading fragments from {fragments_file}")

    fragments = ds_wrapper(fragments_file, fragments_dataset)

    logging.info("Reading gt from %s" %gt_file)

    gt = ds_wrapper(gt_file, gt_dataset)

    logging.info("fragments ROI is {}".format(fragments.roi))
    logging.info("gt roi is {}".format(gt.roi))

    vs = gt.voxel_size

    if crop_roi:
        common_roi = crop_roi
        logging.info(f"common_roi = crop_roi")

    else:
        common_roi = fragments.roi.intersect(gt.roi)
        logging.info(f"common_roi = fragments.roi.intersect(gt.roi)")

    logging.info("common roi is {}".format(common_roi))
    # evaluate only where we have both fragments and GT
    logging.info("Cropping fragments, mask, and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    logging.info("Converting fragments to nd array...")
    fragments = fragments.to_ndarray()

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))
    thresholds = np.round(thresholds, 4)

    logging.info("Evaluating thresholds...")
    
    # parallel process
    manager = Manager()
    metrics = manager.dict()

    for threshold in thresholds:
        metrics[threshold] = manager.dict()

    logging.info(f"Starting pool of {num_workers} workers")

    # with Pool(num_workers) as pool:
    #     pool.starmap(evaluate,[(t,fragments,gt,fragments_file,crop_name,edges_collection,metrics) for t in thresholds])
    def evaluate_args(t):
        return (t, fragments, gt, fragments_file, crop_name, edges_collection, metrics)

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(evaluate, *evaluate_args(t)): t for t in thresholds}
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                future.result()
                logging.info(f"Successfully processed threshold {t}")
            except Exception as exc:
                logging.error(f'An exception occurred during processing threshold {t}: {exc}')

    logging.info('Completed pool.starmap')
    voi_sums = {metrics[x]['voi_sum']:x for x in thresholds}
    #nvi_sums = {metrics[x]['nvi_sum']:x for x in thresholds}
    #nids = {metrics[x]['nid']:x for x in thresholds}

    voi_thresh = voi_sums[sorted(voi_sums.keys())[0]]
    #nvi_thresh = nvi_sums[sorted(nvi_sums.keys())[0]]
    #nid_thresh = nids[sorted(nids.keys())[0]]

    metrics = dict(metrics)
    metrics['best_voi'] = metrics[voi_thresh]

    results[crop_name] = metrics

    logging.info(f"best VOI for {crop_name} is at threshold= {voi_thresh} , VOI= {metrics[voi_thresh]['voi_sum']}, VOI_split= {metrics[voi_thresh]['voi_split']} , VOI_merge= {metrics[voi_thresh]['voi_merge']}")
    logging.info(f"Time to evaluate thresholds = {time.time() - start}")

    fragments_file = os.path.dirname(fragments_file) #reset

    #finally, dump all results to json
    with open(results_file,"w") as f:
        json.dump(results,f,indent=4)


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
    
    logging.info(f'Starting evaluation for threshold: {threshold}')
    segment_ids = get_segmentation(
            fragments,
            fragments_file,
            edges_collection,
            threshold)
    logging.info(f'Segmentation done for threshold: {threshold}')

    results = evaluate_threshold(
            edges_collection,
            crop_name,
            segment_ids,
            gt,
            threshold)
    logging.info(f'Evaluation done for threshold: {threshold}')

    metrics[threshold] = results


def get_segmentation(
        fragments,
        fragments_file,
        edges_collection,
        threshold):

    #logging.info("Loading fragment - segment lookup table for threshold %s..." %threshold)
    fragment_segment_lut_dir = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment')

    fragment_segment_lut_file = os.path.join(
            fragment_segment_lut_dir,
            'seg_%s_%d.npz' % (edges_collection, int(threshold*10000)))

    fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    #logging.info("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids


def evaluate_threshold(
        edges_collection,
        crop_name,
        test,
        truth,
        threshold):

        gt = truth.copy().astype(np.uint64)
        segment_ids = test.copy().astype(np.uint64)

        assert gt.shape == segment_ids.shape

        #if crop is a training crop (i.e a "run")
        if not crop_name.startswith('crop'):
            name = 'run'+crop_name[-2:]

            #return IDs of segment_ids where gt > 0
            chosen_ids = np.unique(segment_ids[gt != 0])

            #mask out all but remaining ids in segment_ids
            replace_where_not(segment_ids,chosen_ids,0)

        else: name = crop_name

        #get VOI and RAND
        rand_voi_report = rand_voi(
                gt,
                segment_ids,
                return_cluster_scores=False)

        #scores = detection_scores(
        #        gt,
        #        segment_ids,
        #        voxel_size=[50,2,2]) #lazy
        
        metrics = rand_voi_report.copy()
        out = {}
        

        for k in {'voi_split_i', 'voi_merge_j'}:
            del metrics[k]

        metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
        metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']
        metrics['merge_function'] = edges_collection.strip('edges_')

        #metrics["com_distance"] = float(scores["avg_distance"])
        #metrics["iou"] = float(scores["avg_iou"])
        #metrics["tp"] = scores["tp"]
        #metrics["fp"] = scores["fp"]
        #metrics["fn"] = scores["fn"]
        #metrics["precision"] = scores["tp"] / (scores["fp"] + scores["tp"])
        #metrics["recall"] = scores["tp"] / (scores["fp"] + scores["fn"])

        #metrics = {name+'_'+k:v for k,v in metrics.items()}
        metrics['threshold'] = threshold

        return metrics

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_thresholds(**config)
