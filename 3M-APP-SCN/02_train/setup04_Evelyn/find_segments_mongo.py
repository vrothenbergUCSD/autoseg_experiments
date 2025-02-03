import daisy
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
import time

from funlib.segment.graphs.impl import connected_components
# from funlib.persistence.graphs import FileGraphProvider
from funlib.persistence.graphs import MongoDbGraphProvider
from funlib.persistence.arrays import open_ds, prepare_ds

logging.getLogger().setLevel(logging.INFO)
# Initialize logging
logging.basicConfig(filename='find_segments.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and log them."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def find_segments(
        fragments_file,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        block_size,
        num_workers,
        crops,
        fragments_dataset=None,
        run_type=None,
        roi_offset=None,
        roi_shape=None,
        **kwargs):

    '''
    Args:
        fragments_file (``string``):
            Path to file (zarr/n5) containing fragments (supervoxels).
        edges_collection (``string``):
            The name of the MongoDB database edges collection to use.
        thresholds_minmax (``list`` of ``int``):
            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.
        thresholds_step (``float``):
            The step size to use when generating thresholds between min/max.
        block_size (``tuple`` of ``int``):
            The size of one block in world units (must be multiple of voxel
            size).
        num_workers (``int``):
            How many workers to use when reading the region adjacency graph
            blockwise.
        fragments_dataset (``string``, optional):
            Name of fragments dataset. Include if using full fragments roi, set
            to None if using a crop (roi_offset + roi_shape).
        run_type (``string``, optional):
            Can be used to direct luts into directory (e.g testing, validation,
            etc).
        roi_offset (array-like of ``int``, optional):
            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.
        roi_shape (array-like of ``int``, optional):
            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.
    '''

    crop = ""

    logging.info("Reading graph")
    start = time.time()
    
    if crop != "":
        fragments_file = os.path.join(fragments_file,os.path.basename(crop)[:-4]+'zarr')
        crop_path = os.path.join(fragments_file,'crop.json')
        with open(crop_path,"r") as f:
            crop = json.load(f)
        
        crop_name = crop["name"]
        crop_roi = daisy.Roi(crop["offset"],crop["shape"])

    else:
        crop_name = ""
        crop_roi = None

    logging.info(f"FRAGS FILE {fragments_file}")
    # block_directory = os.path.join(fragments_file,'block_nodes')

    fragments = open_ds(fragments_file,fragments_dataset)
    logging.info("Read fragments file.")

    if block_size == [0,0,0]: #if processing one block    
        context = [50,40,40]
        block_size = crop_roi.shape if crop_roi else fragments.roi.shape
    
    roi = fragments.roi
    block_size = daisy.Coordinate(block_size)

    # Old implementation
    # graph_provider = FileGraphProvider(
    #     directory=block_directory,
    #     chunk_size=block_size,
    #     edges_collection=edges_collection,
    #     position_attribute=[
    #         'center_z',
    #         'center_y',
    #         'center_x'])

    graph_provider = MongoDbGraphProvider(
        db_name="mongo_lsd",
        host="localhost",  # Optional, if not localhost
        mode="r+",  # Read and write mode
        directed=False,  # Graph is not directed
        # nodes_collection="nodes",  # Optional, if different from default
        edges_collection=edges_collection,
        # meta_collection="meta",  # Optional, if different from default
        position_attribute=['center_z', 'center_y', 'center_x']  # Position attributes
    )

    def convert_list_of_dicts(list_of_dicts):
        keys = list_of_dicts[0].keys()
        converted_dict = {key: [] for key in keys}
        for item in list_of_dicts:
            for key in keys:
                converted_dict[key].append(item[key])
        # Convert lists to numpy arrays for consistency with previous code
        for key in keys:
            converted_dict[key] = np.array(converted_dict[key])
        return converted_dict


    # node_attrs,edge_attrs = graph_provider.read_blockwise(roi,block_size/2,num_workers)
    node_attrs = graph_provider.read_nodes(roi)
    logging.info("Graph provider read nodes.")
    edge_attrs = graph_provider.read_edges(roi, nodes=node_attrs)
    logging.info("Graph provider read edges.")

    logging.info(f"Read graph in {time.time() - start}")

    # Old FileGraphProvider code 
    # if 'id' not in node_attrs:
    #     logging.info('No nodes found in roi %s' % roi)
    #     return
    # nodes = node_attrs['id']

    nodes = np.array([node['id'] for node in node_attrs], dtype=np.uint64)

    if len(nodes) == 0:
        logging.info('No nodes found in roi %s' % roi)
        return

    # Old FileGraphProvider code 
    # edges = np.stack(
    #             [
    #                 edge_attrs['u'].astype(np.uint64),
    #                 edge_attrs['v'].astype(np.uint64)
    #             ],
    #         axis=1)

    # Extract 'u' and 'v' values from each dictionary in edge_attrs
    u_values = [edge['u'] for edge in edge_attrs]
    v_values = [edge['v'] for edge in edge_attrs]

    # Convert to numpy arrays
    u_array = np.array(u_values, dtype=np.uint64)
    v_array = np.array(v_values, dtype=np.uint64)

    # Stack the arrays
    edges = np.stack([u_array, v_array], axis=1)

    # Extract 'merge_score' values from each dictionary in edge_attrs
    merge_score_values = [edge['merge_score'] for edge in edge_attrs]

    # scores = merge_score_values.astype(np.float32)
    # Convert to numpy array
    scores = np.array(merge_score_values, dtype=np.float32)

    logging.info(f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    out_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment')

    if run_type:
        out_dir = os.path.join(out_dir, run_type)

    os.makedirs(out_dir, exist_ok=True)

    # thresholds = [round(i,2) for i in np.arange(
    #     float(thresholds_minmax[0]),
    #     float(thresholds_minmax[1]),
    #     thresholds_step)]
    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))
    thresholds = np.round(thresholds, 4)
    
    logging.info(f"Thresholds: {thresholds}")

    #parallel processing
    
    start = time.time()

    try:

        with mp.Pool(num_workers) as pool:

            pool.starmap(get_connected_components,[(nodes,edges,scores,t,edges_collection,out_dir) for t in thresholds])

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

#    pool = []
#
#    for t in thresholds:
#
#        p = mp.Process(target=get_connected_components, args=(nodes,edges,scores,t,edges_collection,out_dir,))
#        pool.append(p)
#        p.start()
#
#    for p in pool: p.join()

#    for t in thresholds:
#
#        get_connected_components(
#                nodes,
#                edges,
#                scores,
#                t,
#                edges_collection,
#                out_dir)

    logging.info(f"Created and stored lookup tables in {time.time() - start}")

    #reset
    block_size = [0,0,0]
    fragments_file = os.path.dirname(fragments_file)

def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        edges_collection,
        out_dir,
        **kwargs):

    logging.info(f"Getting CCs for threshold {threshold}...")
    components = connected_components(nodes, edges, scores, threshold)

    logging.info(f"Creating fragment-segment LUT for threshold {threshold}...")
    lut = np.array([nodes, components])

    logging.info(f"Storing fragment-segment LUT for threshold {threshold}...")

    lookup = f"seg_{edges_collection}_{int(threshold*10000)}"

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    find_segments(**config)

    logging.info("Complete.")

    logging.info(f'Took {time.time() - start} seconds to find segments and store LUTs')
