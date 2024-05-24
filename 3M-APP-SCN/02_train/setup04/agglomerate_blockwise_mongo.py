import json
import hashlib
import logging
import numpy as np
import os
import daisy
import sys
import time
import subprocess

from funlib.persistence.arrays import open_ds, prepare_ds
# from funlib.persistence.graphs import FileGraphProvider
from funlib.persistence.graphs import MongoDbGraphProvider

from lsd.post import agglomerate_in_block

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename='agglomerate_blockwise.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
def agglomerate(
        base_dir,
        experiment,
        setup,
        iteration,
        file_name,
        crops,
        affs_dataset,
        fragments_dataset,
        block_size,
        context,
        num_workers,
        merge_function,
        **kwargs):

    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.
    Args:
        file_name (``string``):
            The input file containing affs and fragments.
        affs_dataset, fragments_dataset (``string``):
            Where to find the affinities and fragments.
        block_size (``tuple`` of ``int``):
            The size of one block in world units.
        context (``tuple`` of ``int``):
            The context to consider for fragment extraction and agglomeration,
            in world units.
        num_workers (``int``):
            How many blocks to run in parallel.
        merge_function (``string``):
            Symbolic name of a merge function. See dictionary below.
    '''
        
   
    affs_file = os.path.abspath(
            os.path.join(
                base_dir,experiment,"02_train",setup,"prediction",file_name
                )
            )
    
    crop = ""


    if crop != "":

        crop_path = os.path.abspath(os.path.join(base_dir,experiment,"01_data",file_name,crop))

        with open(crop_path,"r") as f:
            crop = json.load(f)
        
        crop_name = crop["name"]
        crop_roi = daisy.Roi(crop["offset"],crop["shape"])

        affs_file = os.path.join(affs_file,crop_name+'.zarr')
    
    else:
        crop_name = ""
        crop_roi = None

    fragments_file = affs_file

    # block_directory = os.path.join(fragments_file,'block_nodes')

    logging.info("Reading affs from %s", affs_file)
    affs = open_ds(affs_file, affs_dataset, mode='r')

    if block_size == [0,0,0]:
        context = [50,40,40]
        block_size = crop_roi.shape if crop_roi else affs.roi.shape

    logging.info("Reading fragments from %s", fragments_file)
    fragments = open_ds(fragments_file, fragments_dataset, mode='r')

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)

    read_roi = daisy.Roi((0,)*affs.roi.dims, block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims, block_size)

    task = daisy.Task(
        'AgglomerateBlockwiseTask',
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: agglomerate_worker(
            b,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            # block_directory,
            write_roi.shape,
            merge_function),
        num_workers=num_workers,
        read_write_conflict=False,
        timeout=5,
        fit='shrink')

    done = daisy.run_blockwise([task])

    if not done:
        raise RuntimeError("at least one block failed!")

    block_size = [0,0,0]

def agglomerate_worker(
        block,
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        # block_directory,
        write_size,
        merge_function):

    waterz_merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }[merge_function]

    logging.info(f"Reading affs from {affs_file}")
    affs = open_ds(affs_file, affs_dataset)

    logging.info(f"Reading fragments from {fragments_file}")
    fragments = open_ds(fragments_file, fragments_dataset)

    #opening RAG file
    # rag_provider = FileGraphProvider(
    #     directory=block_directory,
    #     chunk_size=write_size,
    #     mode='r+',
    #     directed=False,
    #     edges_collection='edges_' + merge_function,
    #     position_attribute=['center_z', 'center_y', 'center_x']
    #     )
    logging.info("Opening RAG MongoDB...")
    rag_provider = MongoDbGraphProvider(
        db_name="mongo_lsd",
        host="localhost",  # Optional, if not localhost
        mode="r+",  # Read and write mode
        directed=False,  # Graph is not directed
        nodes_collection="nodes",  # Optional, if different from default
        edges_collection='edges_' + merge_function,
        meta_collection="meta",  # Optional, if different from default
        position_attribute=['center_z', 'center_y', 'center_x']  # Position attributes
    )
    logging.info("RAG file opened")

    agglomerate_in_block(
            affs,
            fragments,
            rag_provider,
            block,
            merge_function=waterz_merge_function,
            threshold=1.0)

def check_block(blocks_agglomerated, block):

    done = blocks_agglomerated.count({'block_id': block.block_id}) >= 1

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    agglomerate(**config)

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to agglomerate: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
