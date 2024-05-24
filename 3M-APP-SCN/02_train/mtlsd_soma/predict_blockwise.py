import daisy
import datetime
import hashlib
import json
import logging
import numpy as np
import os
import sys
import time
from funlib.persistence.arrays import open_ds, prepare_ds
from funlib.persistence.graphs import FileGraphProvider
from typing import Dict, Any, Optional
import subprocess


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename='predict_blockwise.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def predict_blockwise(
        base_dir,
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        out_base,
        out_file,
        num_workers,
        worker_config,
        **kwargs):

    '''

    Run prediction in parallel blocks. Within blocks, predict in chunks.


    Assumes a general directory structure:


    base
    ├── fib25 (experiment dir)
    │   │
    │   ├── 01_data (data dir)
    │   │   └── training_data (i.e zarr/n5, etc)
    │   │
    │   └── 02_train (train/predict dir)
    │       │
    │       ├── setup01 (setup dir - e.g baseline affinities)
    │       │   │
    │       │   ├── config.json (specifies meta data for inference)
    │       │   │
    │       │   ├── mknet.py (creates network, jsons to be used)
    │       │   │
    │       │   ├── model_checkpoint (saved network checkpoint for inference)
    │       │   │
    │       │   ├── predict.py (worker inference file - logic to be distributed)
    │       │   │
    │       │   ├── train_net.json (specifies meta data for training)
    │       │   │
    │       │   └── train.py (network training script)
    │       │
    │       ├──    .
    │       ├──    .
    │       ├──    .
    │       └── setup{n}
    │
    ├── hemi-brain
    ├── zebrafinch
    ├──     .
    ├──     .
    ├──     .
    └── experiment{n}

    Args:

        base_dir (``string``):

            Path to base directory containing experiment sub directories.

        experiment (``string``):

            Name of the experiment (fib25, hemi, zfinch, ...).

        setup (``string``):

            Name of the setup to predict (setup01, setup02, ...).

        iteration (``int``):

            Training iteration to predict from.

        raw_file (``string``):

            Path to raw file (zarr/n5) - can also be a json container
            specifying a crop, where offset and size are in world units:

                {
                    "container": "path/to/raw",
                    "offset": [z, y, x],
                    "size": [z, y, x]
                }

        raw_dataset (``string``):

            Raw dataset to use (e.g 'volumes/raw'). If using a scale pyramid,
            will try scale zero assuming stored in directory `s0` (e.g
            'volumes/raw/s0')

        out_base (``string``):

            Path to base directory where zarr/n5 should be stored. The out_file
            will be built from this directory, setup, iteration, file name

            **Note:

                out_dataset no longer needed as input, build out_dataset from config
                outputs dictionary generated in mknet.py (config.json for
                example)

        out_file (``string``):

            Output file name, zarr/n5

        num_workers (``int``):

            How many blocks to run in parallel.

        worker_config (``string``):

            "worker_config": {
                "queue": "gpu_a10g",
                "num_cpus": 3,
                "num_cache_workers": 2
            }

    '''

    #get relevant dirs + files

    experiment_dir = os.path.join(base_dir, experiment)
    setup_dir = os.path.join(experiment_dir, '02_train', setup)
    out_file_basename = out_file.split('.zarr')[0]
    out_dir = os.path.join(setup_dir, out_base)

    raw_file = os.path.abspath(raw_file)
    
    out_file_path = os.path.join(out_dir, out_file)

    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    try:
        source = open_ds(raw_file, raw_dataset)
    except:
        raw_dataset = raw_dataset + '/s0'
        source = open_ds(raw_file, raw_dataset)

    logging.info(f'Source shape: {source.shape}')
    logging.info(f'Source roi: {source.roi}')
    logging.info(f'Source voxel size: {source.voxel_size}')

    outputs = {"pred_affs": {"out_dims": 3, "out_dtype": "uint8"}, "pred_lsds": {"out_dims": 10, "out_dtype": "uint8"}}

    # get chunk size and context for network (since unet has smaller output size
    # than input size
    input_shape = [48, 196, 196]
    output_shape = [28, 104, 104]
    net_input_size = daisy.Coordinate(input_shape)*source.voxel_size
    net_output_size = daisy.Coordinate(output_shape)*source.voxel_size

    context = (net_input_size - net_output_size)/2
    logging.info(f'context: {context}') # (500, 460, 460)

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    ndims = source.roi.dims
    block_read_roi = daisy.Roi((0,)*ndims, net_input_size) - context
    block_write_roi = daisy.Roi((0,)*ndims, net_output_size)
    logging.info(f'block_read_roi: {block_read_roi}') # (2400, 1960, 1960)
    logging.info(f'block_write_roi: {block_write_roi}') # (1400, 1040, 1040)

    logging.info('Preparing output dataset...')

    # get output file(s) meta data from config.json, prepare dataset(s)
    for output_name, val in outputs.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        # out_dataset = '%s'%output_name
        out_dataset = os.path.join(output_name)
        logging.info(f'out_dataset: {out_dataset}')
        # lsd chunk_shape (10, 28, 104, 104)

        ds = prepare_ds(
            out_file_path,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            compressor={'id': 'gzip', 'level':5})

    logging.info('Starting block-wise processing...')

    task = daisy.Task(
        task_id='PredictBlockwiseTask',
        total_roi=input_roi,
        read_roi=block_read_roi,
        write_roi=block_write_roi,
        process_function=lambda b: predict_worker(
                experiment,
                setup_dir,
                out_dir,
                iteration,
                raw_file,
                raw_dataset,
                out_file_path,
                worker_config
                ),
        check_function=None,
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang'
    )

    succeeded = daisy.run_blockwise([task])

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")
    

def predict_worker(
        experiment,
        setup_dir,
        out_dir,
        iteration,
        raw_file,
        raw_dataset,
        out_file_path,
        worker_config
        ):
    
    sys.path.append(setup_dir)
    from predict import predict

    # get the relevant worker script to distribute
    # setup_dir = os.path.join(base_dir, experiment, '02_train', setup)
    # predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'out_file_path': out_file_path,
        'setup_dir': setup_dir,
        'worker_config': worker_config
    }

    # # get a unique hash for this configuration
    # config_str = ''.join(['%s'%(v,) for v in config.values()])
    # config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    # get worker id
    daisy_context = daisy.Context.from_env()
    worker_id = int(daisy_context._Context__dict['worker_id'])

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%worker_id


    # logging.info('Running block with config %s...'%config_file)
    model_checkpoint = f"model_checkpoint_{config['iteration']}"

    predict(
        raw_file,
        raw_dataset,
        out_file_path,
        model_checkpoint,
        config)
    
    logging.info('predict command called')


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    predict_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to predict: {seconds}')
