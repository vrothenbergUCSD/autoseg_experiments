import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr

from funlib.learn.torch.models import UNet, ConvPass

import argparse
import shutil
import json



logging.basicConfig(level=logging.INFO)

setup_dir = os.path.dirname(os.path.realpath(__file__))



class MTLSDModel(torch.nn.Module):
    def __init__(self, unet, num_fmaps):
        super(MTLSDModel, self).__init__()

        self.unet = unet
        self.aff_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation="Sigmoid")
        self.lsd_head = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)

        lsds = self.lsd_head(x[0])  # decoder 1
        affs = self.aff_head(x[1])  # decoder 2
        return lsds, affs


def predict(
        raw_file, 
        raw_dataset, 
        out_file_path,
        model_checkpoint,
        run_config,
        **kwargs):
    print('predict.py')
    print(run_config)
    worker_config = run_config['worker_config']
    logging.info(f'predict.py {raw_file}, {raw_dataset}, {out_file_path}, {model_checkpoint}')
    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    # voxel_size = gp.Coordinate((50, 10, 10))
    voxel_size = gp.Coordinate((300, 156, 156))

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5

    downsample_factors = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]

    kernel_size_down = [
        [(3,) * 3, (3,) * 3],
        [(3,) * 3, (3,) * 3],
        [(3,) * 3, (3,) * 3],
        [(1, 3, 3), (1, 3, 3)],
    ]

    kernel_size_up = [
        [(1, 3, 3), (1, 3, 3)],
        [(3,) * 3, (3,) * 3],
        [(3,) * 3, (3,) * 3],
    ]

    unet = UNet(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        constant_upsample=True,
        num_fmaps_out=14,
        num_heads=2,  # uses two decoders
    )

    model = MTLSDModel(unet, num_fmaps=14)

    input_shape = [48, 196, 196]
    # input_shape = [48, 98, 98]
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]
    # output_shape = [28, 104, 104]

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = (input_size - output_size) / 2

    model.eval()

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_affs, output_size)
    chunk_request.add(pred_lsds, output_size)

    print('chunk_request input_size: ', input_size)
    print('chunk_request output_size: ', output_size)


    pipeline = gp.ZarrSource(
        raw_file, 
        datasets = {
            raw: raw_dataset
        },
        array_specs = {
            raw: gp.ArraySpec(interpolatable=True)
        }
    )

    pipeline += gp.Pad(raw, size=None)

    # with gp.build(source):
    #     total_input_roi = source.spec[raw].roi
    #     total_output_roi = total_input_roi.grow(-context, -context)

    # f = zarr.open(out_file, "w")

    # for ds_name, channels in out_datasets:
    #     ds = f.create_dataset(
    #         ds_name,
    #         shape=[channels] + list(total_output_roi.get_shape() / voxel_size),
    #         dtype=np.uint8,
    #     )

    #     ds.attrs["resolution"] = voxel_size
    #     ds.attrs["offset"] = total_output_roi.get_offset()

    pipeline += gp.Normalize(raw)
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Unsqueeze([raw])

    # pipeline += gp.IntensityScaleShift(raw, 2,-1)
    model_checkpoint = os.path.join(run_config['setup_dir'], model_checkpoint)

    pipeline += gp.torch.Predict(
        model,
        checkpoint=model_checkpoint,
        inputs={"input": raw},
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
    )

    pipeline += gp.Squeeze([pred_affs])
    pipeline += gp.Squeeze([pred_lsds])
    pipeline += gp.Normalize(pred_affs)
    pipeline += gp.IntensityScaleShift(pred_affs, 255, 0)
    pipeline += gp.IntensityScaleShift(pred_lsds, 255, 0)

    out_dir, out_file_name = os.path.split(out_file_path) 

    pipeline += gp.ZarrWrite(
        dataset_names={
            pred_affs: 'pred_affs',
            pred_lsds: 'pred_lsds',
        },
        output_dir=out_dir,
        output_filename=out_file_name,
    )

    pipeline += gp.PrintProfilingStats(every=10)

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: 'read_roi',
            pred_affs: 'write_roi',
            pred_lsds: 'write_roi',
        },
        num_workers=worker_config['num_cache_workers']
        )
    
    # pipeline = (
    #     source
    #     + gp.Normalize(raw)
    #     + gp.Unsqueeze([raw])
    #     + gp.Unsqueeze([raw])
    #     + predict
    #     + gp.Squeeze([pred_affs])
    #     + gp.Squeeze([pred_lsds])
    #     + gp.Normalize(pred_affs)
    #     + gp.IntensityScaleShift(pred_affs, 255, 0)
    #     + gp.IntensityScaleShift(pred_lsds, 255, 0)
    #     + write
    #     + gp.PrintProfilingStats(every=10)
    #     + block_request
    # )

    # predict_request = gp.BatchRequest()

    # predict_request[raw] = total_input_roi
    # predict_request[pred_affs] = total_output_roi
    # predict_request[pred_lsds] = total_output_roi
    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    print("Prediction finished.")

def copy_all_datasets_to_output(raw_file, out_file):
    # Open the source and destination files
    source_zarr = zarr.open(raw_file, mode='r')
    dest_zarr = zarr.open(out_file, mode='a')  # 'a' mode allows to read and write to the file
    # Copy all groups, datasets, zattrs files to output
    zarr.copy_all(source_zarr, dest_zarr, if_exists='replace')

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    iteration = run_config['iteration']
    raw_file = run_config['raw_file']
    raw_dataset = run_config['raw_dataset']
    out_file = run_config['out_file']
    graph_dir = run_config['graph_dir']
    
    # args = parser.parse_args()
    model_checkpoint = f'model_checkpoint_{iteration}'
    # iteration = os.path.splitext(os.path.basename(model_checkpoint))[0].split('_')[-1]
    # raw_file = args.raw_file
    raw_file_basename = os.path.splitext(os.path.basename(raw_file))[0]
    # raw_dataset = args.raw_dataset
    out_file = out_file if out_file else f"prediction_{raw_file_basename}_setup04_{iteration}.zarr"
    out_datasets = [("pred_affs", 3), ("pred_lsds", 10)]

    predict(raw_file, raw_dataset, out_file, out_datasets, model_checkpoint, run_config)

    copy_all_datasets_to_output(raw_file, out_file)


