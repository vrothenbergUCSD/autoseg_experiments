from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
import zarr
import argparse

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, "config.json"), "r") as f:
    net_config = json.load(f)

# voxels
input_shape = Coordinate(net_config["input_shape"])
output_shape = Coordinate(net_config["output_shape"])

# nm
voxel_size = Coordinate((50, 10, 10))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size


def predict(iteration, raw_file, raw_dataset, out_file, out_datasets):
    raw = ArrayKey("RAW")
    affs = ArrayKey("AFFS")
    lsds = ArrayKey("LSDS")

    scan_request = BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(affs, output_size)
    scan_request.add(lsds, output_size)

    context = (input_size - output_size) / 2

    source = ZarrSource(
        raw_file,
        datasets={raw: raw_dataset},
        array_specs={
            raw: ArraySpec(interpolatable=True),
        },
    )

    with build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "w")

    # for ds_name, data in out_datasets:
    #     ds = f.create_dataset(
    #         ds_name,
    #         shape=[data["out_dims"]] + list(total_output_roi.get_shape() / voxel_size),
    #         dtype=data["out_dtype"],
    #     )

    #     ds.attrs["resolution"] = voxel_size
    #     ds.attrs["offset"] = total_output_roi.get_offset()
    for ds_name, channels in out_datasets:
        ds = f.create_dataset(
            ds_name,
            shape=[channels] + list(total_output_roi.get_shape() / voxel_size),
            dtype=np.uint8,
        )

        ds.attrs["resolution"] = voxel_size
        ds.attrs["offset"] = total_output_roi.get_offset()

    pipeline = source

    pipeline += Pad(raw, size=None)

    pipeline += Normalize(raw)

    pipeline += IntensityScaleShift(raw, 2, -1)

    pipeline += Predict(
        os.path.join(setup_dir, "train_net_checkpoint_%d" % iteration),
        graph=os.path.join(setup_dir, "config.meta"),
        # max_shared_memory=(2*1024*1024*1024),
        inputs={net_config["raw"]: raw},
        outputs={net_config["affs"]: affs, net_config["embedding"]: lsds},
    )

    pipeline += IntensityScaleShift(affs, 255, 0)
    pipeline += IntensityScaleShift(lsds, 255, 0)

    pipeline += ZarrWrite(
        dataset_names={
            affs: "affs",
            lsds: "lsds",
        },
        output_filename=out_file,
    )

    pipeline += Scan(scan_request)

    predict_request = BatchRequest()

    predict_request.add(raw, total_input_roi.get_shape())
    predict_request.add(affs, total_output_roi.get_shape())
    predict_request.add(lsds, total_output_roi.get_shape())

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")


def copy_all_datasets_to_output(raw_file, out_file):
    # Open the source and destination files
    source_zarr = zarr.open(raw_file, mode='r')
    dest_zarr = zarr.open(out_file, mode='a')  # 'a' mode allows to read and write to the file
    # Copy all groups, datasets, zattrs files to output
    zarr.copy_all(source_zarr, dest_zarr, if_exists='replace')

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

#     iteration = 400000
#     raw_file = "../../../01_data/funke/zebrafinch/training/gt_z255-383_y1407-1663_x1535-1791.zarr"
#     raw_dataset = "volumes/raw"
#     out_file = "test_prediction.zarr"
#     out_datasets = net_config["outputs"]

#     predict(iteration, raw_file, raw_dataset, out_file, out_datasets)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Prediction script, generates affinities and local shape descriptors.")
    parser.add_argument("--model", "-m", type=str, help="Model checkpoint path")
    parser.add_argument("--raw_file", "-rf", type=str, default="/data/lsd_nm_experiments/03_salk/salk/3M-APP-SCN/training/CompImBio10k.zarr", help="Path to the raw data file")
    parser.add_argument("--raw_dataset", "-rd", type=str, default="volumes/raw", help="Raw data dataset path within the file")
    parser.add_argument("--out_file", "-o", type=str, help="Output prediction file name")


    args = parser.parse_args()


    model_checkpoint = args.model
    iteration = os.path.splitext(os.path.basename(model_checkpoint))[0].split('_')[-1]
    raw_file = args.raw_file
    raw_file_basename = os.path.splitext(os.path.basename(args.raw_file))[0]
    raw_dataset = args.raw_dataset
    out_file = args.out_file if args.out_file else f"prediction_{raw_file_basename}_mtlsd_04_{iteration}.zarr"
    out_datasets = [("pred_affs", 3), ("pred_lsds", 10)]

    predict(int(iteration), raw_file, raw_dataset, out_file, out_datasets)

    copy_all_datasets_to_output(raw_file, out_file)
