'''
Usage: python -i view_data.py -d /data/base/3M-APP-SCN/01_data/APP-3M-SCN-3000.zarr

Adjust datasets to match Zarr
'''

import argparse
import neuroglancer
import numpy as np
import os
import sys
import zarr

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    "-d",
    type=str,
    action="append",
    help="The path to the zarr container to show",
)

parser.add_argument(
    "--bind-address",
    "-b",
    type=str,
    default="localhost",
    help="The bind address to use",
)

args = parser.parse_args()

neuroglancer.set_server_bind_address(args.bind_address)

f = zarr.open(args.data_dir[0])



datasets = [
    "raw",
    # "labels",
    "soma",
    "soma_mask",
    "soma_invert"
    # "soma_segmentation_0.3_filtered"
    # "labels_mask",
    # "unlabelled"
]

viewer = neuroglancer.Viewer()

shader = """
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}"""

with viewer.txn() as s:
    for ds in datasets:
        print('ds', ds)
        res = f[ds].attrs["resolution"]
        offset = f[ds].attrs["offset"]


        print(res, offset)

        dims = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"], units="nm", scales=res
        )

        # dims = neuroglancer.CoordinateSpace(
        #     names=["x", "y", "z"], units="nm", scales=res
        # )

        #Convert offset to voxel coordinates
        voxel_offset = [o / r for o, r in zip(offset, res)]

        data = f[ds][:]

        if "mask" in ds:
            data *= 255

        layer = neuroglancer.LocalVolume(
            data=data, voxel_offset=voxel_offset, dimensions=dims
        )

        if "label" in ds or "soma" in ds:
            print('Label - Segmentation')
            layer_type = neuroglancer.SegmentationLayer
            s.layers[ds] = layer_type(source=layer)
        elif data.dtype == np.uint64:
            layer_type = neuroglancer.SegmentationLayer
            s.layers[ds] = layer_type(source=layer)
        else:
            layer_type = neuroglancer.ImageLayer
            s.layers[ds] = layer_type(source=layer, shader=shader)

        # layer_type = (
        #     neuroglancer.SegmentationLayer
        #     if data.dtype == np.uint64
        #     else neuroglancer.ImageLayer
        # )

        # s.layers[ds] = layer_type(source=layer)

print(viewer)