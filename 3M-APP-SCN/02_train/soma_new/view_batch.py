import argparse
import neuroglancer
import numpy as np
import os
import sys
# import h5py
import zarr

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file", "-f", type=str, action="append", help="The path to the hdf batch to show"
)

parser.add_argument(
    "--bind-address",
    "-b",
    type=str,
    default="0.0.0.0",
    help="The bind address to use",
)

parser.add_argument(
    "--bind-port",
    "-p",
    type=int,
    default=8080,  # default to 0 (random port)
    help="The bind port to use",
)

args = parser.parse_args()

neuroglancer.set_server_bind_address(args.bind_address, args.bind_port)

# f = h5py.File(args.file[0])
f = zarr.open(args.file[0])

datasets = [
    "raw",
    "gt_affs",
    "pred_affs",
    "labels",
    "pred_lsds",
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
        print(ds)
        res = list(f[ds].attrs["resolution"])
        offset = [i / j for i, j in zip(list(f[ds].attrs["offset"]), res)]

        data = f[ds][:]

        if len(data.shape) == 4:
            names = ["c^", "x", "y", "z"]
            offset = [0] + offset
            res = [1] + res
        else:
            names = ["x", "y", "z"]

        dims = neuroglancer.CoordinateSpace(names=names, units="nm", scales=res)

        if "gt" in ds:
            data = data.astype(np.float32)

        layer = neuroglancer.LocalVolume(
            data=data, voxel_offset=offset, dimensions=dims
        )

        layer_type = (
            neuroglancer.SegmentationLayer
            if data.dtype == np.uint64
            else neuroglancer.ImageLayer
        )

        if layer_type == neuroglancer.SegmentationLayer:
            s.layers[ds] = layer_type(source=layer)
        else:
            s.layers[ds] = layer_type(source=layer, shader=shader)

print(viewer.get_viewer_url())

aws_external = 'localhost'
aws_internal = 'http://ip-172-31-7-233.us-west-2.compute.internal'
url = viewer.get_viewer_url()
new_url = url.replace(aws_internal, aws_external)
print(new_url)

# Usage:
# python -i view_batch.py -f /data/base/3M-APP-SCN/02_train/soma_new/snapshots/batch_295001.zarr

# Forward port 8080
# Go to localhost:8080/v/de6b91f763b546f985184e2f16c9ed5079ca0fd1/