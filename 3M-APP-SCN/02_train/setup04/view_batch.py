import argparse
import neuroglancer
import numpy as np
import os
import sys
import h5py

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

f = h5py.File(args.file[0])

datasets = [
    "volumes/raw",
    "volumes/gt_affinities",
    "volumes/pred_affinities",
    "volumes/gt_embedding",
    "volumes/pred_embedding",
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

        s.layers[ds] = layer_type(source=layer, shader=shader)

aws_external = 'ec2-18-236-204-101.us-west-2.compute.amazonaws.com'
aws_internal = 'ip-172-31-1-72.us-west-2.compute.internal'
url = viewer.get_viewer_url()
new_url = url.replace(aws_internal, aws_external)
print(new_url)
