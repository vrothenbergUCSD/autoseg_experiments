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

f = zarr.open(args.data_dir[0])

print(f)

def shader_fxn(x,y,z):
    string = f"""\
void main() {{
    emitRGB(
        vec3(
            toNormalized(getDataValue({x})),
            toNormalized(getDataValue({y})),
            toNormalized(getDataValue({z})))
        );
}}"""
    return string

shader = """void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}"""
shader_1= """void main() {
    float v = toNormalized(getDataValue(10));
    vec4 rgba = vec4(0,0,0,0);
    if (v != 0.0) {
        rgba = vec4(colormapJet(v), 1.0);
    }
    emitRGBA(rgba);
}"""

   
# Add 'pred_affs' to the datasets to visualize
datasets = ["raw", 
            "pred_affs",
            "pred_lsds",
            "labels", 
            ]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    for ds in datasets:
        print('ds:', ds)
        res = f[ds].attrs["resolution"]
        print('res', res)
        if len(res) == 3:
            res = [1] + res
            print('new res:', res)
        offset = list(f[ds].attrs["offset"])
        print('offset:', offset)
        if len(offset) == 3:
            offset = [0] + offset
            print('new offset:', offset)
        offset = [int(x / y) for x, y in zip(offset, res)]
        print('adjusted_offset', offset)

        dims = neuroglancer.CoordinateSpace(
            names=["c^", "z", "y", "x"], units="nm", scales=res
        )
        
        data = f[ds][:]
        print('data:', data.shape)

        # Add additional dimension for 3d raw volumes
        if data.ndim == 3:
            data = data[np.newaxis, :]

        print('data:', data.shape)
        print('dims:', dims)

        if "mask" in ds:
            data *= 255

        layer = neuroglancer.LocalVolume(
            data=data, voxel_offset=offset, dimensions=dims
        )

        layer_type = (
            neuroglancer.SegmentationLayer
            if data.dtype == np.uint64
            else neuroglancer.ImageLayer
        )

        if 'lsd' in ds:
            com = data[0:3, :, :, :]
            print('com.shape', com.shape)
            cov = data[3:6, :, :, :]
            print('cov.shape', cov.shape)
            cor = data[6:9, :, :, :]
            print('cor.shape', cor.shape)
            vox = data[9:10, :, :, :]
            print('vox.shape', vox.shape)
            layer_com = neuroglancer.LocalVolume(
                data=com, voxel_offset=offset, dimensions=dims
            )
            layer_cov = neuroglancer.LocalVolume(
                data=cov, voxel_offset=offset, dimensions=dims
            )
            layer_cor = neuroglancer.LocalVolume(
                data=cor, voxel_offset=offset, dimensions=dims
            )
            layer_vox = neuroglancer.LocalVolume(
                data=vox, voxel_offset=offset, dimensions=dims
            )
            s.layers['com'] = neuroglancer.ImageLayer(source=layer_com, shader=shader)
            s.layers['cov'] = neuroglancer.ImageLayer(source=layer_cov, shader=shader)
            s.layers['cor'] = neuroglancer.ImageLayer(source=layer_cor, shader=shader)
            s.layers['vox'] = neuroglancer.ImageLayer(source=layer_vox, shader=shader_1)
            
        else:
            s.layers[ds] = layer_type(source=layer)

import requests

def get_public_hostname():
    url = "http://169.254.169.254/latest/meta-data/public-hostname"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")

public_hostname = get_public_hostname()
if public_hostname:
    print(f"Public Hostname: {public_hostname}")

url = viewer.get_viewer_url()
new_url = url.replace('ip-172-31-1-72.us-west-2.compute.internal', public_hostname)
print(new_url)
