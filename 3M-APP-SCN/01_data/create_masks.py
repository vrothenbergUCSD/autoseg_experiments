import numpy as np
import os
import zarr

container = zarr.open("/data/base/3M-APP-SCN/01_data/CompImBio800M.zarr", "a")

labels = container["labels"]
offset = labels.attrs["offset"]
resolution = labels.attrs["resolution"]

# labels_name = "labels"
labels_mask_name = "labels_mask"

labels_mask = np.where(labels[:] > 0, 1, 0).astype(np.uint8)

container[labels_mask_name] = labels_mask
container[labels_mask_name].attrs["offset"] = offset
container[labels_mask_name].attrs["resolution"] = resolution

# for sample in samples:
#     f = zarr.open(sample, "a")

#     labels = f[labels_name][:]
#     offset = f[labels_name].attrs["offset"]
#     resolution = f[labels_name].attrs["resolution"]

#     labels_mask = np.ones_like(labels).astype(np.uint8)


#     f[labels_mask_name] = labels_mask
#     f[labels_mask_name].attrs["offset"] = offset
#     f[labels_mask_name].attrs["resolution"] = resolution
