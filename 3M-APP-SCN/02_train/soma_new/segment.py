import numpy as np
import waterz
import zarr


def get_segmentation(affinities, fragments, threshold):
    thresholds = [threshold]
    segmentations = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(segmentations)
    return segmentation


if __name__ == "__main__":
    f = zarr.open("prediction_200k.zarr", "a")

    affs = f["pred_affs"][:]
    frags = f["frags"][:]

    affs = (affs / np.max(affs)).astype(np.float32)

    # add thresholds as needed
    thresholds = [0.5]

    for threshold in thresholds:
        print(f"segmenting {threshold}")

        segmentation = get_segmentation(affs, frags, threshold)

        f[f"seg_{threshold}"] = segmentation
        f[f"seg_{threshold}"].attrs["offset"] = f["pred_affs"].attrs["offset"]
        f[f"seg_{threshold}"].attrs["resolution"] = f["pred_affs"].attrs["resolution"]
