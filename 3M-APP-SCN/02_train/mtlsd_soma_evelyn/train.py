import gunpowder as gp
import logging
import math
import sys
import numpy as np
import random
import torch
import zarr
# import tensorflow as tf
from funlib.learn.torch.models import UNet, ConvPass
from lsd.train.gp import AddLocalShapeDescriptor
from skimage.measure import label as relabel

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

data_path = "/data/base/3M-APP-SCN/01_data/APP-3M-SCN-SomaGT.zarr"

def calc_max_padding(
    output_size, voxel_size, neighborhood=None, sigma=None, mode="shrink"
):
    if neighborhood is not None:
        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = gp.Coordinate(
            [np.abs(aff) for val in neighborhood for aff in val if aff != 0]
        )

        method_padding = voxel_size * max_affinity

    if sigma:
        method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, prediction, target, weights):
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        pred_lsds=None,
        gt_lsds=None,
        lsds_weights=None,
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
    ):
        lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)
        aff_loss = self._calc_loss(pred_affs, gt_affs, affs_weights)

        return lsd_loss + aff_loss


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


def pipeline(iteration):
    # if tf.train.latest_checkpoint("."):
    #     trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    # else:
    #     trained_until = 0
    # if trained_until >= max_iteration:
    #     return
    

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    # unlabelled = gp.ArrayKey("UNLABELLED")
    # object_mask = gp.ArrayKey("OBJECT_MASK")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    gt_lsds_mask = gp.ArrayKey("LSDS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    voxel_size = gp.Coordinate((50, 10, 10))

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

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    input_shape = [48, 196, 196]
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    sigma = 100

    # if using an elastic augment
    labels_padding = calc_max_padding(output_size, voxel_size, sigma=sigma)

    # for now this is fine
    # labels_padding = (input_size - output_size) / 2

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    # request.add(unlabelled, output_size)
    # request.add(object_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds, output_size)
    request.add(affs_weights, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(gt_lsds_mask, output_size)
    request.add(pred_affs, output_size)
    request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
        # "../data.zarr",
        data_path,
        {
            raw: "raw",
            labels: "soma",
            labels_mask: "soma_mask",
            # object_mask: "volumes/labels/object_mask",
            # unlabelled: "unlabelled",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
            # object_mask: gp.ArraySpec(interpolatable=False),
            # unlabelled: gp.ArraySpec(interpolatable=False),
        },
    )

    source += gp.Normalize(raw)
    source += gp.Pad(raw, None)
    source += gp.Pad(labels, labels_padding)
    source += gp.Pad(labels_mask, labels_padding)
    # source += gp.Pad(object_mask, labels_padding)
    # source += gp.Pad(unlabelled, labels_padding)
    # source += gp.RandomLocation(mask=object_mask, min_masked=0.3)
    source += gp.RandomLocation()

    pipeline = source

    pipeline += gp.RandomProvider()

    # leave out for now

    pipeline += gp.ElasticAugment(
    control_point_spacing=[5, 5, 10],
    jitter_sigma=[0, 2, 2],
    rotation_interval=[0, math.pi / 2.0],
    prob_slip=0.05, # if you need to simulate shifted sections
    prob_shift=0.05, # same as above
    max_misalign=10, # same as above
    subsample=8, # for efficiency
    )

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    # hmm it seems like there are a few thin labels that will be screwed up if
    # we grow the boundary and there is already sufficient space between labels
    # so should be fine to skip

    # pipeline += gp.GrowBoundary(labels, labels_mask, steps=1, only_xy=True)

    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        sigma=sigma,
        # unlabelled=unlabelled,
        lsds_mask=gt_lsds_mask,
        downsample=2,
    )

    pipeline += gp.AddAffinities(
        affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        labels=labels,
        affinities=gt_affs,
        # unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: gt_lsds_mask,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights,
        },
        outputs={0: pred_lsds, 1: pred_affs},
        save_every=5000,
        log_dir="log"
    )

    pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])
    pipeline += gp.Squeeze([raw])

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            labels: "labels",
            gt_lsds: "gt_lsds",
            pred_lsds: "pred_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        every=5000,
    )

    with gp.build(pipeline):
        for i in range(iteration):
            pipeline.request_batch(request)


if __name__ == "__main__":
    iteration = int(sys.argv[1])
    if not isinstance(iteration, int):
        iteration = 50000
    pipeline(iteration)

