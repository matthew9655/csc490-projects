import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from detection.types import Detections


@dataclass
class VoxelizerConfig:
    """Configuration class for the voxelizer.

    Attributes:
        x_range: Image range along the x axis in metres ("forward")
        y_range: Image range along the y axis in metres ("sideways")
        z_range: Image range along the z axis in metres ("up")
        step: Voxelization step in metres (i.e. pixel resolution)
    """

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    step: float

    @property
    def bev_size(self) -> Tuple[int, int, int]:
        """Return (depth, height, width) of the voxel grid.

        depth is the size along the z-axis ("up"), height is the size along the y-axis ("sideways"),
        and width is the size along the x-axis ("forward").
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        return (
            int((z_max - z_min) / self.step),
            int((y_max - y_min) / self.step),
            int((x_max - x_min) / self.step),
        )


class Voxelizer(torch.nn.Module):
    """Voxelizer for converting Lidar point cloud to image"""

    def __init__(self, config: VoxelizerConfig) -> None:
        super().__init__()
        self._step = config.step
        self._x_min, self._x_max = config.x_range
        self._y_min, self._y_max = config.y_range
        self._z_min, self._z_max = config.z_range
        self._depth, self._height, self._width = config.bev_size

    def forward(self, pointclouds: List[torch.Tensor]) -> torch.Tensor:
        """Voxelize a batch of lidar pointclouds into a BEV occupacy image.

        This method produces a [batch_size x D x H x W] tensor, where D is the
        size of the voxel grid along the z-axis, H is that along the y-axis, and
        W is that along the x-axis. Each cell in a [D x H x W] voxel grid denotes
        whether a LiDAR point occupies the cell (1) or not (0).

        A LiDAR point (x, y, z) occupies a cell (i, j, k) if and only if:
            floor[(z - z_min) / step] = i
            floor[(y_max - y) / step] = j
            floor[(x - x_min) / step] = k

        The z coordinate of all LiDAR points are clamped between [z_min, z_max].
        The x and y coordinates of all LiDAR points are not. Therefore, if a point
        with coordinates (x, y, z) lies outside the bounds of [x_min, x_max] and
        [y_min, y_max], it will not be represented in the voxel grid.

        Args:
            pointclouds: Batch of [N x 3] lidar pointclouds. Each row is (x, y, z).

        Returns:
            BEV occupacy image as a [batch_size x D x H x W] tensor.
        """
        # TODO: Replace this stub code.
        n = len(pointclouds)
        BEVs = torch.zeros(n, self._depth, self._height, self._width)

        for i in range(n):
            pc = pointclouds[i]
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

            bev_x = torch.floor((x - self._x_min) / self._step)
            bev_x = torch.clip(bev_x, 0, self._width - 1)
            bev_y = torch.floor((self._y_max - y) / self._step)
            bev_y = torch.clip(bev_y, 0, self._height - 1)
            bev_z = torch.floor((z - self._z_min) / self._step)
            bev_z = torch.clip(bev_z, 0, self._depth - 1)

            cx1 = x.ge(self._x_min)
            cx2 = x.le(self._x_max)
            mask_x = torch.all(torch.stack((cx1, cx2), dim=-1), dim=1)

            cy1 = y.ge(self._y_min)
            cy2 = y.le(self._y_max)
            mask_y = torch.all(torch.stack((cy1, cy2), dim=-1), dim=1)

            total_mask = mask_x * mask_y

            wanted_idxs = total_mask.nonzero().flatten()

            stacked_bevs = torch.stack((bev_z, bev_y, bev_x), dim=-1)
            stacked_bevs = stacked_bevs.long()

            BEV_idx = torch.index_select(stacked_bevs, 0, wanted_idxs)

            for j in range(BEV_idx.size()[0]):
                BEVs[i, BEV_idx[j, 0], BEV_idx[j, 1], BEV_idx[j, 2]] = 1

        # for i in range(n):
        #     pc = pointclouds[i]
        #     for j in range(pc.size()[0]):
        #         x, y, z = pc[j]
        #         if self._x_min <= x <= self._x_max and self._y_min <= y <= self._y_max:
        #             vi = math.floor((z - self._z_min) / self._step)
        #             vi = torch.clip(torch.tensor(vi), 0, self._depth - 1).item()
        #             vj = math.floor((self._y_max - y) / self._step)
        #             vk = math.floor((x - self._x_min) / self._step)

        #             BEVs[i, vi, vj, vk] = 1

        return BEVs

    def project_detections(self, detections: Detections) -> Detections:
        """Project detections to voxelized frame and filter out-of-bounds ones.

        Args:
            detections: 2D bounding box detections, in vehicle coordinates.

        Returns:
            2D bounding box detections, in voxel grid coordinates.
        """
        # Remove out of bounds detections
        out_of_bounds = torch.any(
            torch.stack(
                [
                    detections.centroids[:, 0] < self._x_min,
                    detections.centroids[:, 0] > self._x_max,
                    detections.centroids[:, 1] < self._y_min,
                    detections.centroids[:, 1] > self._y_max,
                ]
            ),
            dim=0,
        )
        mask = ~out_of_bounds

        # Transform vehicle coordinates to image
        x, y = detections.centroids[mask, :2].T
        ix = (x - self._x_min) / self._step
        iy = (self._y_max - y) / self._step
        centroids = torch.stack([ix, iy], dim=1)

        # Transform vehicle-frame relative lengths to image lengths (pixels)
        boxes = detections.boxes[mask, :2] / self._step

        # Yaws are unchanged
        yaws = detections.yaws[mask]

        # Scores are unchanged
        scores = detections.scores[mask] if detections.scores is not None else None

        return Detections(centroids, yaws, boxes, scores)
