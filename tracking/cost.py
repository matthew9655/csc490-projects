import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
from shapely.geometry import Point, box


def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]

    # TODO: Replace this stub code.
    intersection = np.zeros((M, N))
    union = np.zeros((M, N))

    poly1 = []
    poly2 = []

    # get polygons for bboxes1
    for i in range(M):
        x, y, l, w, yaw = bboxes1[i, :]
        dx = np.cos(yaw) * l - np.sin(yaw) * w
        dy = np.sin(yaw) * l + np.cos(yaw) * w

        min_x = x - dx / 2
        min_y = y - dy / 2

        polygon = box(min_x, min_y, min_x + l, min_y + w)
        poly1.append(
            affinity.rotate(
                polygon, yaw, origin=Point((min_x, min_y)), use_radians=True
            )
        )

    # get polygons for bboxes2
    for i in range(N):
        x, y, l, w, yaw = bboxes2[i, :]
        dx = np.cos(yaw) * l - np.sin(yaw) * w
        dy = np.sin(yaw) * l + np.cos(yaw) * w

        min_x = x - dx / 2
        min_y = y - dy / 2

        polygon = box(min_x, min_y, min_x + l, min_y + w)
        poly2.append(
            affinity.rotate(
                polygon, yaw, origin=Point((min_x, min_y)), use_radians=True
            )
        )

    for i in range(M):
        for j in range(N):
            union[i, j] = poly1[i].union(poly2[j]).area
            intersection[i, j] = poly1[i].intersection(poly2[j]).area

    return intersection / union


if __name__ == "__main__":
    a, b = np.zeros((5, 5)), np.zeros((5, 5))
    a = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]])
    b = np.array([[1.0, 0.0, 2.0, 1.0, 0.0]])
    iou_2d(a, b)
