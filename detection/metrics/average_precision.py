from dataclasses import dataclass
from ssl import DER_cert_to_PEM_cert
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def eucli_dist(lx, ly, dx, dy):
    return torch.sqrt((lx - dx) ** 2 + (ly - dy) ** 2)


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    # TODO: Replace this stub code.
    # return PRCurve(torch.zeros(0), torch.zeros(0))
    n = len(frames)
    detections = torch.empty(2, 1)
    FN = 0
    total_TL = 0

    for i in range(n):
        temp_detect = frames[i].detections
        label_cents = frames[i].labels.centroids
        total_TL += len(frames[i].labels)

        TP_arr = torch.zeros(len(temp_detect))

        for cent in label_cents:
            ed = eucli_dist(
                cent[0], cent[1], temp_detect.centroids_x, temp_detect.centroids_y
            )

            ed_mask = ed.lt(threshold)

            if not ed_mask.count_nonzero():
                FN += 1
                continue

            scores_mask = temp_detect.scores * ed_mask
            max_score_idx = torch.argmax(scores_mask)
            TP_arr[max_score_idx] = 1

        scores_concat = torch.stack((TP_arr, temp_detect.scores), dim=-1).permute(1, 0)
        detections = torch.cat((detections, scores_concat), dim=-1)

    detections, _ = torch.sort(detections[:, 1:], dim=1, descending=True)
    detections = detections.permute(1, 0)

    m = detections.size()[0]

    precision = torch.zeros(m)
    recall = torch.zeros(m)

    for i in range(1, m + 1):
        TP = torch.count_nonzero(detections[:i, 0])
        precision[i - 1] = TP / i
        recall[i - 1] = TP / total_TL

    return PRCurve(precision, recall)


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Replace this stub code.
    p = curve.precision
    r = curve.recall
    r[0] = 0
    sums = 0

    for i in range(1, p.size()[0]):
        sums += (r[i] - r[i - 1]) * p[i]

    return sums


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Replace this stub code.
    pr_curve = compute_precision_recall_curve(frames, threshold)
    ap = compute_area_under_curve(pr_curve)
    return AveragePrecisionMetric(ap, pr_curve)
