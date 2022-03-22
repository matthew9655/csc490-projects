from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the greedy matching algorithm.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    M, N = cost_matrix.shape[0], cost_matrix.shape[1]
    min_values = np.dstack(np.unravel_index(np.argsort(cost_matrix.ravel()), (M, N)))[0]
    i_set, j_set = set(), set()
    row_ids, col_ids = [], []

    count = 0
    max_count = min(M, N)
    for i in range(min_values.shape[0]):
        if count == max_count:
            break
        min_i, min_j = min_values[i]
        if min_i not in i_set and min_j not in j_set:
            i_set.add(min_i)
            j_set.add(min_j)
            row_ids.append(min_i)
            col_ids.append(min_j)
            count += 1

    return row_ids, col_ids


def hungarian_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the Hungarian matching algorithm.
    For simplicity, we just call the scipy `linear_sum_assignment` function. Please refer to
    https://en.wikipedia.org/wiki/Hungarian_algorithm and
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    for more details of the hungarian matching implementation.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    return list(row_ids), list(col_ids)
