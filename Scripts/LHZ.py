import numpy as np
from typing import Iterable

def normalize_label(label):
    """
    Convert labels such as 0, (0,), [0, 1], {0, 1} into sorted tuples.
    """
    if isinstance(label, int):
        return (label,)

    if isinstance(label, tuple):
        return tuple(sorted(label))

    if isinstance(label, (list, set, frozenset)):
        return tuple(sorted(label))

    raise TypeError(f"Unsupported label type: {type(label)}")


def symdiff_labels(labels: Iterable):
    """
    Symmetric difference of tuple labels.
    """
    out = set()
    for label in labels:
        out ^= set(label)
    return tuple(sorted(out))


def next_lhz_label(label, n: int):
    """
    Horizontal update rule for the extended LHZ code.
    """
    if len(label) == 1:
        x = label[0]
        if x < n - 1:
            return (0, x + 1)
        return (0,)

    if len(label) == 2:
        x, y = label

        if x < n - 1 and y < n - 1:
            return (x + 1, y + 1)

        if y == n - 1:
            return (x + 1,)

    raise ValueError(f"Invalid label transition from {label}")


def build_extended_lhz_array(n: int) -> np.ndarray:
    """
    Build the single extended LHZ architecture as an n x 2(n+1) numpy array.

    Entries are either None or tuple-valued parity labels.

    The horizontal direction is periodic.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    n_rows = n
    n_cols = 2 * (n + 1)

    grid = np.full((n_rows, n_cols), None, dtype=object)

    for row in range(n_rows):
        start_col = row % 2

        if row == 0:
            label = (0,)
        elif row == 1:
            label = (1,)
        else:
            label_above = grid[row - 2, start_col]

            if label_above is None:
                raise RuntimeError(
                    f"Expected label two rows above at {(row - 2, start_col)}."
                )

            if len(label_above) == 1:
                x = label_above[0]
                label = (x + 1, n - 1)
            elif len(label_above) == 2:
                x, y = label_above
                label = (x + 1, y - 1)
            else:
                raise ValueError(f"Invalid label above: {label_above}")

        for col in range(start_col, n_cols, 2):
            grid[row, col] = label

            if col + 2 < n_cols:
                label = next_lhz_label(label, n)

    return grid