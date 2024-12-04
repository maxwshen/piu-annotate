import numpy as np
import itertools

# panel index -> (x, y) coordinate
pos = {
    0: (236, 535),
    1: (236, 313),
    2: (419, 419),
    3: (603, 313),
    4: (603, 535),
    5: (1111, 535),
    6: (1111, 313),
    7: (1296, 419),
    8: (1480, 313),
    9: (1480, 535),
}
pos = {k: np.array(v) for k, v in pos.items()}


def line_to_xy(line: str) -> np.ndarray:
    """ Converts a line into (x, y) coordinate.
        Assumes only one limb is used for all (1) in line.
        If multiple (1) are present, indicating bracket, return mean foot position
        among downpressed arrows.
    """
    poss = [pos[i] for i, symbol in enumerate(line) if symbol == '1']
    if len(poss) == 1:
        return poss[0]
    assert len(poss) <= 2, 'This case is not supported'
    return np.mean(poss, axis = 0)


def calc_travel(run_lines: list[str]) -> list[float]:
    """ Calculates distance traveled by each foot to hit each arrow.
        Assumes that lines are from run, so that each line alternates limbs.

        Lines are assumed to only have (1), but can have multiple (1) = bracket.
    """
    if len(run_lines) <= 2:
        return []

    # reduce brackets in lines to average foot position
    xys = [line_to_xy(l) for l in run_lines]

    # split by every other line
    # each group is assumed to use the same limb
    evens = [xys[i] for i in range(0, len(xys), 2)]
    odds = [xys[i] for i in range(1, len(xys), 2)]

    dists = []
    for lines_group in [evens, odds]:
        for point1, point2 in itertools.pairwise(lines_group):
            dist = np.linalg.norm(point1 - point2) / 250
            dists.append(dist)

    return dists