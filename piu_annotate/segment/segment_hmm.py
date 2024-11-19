"""
    ChartStruct segmentation and description
"""
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import math
from loguru import logger
import itertools
from tqdm import tqdm

import ruptures as rpt
from ruptures.costs import CostRbf
from ruptures.base import BaseCost

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import nps
from piu_annotate.segment.segment import Section
from piu_annotate.segment.skills import annotate_skills


def featurize(cs: ChartStruct) -> npt.NDArray:
    """ Featurize `cs` into (n_lines, d_ft) np array with floats/ints.
    """
    cs.annotate_time_since_downpress()
    cs.annotate_num_downpresses()
    annotate_skills(cs)

    # featurize
    nps = np.array(1 / cs.df['__time since prev downpress'])
    skill_cols = [
        '__drill',
        '__run',
        '__twist 90',
        '__twist over90',
        '__jump',
        # '__anchor run',
        '__bracket',
        '__staggered bracket',
        '__doublestep',
        # '__side3 singles',
        # '__mid4 doubles',
        '__jack',
        '__footswitch',
    ]
    skill_fts = np.array(cs.df[skill_cols].astype(int))
    x = np.concatenate((nps.reshape(-1, 1), skill_fts), axis = 1)
    # shape: (n_lines, n_fts)

    # convolve skill features with triangular function
    # this makes "sparse" skills like twists to be featurized
    # more similarly to long drill/run sections, which by definition
    # tend to occur in long clusters
    conv_fts = ['__twist 90', '__twist over90', '__bracket', '__staggered bracket',
                '__doublestep', '__jump', '__jack', '__footswitch']
    for conv_ft in conv_fts:
        i = 1 + skill_cols.index(conv_ft)
        x[:, i] = np.convolve(x[:, i], [0, 1, 2, 1, 0], 'same')

    # drop fts that are all constant values
    n_fts = x.shape[1]
    ok_ft_dims = [i for i in range(n_fts) if len(set(x[:, i])) > 1]
    x = x[:, ok_ft_dims]

    # normalize each skill dim to be mean 0, std 1
    normalize = lambda x: (x - np.mean(x, axis = 0)) / (np.std(x, axis = 0))
    x[:, 1:] = normalize(x[:, 1:])

    # exponentially scale normalized nps -- reduces importance of slow/slower sections
    # x[:, 0] = np.power(x[:, 0] / np.mean(x[:, 0]), 3)
    # current issue - does not separate breaks well

    return x


def get_best_segmentation(cs: ChartStruct) -> list[Section]:
    x = featurize(cs)

    times = list(cs.df['Time'])
    chart_time_len = max(times)

    def score_segmentation(changepoints: list[int]) -> float:
        """ Score a list of changepoints, using time lengths """
        # ~ 1 segment per 15 seconds is roughly good
        ideal_num_segments = chart_time_len / 15
        num_cost = (len(changepoints) - ideal_num_segments) ** 2

        def segment_len_cost(start: int, end: int) -> float:
            t = times[end] - times[start]
            min_len = 7
            max_len = 20
            if min_len <= t <= max_len:
                return 0
            elif t < min_len:
                return 10 * np.abs(t - min_len)**4
            elif t > max_len:
                return (t - max_len)**2

        sections = [0] + changepoints[:-1] + [len(cs.df) - 1]
        segment_costs = [segment_len_cost(s, e) for s, e
                         in itertools.pairwise(sections)]
        return -1 * (num_cost + 0.5 * sum(segment_costs))

    # Use jump=5 for faster speed, to find best penalty
    algo = rpt.Pelt(model = 'rbf', jump = 5).fit(x)
    penalties = np.linspace(5, 20, 10)
    segments = [algo.predict(pen = p) for p in tqdm(penalties)]
    scores = [score_segmentation(s) for s in segments]

    best_idx = scores.index(max(scores))
    best_penalty = penalties[best_idx]
    best_segments = segments[best_idx]
    # print(best_segments)

    # Rerun with jump=1 with best penalty, for finetuning changepoints
    algo = rpt.Pelt(model = 'rbf', jump = 1).fit(x)
    j1_changepoints = algo.predict(pen = best_penalty)
    return j1_changepoints


def segment_hmm(cs: ChartStruct) -> list[Section]:
    result = get_best_segmentation(cs)
    print(cs.df['Time'].iloc[result[:-1]])
    import code; code.interact(local=dict(globals(), **locals()))

    return