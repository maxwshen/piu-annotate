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
from piu_annotate.segment.skills import annotate_skills


@dataclass
class Section:
    start: int
    end: int
    start_time: float
    end_time: float

    def time_length(self) -> float:
        return self.end_time - self.start_time

    def __lt__(self, other) -> bool:
        return self.start_time < other.start_time

    def to_tuple(self) -> tuple:
        return (self.start_time, self.end_time)

    def __contains__(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time


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


class Segmenter:
    def __init__(self, cs: ChartStruct, debug: bool = False):
        self.cs = cs
        self.debug = debug

        self.times = list(cs.df['Time'])
        self.chart_time_len = max(self.times)
        self.x = featurize(cs)

    """
        Scoring and costs
    """
    def segment_cost(self, start: int, end: int) -> float:
        t = self.times[end] - self.times[start]
        min_len = 7
        max_len = 20
        if min_len <= t <= max_len:
            return 0
        elif t < min_len:
            return 10 * np.abs(t - min_len)**4
        elif t > max_len:
            return (t - max_len)**2

    def score(self, changepoints: list[int]) -> float:
        """ Score a list of changepoints, using time lengths.
            Penalizes deviation from target num. changepoints,
            and segments that are too short or too long.
            Places heavier penalty on short sections, so errs 
            towards longer segments.
        """
        # ~ 1 segment per 15 seconds is roughly good
        ideal_num_segments = self.chart_time_len / 15
        num_cost = (len(changepoints) - ideal_num_segments) ** 2

        sections = [0] + changepoints[:-1] + [len(self.cs.df) - 1]
        segment_costs = [self.segment_cost(s, e) for s, e
                         in itertools.pairwise(sections)]
        return -1 * (num_cost + 0.5 * sum(segment_costs))

    """
        Segmentation
    """
    def segment(self) -> list[Section]:
        # perform initial segmentation
        algo = rpt.KernelCPD(kernel = 'rbf').fit(self.x)
        penalties = np.linspace(5, 20, 15)
        segments = [algo.predict(pen = p) for p in penalties]
        scores = [self.score(s) for s in segments]
        best_idx = scores.index(max(scores))
        best_segments = segments[best_idx]

        # convert changepoints to list of sections
        best_segments.insert(0, 0)
        # final element from ruptures is = len(df), so subtract 1 to index last element
        best_segments[-1] -= 1
        sections = [Section(i, j, self.times[i], self.times[j])
                    for i, j in itertools.pairwise(best_segments)]

        # try splitting long sections
        new_sections = []
        for section in sections:
            if section.time_length() >= 24:
                new_sections += self.split(section)
            else:
                new_sections.append(section)

        if self.debug:
            print(self.cs.df['Time'].iloc[best_segments[:-1]])
            for s in new_sections:
                print(s)
            print('Entering interactive mode for debugging ...')
            import code; code.interact(local=dict(globals(), **locals()))

        return new_sections

    def split(self, sec: Section) -> list[Section]:
        """ Splits section into two if that improves score.
        """
        start, end = sec.start, sec.end
        base_cost = self.segment_cost(start, end)
        n_lines = sec.end - sec.start

        algo = rpt.KernelCPD(kernel = 'rbf', min_size = n_lines // 4).fit(self.x[start : end])
        changepoint = sec.start + algo.predict(n_bkps = 1)[0]
        split_cost = sum([
            self.segment_cost(start, changepoint),
            self.segment_cost(changepoint, end)
        ])

        if split_cost <= base_cost:
            return [
                Section(start, changepoint, self.times[start], self.times[changepoint]),
                Section(changepoint, end, self.times[changepoint], self.times[end])
            ]
        else:
            return [sec]


def segmentation(cs: ChartStruct, debug: bool = False) -> list[Section]:
    segmenter = Segmenter(cs, debug)
    result = segmenter.segment()
    # print(result)
    # print(cs.df['Time'].iloc[result[:-1]])
    # import code; code.interact(local=dict(globals(), **locals()))

    return result