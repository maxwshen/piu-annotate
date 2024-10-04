"""
    Pattern reasoning
"""
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import pandas as pd
from numpy.typing import NDArray
import itertools
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines
from piu_annotate.ml import run_reasoning

class LimbUse(Enum):
    alternate = 1
    same = 2


@dataclass
class LimbReusePattern:
    downpress_idxs: list[int]
    limb_pattern: list[LimbUse]

    @staticmethod
    def from_run(start_dp_idx: int, end_dp_idx: int):
        length = end_dp_idx - start_dp_idx
        return LimbReusePattern(
            list(range(start_dp_idx, end_dp_idx)),
            [LimbUse.alternate] * (length - 1)
        )

    def __len__(self) -> int:
        return len(self.downpress_idxs)

    def check(self, downpress_limbs: list[int | str]) -> tuple[bool, any]:
        """ Checks if LimbReusePattern matches `downpress_limbs`.
            Returns OK or not, and optional object
        """
        limbs = [downpress_limbs[i] for i in self.downpress_idxs]
        pairs = itertools.pairwise(limbs)
        for i, ((la, lb), limbuse) in enumerate(zip(pairs, self.limb_pattern)):
            if any([
                limbuse == LimbUse.alternate and la == lb,
                limbuse == LimbUse.same and la != lb,
            ]):
                return False, (self.downpress_idxs[i], self.downpress_idxs[i + 1])
        return True, None


class PatternReasoner:
    def __init__(self, cs: ChartStruct, verbose: bool = False):
        """ A PatternReasoner annotates limbs by:
            1. Nominate chart sections with limb reuse patterns
                (alternate or same limb)
            2. Score specific limb sequences compatible with limb reuse pattern

            Uses `cs` as primary data representation
        """
        self.cs = cs
        self.df = cs.df
        self.verbose = verbose

        self.cs.annotate_time_since_downpress()
        self.cs.annotate_line_repeats_previous()
        self.cs.annotate_line_repeats_next()

        self.downpress_coords = self.cs.get_prediction_coordinates()

        self.MIN_TIME_SINCE = args.setdefault('reason.run_no_jacks.min_time_since', 1/13)
        self.MAX_TIME_SINCE = args.setdefault('reason.run_no_jacks.max_time_since', 1/3.3)
        self.MIN_RUN_LENGTH = args.setdefault('reason.run_no_jacks.min_run_length', 5)

    """
        Convert between line idxs and downpress idxs
    """
    def line_to_downpress_idx(self, row_idx: int, limb_idx: int):
        for dp_idx, ac in enumerate(self.downpress_coords):
            if ac.row_idx == row_idx and ac.limb_idx == limb_idx:
                return dp_idx
        assert False

    def limb_annots_at_downpress_idxs(self) -> list[str]:
        """ Get elements from Limb annotation column at downpress idxs """
        las = []
        limb_annots = list(self.cs.df['Limb annotation'])
        for ac in self.downpress_coords:
            limbs = limb_annots[ac.row_idx]
            if limbs != '':
                las.append(limbs[ac.limb_idx])
            else:
                las.append('?')
        return las

    def downpress_idx_to_time(self, dp_idx: int) -> float:
        row_idx = self.downpress_coords[dp_idx].row_idx
        return float(self.cs.df.iloc[row_idx]['Time'])

    """
    """
    def nominate(self) -> list[LimbReusePattern]:
        """ Nominate sections with runs in self.df,
            returning a list of (start_idx, end_idx)
        """
        runs = self.find_runs_without_jacks()
        return [LimbReusePattern.from_run(run[0], run[1]) for run in runs]

    def check(self, breakpoint: bool = False) -> dict[str, any]:
        """ Checks limb reuse pattern on nominated chart sections
            against self.cs Limb Annotation column

            implicit limb pattern -- all lines alternate limbs
        """
        lr_patterns = self.nominate()
        limb_annots = self.limb_annots_at_downpress_idxs()

        num_violations = 0
        time_of_violations = []
        for lrp in lr_patterns:
            ok, pkg = lrp.check(limb_annots)

            if not ok:
                bad_dp_idx1, bad_dp_idx2 = pkg
                num_violations += 1
                bad_time_1 = self.downpress_idx_to_time(bad_dp_idx1)
                bad_time_2 = self.downpress_idx_to_time(bad_dp_idx2)
                time_of_violations.append((bad_time_1, bad_time_2))

                if breakpoint:
                    logger.error(self.cs.source_file)
                    logger.error((bad_time_1, bad_time_2))
                    import code; code.interact(local=dict(globals(), **locals())) 
        stats = {
            'Line coverage': sum(len(lrp) for lrp in lr_patterns) / len(self.df),
            'Num violations': num_violations,
            'Time of violations': time_of_violations,
        }
        return stats

    """
        Find runs
    """
    def is_in_run(self, start_row: pd.Series, query_row: pd.Series) -> bool:
        start_line = start_row['Line with active holds']
        query_line = query_row['Line with active holds']
        return all([
            math.isclose(start_row['__time since prev downpress'],
                         query_row['__time since prev downpress']), 
            query_row['__time since prev downpress'] >= self.MIN_TIME_SINCE,
            query_row['__time since prev downpress'] < self.MAX_TIME_SINCE,
            notelines.num_downpress(start_line) == 1,
            notelines.num_downpress(query_line) == 1,
            '4' not in start_line,
            '3' not in start_line,
            '4' not in query_line,
            '3' not in query_line,
            not query_row['__line repeats previous downpress line'],
            '1' in start_line,
            '1' in query_line,
        ])
    
    def is_run_start(self, start_row: pd.Series, query_row: pd.Series):
        """ Returns whether `start_row` can be starting line of run """
        start_line = start_row['Line with active holds']
        return all([
            notelines.num_downpress(start_line) == 1,
            '1' in start_line,
            '4' not in start_line,
            '3' not in start_line,
            not query_row['__line repeats previous downpress line'],
        ])
    
    def merge(self, runs: list[tuple[int]]) -> list[tuple[int]]:
        """ Merge overlapping run sections,
            e.g., combine (10, 15), (14, 19) -> (10, 19),
            which can merge neighboring run sections with different time since downpress;
            for example starting at 8th note rhythm, then 16th note rhythm.
            
            In general, this function may need to be called multiple times
            to merge all possible merge-able runs.
        """
        new_runs = []
        def can_merge(run1, run2):
            if run1[1] == run2[0] + 1:
                return True
            return False

        idx = 0
        while idx < len(runs):
            run = runs[idx]
            if idx + 1 == len(runs):
                new_runs.append(run)
                break
            next_run = runs[idx + 1]
            if can_merge(run, next_run):
                new_runs.append((run[0], next_run[1]))
                idx += 1
            else:
                new_runs.append(run)
            idx += 1
        return new_runs

    def find_runs_without_jacks(self) -> list[tuple[int]]:
        """ Find runs in self.cs, returning a list of
            (start_downpress_idx, end_downpress_idx).
        """
        runs = []
        df = self.df
        curr_run_start_idx = None
        for row_idx, row in df.iterrows():
            if curr_run_start_idx is None:
                curr_run_start_idx = row_idx
            else:
                if self.is_in_run(df.iloc[curr_run_start_idx], row):
                    pass
                else:
                    run_length = row_idx - curr_run_start_idx
                    if run_length >= self.MIN_RUN_LENGTH:
                        if self.is_run_start(df.iloc[curr_run_start_idx - 1], df.iloc[curr_run_start_idx]):
                            runs.append((curr_run_start_idx - 1, row_idx))
                        else:
                            runs.append((curr_run_start_idx, row_idx))
                    curr_run_start_idx = row_idx

        if self.verbose:
            logger.debug(f'Found {len(runs)}: {runs}')
        while (merged_runs := self.merge(runs)) != runs:
            runs = merged_runs
        if self.verbose:
            logger.debug(f'Merged into {len(runs)}: {runs}')

        # convert to downpress_idxs
        dp_runs = [(self.line_to_downpress_idx(run[0], 0), 
                    self.line_to_downpress_idx(run[1], 0)) for run in runs]
        return dp_runs