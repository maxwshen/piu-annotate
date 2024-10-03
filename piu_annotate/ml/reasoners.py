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

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines
from piu_annotate.ml import run_reasoning


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

        self.MIN_TIME_SINCE = args.setdefault('reason.run_no_jacks.min_time_since', 1/13)
        self.MAX_TIME_SINCE = args.setdefault('reason.run_no_jacks.max_time_since', 1/3.3)
        self.MIN_RUN_LENGTH = args.setdefault('reason.run_no_jacks.min_run_length', 6)


    def nominate_sections(self):
        runs = self.find_runs_without_jacks()
        return runs
    
    def check(self):
        """ Checks limb reuse pattern on nominated chart sections
            against self.cs Limb Annotation column

            implicit limb pattern -- all lines alternate limbs
        """
        runs = self.nominate_sections()
        limb_annots = self.df['Limb annotation']

        def alternates(limbs):
            return all([
                len(set(limbs[::2])) == 1,
                len(set(limbs[1::2])) == 1,
                limbs[0] != limbs[1]
            ])

        num_violations = 0
        for (start, end) in runs:
            limbs = list(limb_annots[start : end])
            if not alternates(limbs):
                num_violations += 1

                # logger.error(self.cs.source_file)
                # logger.error((start, end))
                # import code; code.interact(local=dict(globals(), **locals()))            
        return num_violations

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
        """ Find runs in self.cs, returning a list of (run_start_idx, run_end_idx).
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
        return runs