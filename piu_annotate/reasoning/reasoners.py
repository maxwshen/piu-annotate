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
from piu_annotate.reasoning import pattern_store


class LimbUse(Enum):
    alternate = 1
    same = 2


class LimbReusePattern:
    def __init__(self, downpress_idxs: list[int], limb_pattern: list[LimbUse]):
        self.downpress_idxs = downpress_idxs
        self.limb_pattern = limb_pattern
        self.validate()

    def __repr__(self):
        return f'{self.downpress_idxs[0]}-{self.downpress_idxs[-1]}'

    def __len__(self) -> int:
        return len(self.downpress_idxs)

    def check(self, downpress_limbs: list[int | str]) -> tuple[bool, any]:
        """ Checks if LimbReusePattern matches `downpress_limbs`.
            Returns OK or not, and optional object
        """
        self.validate()
        limbs = [downpress_limbs[i] for i in self.downpress_idxs]
        pairs = itertools.pairwise(limbs)
        for i, ((la, lb), limbuse) in enumerate(zip(pairs, self.limb_pattern)):
            if any([
                limbuse == LimbUse.alternate and la == lb,
                limbuse == LimbUse.same and la != lb,
            ]):
                return False, (self.downpress_idxs[i], self.downpress_idxs[i + 1])
        return True, None

    def fill_limbs(self, starting_limb: str) -> NDArray:
        """ `starting_limb`: 'left' or 'right'
            Fills limbs from `starting_limb` using self.limb_pattern.
            Returns NDArray
        """
        self.validate()
        assert starting_limb in ['left', 'right']
        limbs = [0] if starting_limb == 'left' else [1]
        for limbuse in self.limb_pattern:
            if limbuse == LimbUse.same:
                limbs.append(limbs[-1])
            else:
                limbs.append(1 - limbs[-1])
        return np.array(limbs)

    def validate(self):
        assert len(self.downpress_idxs) == len(self.limb_pattern) + 1


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
        self.cs.annotate_time_to_next_downpress()
        self.cs.annotate_line_repeats_previous()
        self.cs.annotate_line_repeats_next()
        self.cs.annotate_num_downpresses()
        self.cs.annotate_single_hold_ends_immediately()

        self.downpress_coords = self.cs.get_prediction_coordinates()

        self.MIN_TIME_SINCE = args.setdefault('reason.run_no_jacks.min_time_since', 1/13)
        self.MAX_TIME_SINCE = args.setdefault('reason.run_no_jacks.max_time_since', 1/2.5)
        self.MIN_RUN_LENGTH = args.setdefault('reason.run_no_jacks.min_run_length', 5)

    """
        Core
    """
    def nominate(self) -> list[LimbReusePattern]:
        """ Nominate sections with runs in self.df,
            returning a list of (start_idx, end_idx)
        """
        return self.find_runs()

    def decide_limbs_for_pattern(self, lr_pattern: LimbReusePattern) -> NDArray | None:
        """ Returns NDArray of limbs to use for each
            downpress_idx in `lr_pattern`, or None to abstain
        """
        acs = [self.downpress_coords[i] for i in lr_pattern.downpress_idxs]

        start_left_limbs = lr_pattern.fill_limbs('left')
        start_right_limbs = lr_pattern.fill_limbs('right')

        left_score = pattern_store.score_run(acs, start_left_limbs)
        right_score = pattern_store.score_run(acs, start_right_limbs)

        if left_score > right_score:
            return start_left_limbs
        elif left_score < right_score:
            return start_right_limbs
        return None

    def propose_limbs(self) -> NDArray:
        """ Propose limbs.
            Returns `pred_limbs` array with -1 for unpredicted, 0 for left, 1 for right.
        """
        lr_patterns = self.nominate()

        pred_limbs = np.array([-1] * len(self.downpress_coords))
        abstained_lr_patterns = []
        edited = []
        for lr_pattern in lr_patterns:
            limbs = self.decide_limbs_for_pattern(lr_pattern)
            if limbs is None:
                abstained_lr_patterns.append(lr_pattern)
                continue

            # edit
            for limb, dp_idx in zip(limbs, lr_pattern.downpress_idxs):
                pred_limbs[dp_idx] = limb
            edited.append(lr_pattern)
        
        if self.verbose:
            logger.debug(f'Found {lr_patterns=}')
            logger.debug(f'Edited {edited}')
            edit_coverage = sum(len(lrp) for lrp in edited) / len(self.downpress_coords)
            logger.debug(f'Edited coverage: {edit_coverage:.2%}')
            total_coverage = sum(len(lrp) for lrp in lr_patterns) / len(self.downpress_coords)
            logger.debug(f'Total coverage: {total_coverage:.2%}')

            not_edited_times = []
            for lrp in lr_patterns:
                if lrp not in edited:
                    t1 = self.downpress_idx_to_time(lrp.downpress_idxs[0])
                    t2 = self.downpress_idx_to_time(lrp.downpress_idxs[-1])
                    not_edited_times.append(f'{t1:.2f}-{t2:.2f}')
            logger.debug(f'Not edited: {not_edited_times}')

        return pred_limbs, abstained_lr_patterns

    """
        Checks
    """
    def check(self, breakpoint: bool = False) -> dict[str, any]:
        """ Checks limb reuse pattern on nominated chart sections
            against self.cs Limb Annotation column

            implicit limb pattern -- all lines alternate limbs
        """
        lr_patterns = self.nominate()
        limb_annots = self.limb_annots_at_downpress_idxs()

        num_violations = 0
        time_of_violations = []
        fractions_center = []
        for lrp in lr_patterns:
            ok, pkg = lrp.check(limb_annots)

            if not ok:
                bad_dp_idx1, bad_dp_idx2 = pkg
                num_violations += 1
                bad_time_1 = self.downpress_idx_to_time(bad_dp_idx1)
                bad_time_2 = self.downpress_idx_to_time(bad_dp_idx2)
                time_of_violations.append((bad_time_1, bad_time_2))

                lrp_start = lrp.downpress_idxs[0]
                lrp_end = lrp.downpress_idxs[-1]
                t1 = self.downpress_idx_to_time(lrp_start)
                t2 = self.downpress_idx_to_time(lrp_end)

                acs = self.downpress_coords[lrp_start:lrp_end]
                is_center = lambda arrow_pos: arrow_pos == 2 or arrow_pos == 7
                frac_center = len([ac for ac in acs if is_center(ac.arrow_pos)]) / len(acs)
                fractions_center.append(frac_center)

                if breakpoint:
                    logger.error(self.cs.source_file)
                    logger.error((bad_time_1, bad_time_2))
                    logger.error((t1, t2))
                    logger.error(frac_center)
                    import code; code.interact(local=dict(globals(), **locals())) 
        stats = {
            'Line coverage': sum(len(lrp) for lrp in lr_patterns) / len(self.df),
            'Num violations': num_violations,
            'Time of violations': time_of_violations,
            'Fraction center': fractions_center,
        }
        return stats


    def check_proposals(self, breakpoint: bool = False) -> dict[str, any]:
        lr_patterns = self.nominate()
        gold_limb_annots = self.limb_annots_at_downpress_idxs()

        num_violations = 0
        time_of_violations = []
        for lrp in lr_patterns:
            limbs = self.decide_limbs_for_pattern(lrp)
            if limbs is None:
                continue
            
            limb_int_to_str = {0: 'l', 1: 'r'}
            pred_limbs = [limb_int_to_str[l] for l in limbs]

            # check proposed limbs against gold standard annotations
            gold_limbs = [gold_limb_annots[i] for i in lrp.downpress_idxs]

            if pred_limbs != gold_limbs:
                num_violations += 1
                t1 = self.downpress_idx_to_time(lrp.downpress_idxs[0])
                t2 = self.downpress_idx_to_time(lrp.downpress_idxs[-1])
                time_of_violations.append((t1, t2))

                if breakpoint:
                    logger.error(self.cs.source_file)
                    logger.error((t1, t2))
                    import code; code.interact(local=dict(globals(), **locals())) 

        stats = {
            'Num violations': num_violations,
            'Time of violations': time_of_violations,
        }
        return stats


    """
        Convert between line idxs and downpress idxs
    """
    def line_to_downpress_idx(self, row_idx: int, limb_idx: int):
        for dp_idx, ac in enumerate(self.downpress_coords):
            if ac.row_idx == row_idx and ac.limb_idx == limb_idx:
                return dp_idx
        logger.error(f'Queried bad {row_idx=}, {limb_idx=}; not found')
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
        Find runs
    """
    def is_in_run(self, start_row: pd.Series, query_row: pd.Series) -> bool:
        """ Accept jacks, unless they are on center panel, this could be footswitch
        """
        start_line = start_row['Line with active holds']
        query_line = query_row['Line with active holds']
        jack_on_center_panel = all([
            query_row['__line repeats previous downpress line'],
            notelines.has_center_arrow(query_line),
        ])
        start_ok_downpress = any([
            '1' in start_line,
            start_row['__single hold ends immediately']
        ])
        query_ok_downpress = any([
            '1' in query_line,
            query_row['__single hold ends immediately']
        ])
        return all([
            start_row['__time since prev downpress'] >= self.MIN_TIME_SINCE,
            query_row['__time since prev downpress'] >= self.MIN_TIME_SINCE,
            query_row['__time since prev downpress'] < self.MAX_TIME_SINCE,
            query_row['__time to next downpress'] >= self.MIN_TIME_SINCE,
            notelines.num_downpress(start_line) == 1,
            notelines.num_downpress(query_line) == 1,
            '4' not in start_line,
            '3' not in start_line,
            '4' not in query_line,
            '3' not in query_line,
            not jack_on_center_panel,
            start_ok_downpress,
            query_ok_downpress,
        ])
    
    def is_run_start(self, start_row: pd.Series, query_row: pd.Series):
        """ Returns whether `start_row` can be starting line of run """
        start_line = start_row['Line with active holds']
        query_line = query_row['Line with active holds']
        jack_on_center_panel = all([
            query_row['__line repeats previous downpress line'],
            notelines.has_center_arrow(query_line),
        ])
        start_ok_downpress = any([
            '1' in start_line,
            start_row['__single hold ends immediately']
        ])
        # allow start_row to be very first row; 
        start_ok_time_since = any([
            start_row['__time since prev downpress'] >= self.MIN_TIME_SINCE,
            start_row['__time since prev downpress'] == -1,
        ])
        return all([
            start_ok_time_since,
            query_row['__time since prev downpress'] >= self.MIN_TIME_SINCE,
            query_row['__time since prev downpress'] < self.MAX_TIME_SINCE,
            notelines.num_downpress(start_line) == 1,
            start_ok_downpress,
            '4' not in start_line,
            '3' not in start_line,
            not jack_on_center_panel,
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

    def find_runs(self) -> list[LimbReusePattern]:
        """ Find runs in self.cs, and execute them by alternating on different
            arrows, and repeating limb on same arrows.
        """
        runs = []
        df = self.df
        curr_run = None
        downpress_df = df[df['__num downpresses'] > 0]
        for row_idx, row in downpress_df.iterrows():
            if curr_run is None:
                curr_run = [row_idx]
            else:
                if self.is_in_run(df.iloc[curr_run[0]], row):
                    pass
                    curr_run.append(row_idx)
                else:
                    run_length = len(curr_run)
                    if run_length >= self.MIN_RUN_LENGTH:
                        if curr_run[0] > 0 and self.is_run_start(
                            df.iloc[curr_run[0] - 1], 
                            df.iloc[curr_run[0]]
                        ):
                            curr_run.insert(0, curr_run[0] - 1)
                        runs.append(curr_run)
                    curr_run = [row_idx]

        # if self.verbose:
            # logger.debug(f'Found {len(runs)}: {runs}')
        while (merged_runs := self.merge(runs)) != runs:
            runs = merged_runs
        # if self.verbose:
            # logger.debug(f'Found {len(runs)} candidate runs: {runs}')

        # get limb pattern
        limb_patterns = []
        repeats_next_downpress = self.cs.df['__line repeats next downpress line']
        rnd_map = {True: LimbUse.same, False: LimbUse.alternate}
        for run in runs:
            lp = [rnd_map[x] for x in repeats_next_downpress.iloc[run[:-1]]]
            limb_patterns.append(lp)

        # convert to downpress_idxs
        dp_runs = [[self.line_to_downpress_idx(r, 0) for r in run] for run in runs]

        # convert to LimbReusePatterns
        lr_patterns: list[LimbReusePattern] = []
        for dp_run, limb_pattern in zip(dp_runs, limb_patterns):
            lrp = LimbReusePattern(
                downpress_idxs = dp_run,
                limb_pattern = limb_pattern
            )
            lr_patterns.append(lrp)

        if self.verbose:
            logger.debug(f'Found {len(lr_patterns)} runs: {lr_patterns=}')
            get_time = lambda dp_idx: self.downpress_idx_to_time(dp_idx)
            times = [(f'{get_time(lrp.downpress_idxs[0]):.2f}',
                      f'{get_time(lrp.downpress_idxs[-1]):.2f}'
                         ) for lrp in lr_patterns]
            logger.debug(f'Covers times: {times}')

        return lr_patterns