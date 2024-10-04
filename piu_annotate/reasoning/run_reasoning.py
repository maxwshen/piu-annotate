"""
    Find runs with no jacks, and reason about patterns in runs
"""

from numpy.typing import NDArray
import math
from hackerargs import args
from loguru import logger
from dataclasses import dataclass

from piu_annotate.ml.datapoints import ArrowDataPoint


@dataclass
class LineWithLimb:
    line: str
    limb: str

    def matches(self, adp: ArrowDataPoint, pred_limb: str) -> bool:
        return all([
            self.line == adp.line_with_active_holds,
            self.limb == pred_limb
        ])
    
    def __hash__(self):
        return hash((self.line, self.limb))


singles_line_patterns_to_score = {
    # upper spin
    (LineWithLimb('00010', 'l'),
     LineWithLimb('01000', 'r'),
     LineWithLimb('00100', 'l'),
    ): -10,
    (LineWithLimb('01000', 'r'),
     LineWithLimb('00010', 'l'),
     LineWithLimb('00100', 'r'),
    ): -10,
    # lower spin
    (LineWithLimb('00001', 'l'),
     LineWithLimb('10000', 'r'),
     LineWithLimb('00100', 'l'),
    ): -10,
    (LineWithLimb('10000', 'r'),
     LineWithLimb('00001', 'l'),
     LineWithLimb('00100', 'r'),
    ): -10,
}
doubles_line_patterns_to_score = {
    # middle spin
    (LineWithLimb('0000010000', 'r'),
     LineWithLimb('0000001000', 'l'),
     LineWithLimb('0001000000', 'r'),
     LineWithLimb('0000100000', 'l'),
    ): -10,
    (LineWithLimb('0000100000', 'l'),
     LineWithLimb('0001000000', 'r'),
     LineWithLimb('0000001000', 'l'),
     LineWithLimb('0000010000', 'r'),
    ): -10,
}
for pattern, score in singles_line_patterns_to_score.items():
    dp1 = tuple(LineWithLimb(l.line + '0'*5, l.limb) for l in pattern)
    doubles_line_patterns_to_score[dp1] = score
    dp2 = tuple(LineWithLimb('0'*5 + l.line, l.limb) for l in pattern)
    doubles_line_patterns_to_score[dp2] = score

# merged dict
line_patterns_to_score = singles_line_patterns_to_score | doubles_line_patterns_to_score


def count_pattern_matches(
    pattern: tuple[LineWithLimb], 
    adps: list[ArrowDataPoint], 
    pred_limbs: NDArray
) -> int:
    """ Finds `pattern` in a longer list of `adps` and `pred_limbs`,
        returning number of occurences
    """
    length = len(pattern)
    limb_int_to_str = {0: 'l', 1: 'r'}
    pred_str_limbs = [limb_int_to_str[l] for l in pred_limbs]

    def match(pattern, slice_adps, slice_limbs) -> bool:
        """ Matches `pattern` to `slice_adps` and `slice_limbs` with same
            number of items as lines in pattern
        """
        return all(pline.matches(adp, limb) for pline, adp, limb
                   in zip(pattern, slice_adps, slice_limbs))

    num_matches = 0
    for i in range(len(adps) - length + 1):
        slice_adps = adps[i:i + length]
        slice_limbs = pred_str_limbs[i:i + length]
        if match(pattern, slice_adps, slice_limbs):
            num_matches += 1
    return num_matches


def score_run(adps: list[ArrowDataPoint], pred_limbs: NDArray) -> float:
    """ Scores a run defined by `adps` executed using `pred_limbs`,
        based on patterns in the run.
        Used by Tactician to decide limb annotation for runs
    """
    assert len(adps) == len(pred_limbs)

    total_score = 0
    for pattern, score in line_patterns_to_score.items():
        n_matches = count_pattern_matches(pattern, adps, pred_limbs)
        total_score += score * n_matches
    return total_score


"""
    Find runs
"""
def is_in_run(start_adp: ArrowDataPoint, query_adp: ArrowDataPoint) -> bool:
    """ Whether `query_adp` is in a run with `start_adp`
        A run is a sequence of lines all with 1 downpress, with the same note
        type (1 or 2), with the same time interval between downpresses,
        without jacks.
        If run uses holds, all holds must end before next downpress
        Runs found this way are assumed to be executed by
        alternating limbs on downpresses.

        Time between downpress cannot be too long
        or too short (conflicts with staggered brackets)
    """
    MIN_TIME_SINCE = args.setdefault('reason.run_no_jacks.min_time_since', 1/13)
    MAX_TIME_SINCE = args.setdefault('reason.run_no_jacks.max_time_since', 1/3.3)
    return all([
        math.isclose(query_adp.time_since_prev_downpress,
                        start_adp.time_since_prev_downpress), 
        query_adp.time_since_prev_downpress >= MIN_TIME_SINCE,
        query_adp.time_since_prev_downpress < MAX_TIME_SINCE,
        query_adp.num_downpress_in_line == 1,
        not query_adp.line_repeats_previous_downpress_line,
        query_adp.arrow_symbol == start_adp.arrow_symbol,
        (query_adp.next_line_only_releases_hold_on_this_arrow
         if query_adp.arrow_symbol == '2' else True)
    ])


def is_run_start(start_adp: ArrowDataPoint, query_adp: ArrowDataPoint):
    """ Returns whether `start_adp` can be starting line of run """
    return all([
        start_adp.num_downpress_in_line == 1,
        query_adp.arrow_symbol == start_adp.arrow_symbol,
        (start_adp.next_line_only_releases_hold_on_this_arrow
         if start_adp.arrow_symbol == '2' else True)
    ])


def merge(runs: list[tuple[int]]) -> list[tuple[int]]:
    """ Merge overlapping run sections,
        e.g., combine (10, 15), (14, 19) -> (10, 19),
        which can merge neighboring run sections with different time since downpress;
        for example starting at 8th note rhythm, then 16th note rhythm.
        
        In general, this function may need to be called multiple times
        to merge all possible merge-able runs.
    """
    new_runs = []
    can_merge = lambda ra, rb: ra[1] == rb[0] + 1
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


def find_runs_without_jacks(
    adps: list[ArrowDataPoint], 
    verbose: bool = False
) -> list[tuple[int]]:
    """ Find runs in `adps`, returning a list of (run_start_idx, run_end_idx).
    """
    MIN_RUN_LENGTH = args.setdefault('reason.run_no_jacks.min_run_length', 6)

    runs = []
    curr_run_start_idx = None
    for pc_idx, adp in enumerate(adps):
        if curr_run_start_idx is None:
            curr_run_start_idx = pc_idx
        else:
            if is_in_run(adps[curr_run_start_idx], adp):
                pass
            else:
                run_length = pc_idx - curr_run_start_idx
                if run_length >= MIN_RUN_LENGTH:
                    if is_run_start(adps[curr_run_start_idx - 1], adps[curr_run_start_idx]):
                        runs.append((curr_run_start_idx - 1, pc_idx))
                    else:
                        runs.append((curr_run_start_idx, pc_idx))
                curr_run_start_idx = pc_idx

    num_runs_premerge = len(runs)
    while (merged_runs := merge(runs)) != runs:
        runs = merged_runs
    if verbose:
        logger.debug(f'Found {num_runs_premerge} runs, merged into {len(runs)}: {runs}')
    return runs


