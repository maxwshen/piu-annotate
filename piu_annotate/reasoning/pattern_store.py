from numpy.typing import NDArray
import math
from hackerargs import args
from loguru import logger
from dataclasses import dataclass

from piu_annotate.formats.chart import ArrowCoordinate


@dataclass
class LineWithLimb:
    line: str
    limb: str

    def matches(self, ac: ArrowCoordinate, pred_limb: str) -> bool:
        return all([
            self.line == ac.line_with_active_holds,
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
    # side pattern
    (LineWithLimb('00001', 'r'),
     LineWithLimb('00010', 'l'),
     LineWithLimb('00100', 'r'),
     LineWithLimb('00001', 'l'),
    ): -20,
    (LineWithLimb('10000', 'l'),
     LineWithLimb('01000', 'r'),
     LineWithLimb('00100', 'l'),
     LineWithLimb('10000', 'r'),
    ): -20,
    # both sides
    (LineWithLimb('00001', 'l'),
     LineWithLimb('01000', 'r'),
     LineWithLimb('00100', 'l'),
     LineWithLimb('10000', 'r'),
    ): -20,
    (LineWithLimb('00010', 'l'),
     LineWithLimb('01000', 'r'),
     LineWithLimb('00100', 'l'),
     LineWithLimb('10000', 'r'),
    ): -20,
    (LineWithLimb('10000', 'r'),
     LineWithLimb('00010', 'l'),
     LineWithLimb('00100', 'r'),
     LineWithLimb('00001', 'l'),
    ): -20,
    (LineWithLimb('01000', 'r'),
     LineWithLimb('00010', 'l'),
     LineWithLimb('00100', 'r'),
     LineWithLimb('00001', 'l'),
    ): -20,
    (LineWithLimb('00001', 'l'),
     LineWithLimb('00100', 'r'),
     LineWithLimb('00010', 'l'),
     LineWithLimb('01000', 'r'),
    ): -20,
    (LineWithLimb('10000', 'r'),
     LineWithLimb('00100', 'l'),
     LineWithLimb('01000', 'r'),
     LineWithLimb('00010', 'l'),
    ): -20,
}
doubles_line_patterns_to_score = {
    # middle spin
    (LineWithLimb('0000010000', 'r'),
     LineWithLimb('0000001000', 'l'),
     LineWithLimb('0001000000', 'r'),
     LineWithLimb('0000100000', 'l'),
    ): -20,
    (LineWithLimb('0000100000', 'l'),
     LineWithLimb('0001000000', 'r'),
     LineWithLimb('0000001000', 'l'),
     LineWithLimb('0000010000', 'r'),
    ): -20,
    (LineWithLimb('0000001000', 'l'),
     LineWithLimb('0001000000', 'r'),
     LineWithLimb('0000010000', 'l'),
     LineWithLimb('0000100000', 'r'),
    ): -20,
    (LineWithLimb('0001000000', 'r'),
     LineWithLimb('0000001000', 'l'),
     LineWithLimb('0000100000', 'r'),
     LineWithLimb('0000010000', 'l'),
    ): -20,
    # mid 6 sides
    (LineWithLimb('0000000100', 'l'),
     LineWithLimb('0000001000', 'r'),
     LineWithLimb('0000010000', 'l'),
     LineWithLimb('0000100000', 'r'),
    ): -20,
    (LineWithLimb('0010000000', 'r'),
     LineWithLimb('0001000000', 'l'),
     LineWithLimb('0000100000', 'r'),
     LineWithLimb('0000010000', 'l'),
    ): -20,
    (LineWithLimb('0000100000', 'r'),
     LineWithLimb('0000010000', 'l'),
     LineWithLimb('0000001000', 'r'),
     LineWithLimb('0000000100', 'l'),
    ): -20,
     (LineWithLimb('0000010000', 'l'),
     LineWithLimb('0000100000', 'r'),
     LineWithLimb('0001000000', 'l'),
     LineWithLimb('0010000000', 'r'),
    ): -20,
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
    acs: list[ArrowCoordinate], 
    pred_limbs: NDArray
) -> int:
    """ Finds `pattern` in a longer list of `acs` and `pred_limbs`,
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
    for i in range(len(acs) - length + 1):
        slice_acs = acs[i:i + length]
        slice_limbs = pred_str_limbs[i:i + length]
        if match(pattern, slice_acs, slice_limbs):
            num_matches += 1
    return num_matches


def score_run(acs: list[ArrowCoordinate], pred_limbs: NDArray) -> float:
    """ Scores a run defined by `adps` executed using `pred_limbs`,
        based on patterns in the run.
        Used by PatternReasoner to decide limb annotation for runs
    """
    assert len(acs) == len(pred_limbs)

    total_score = 0
    for pattern, score in line_patterns_to_score.items():
        n_matches = count_pattern_matches(pattern, acs, pred_limbs)
        total_score += score * n_matches
    return total_score