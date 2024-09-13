from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from piu_annotate.formats import notelines


@dataclass
class AbstractArrowDataPoint:
    pass


@dataclass
class ArrowDataPoint(AbstractArrowDataPoint):
    """ Datapoint representing a single arrow.
        A line can have multiple arrows.
    """
    arrow_pos: int
    time_since_prev_downpress: float
    n_arrows_in_same_line: int
    line_repeats_previous: bool
    line_repeats_next: bool
    singles_or_doubles: str

    def to_array(self) -> NDArray:
        assert self.singles_or_doubles in ['singles', 'doubles']
        sd_to_len = {'singles': 5, 'doubles': 10}
        arrows = [0] * sd_to_len[self.singles_or_doubles]
        assert self.arrow_pos < len(arrows)
        arrows[self.arrow_pos] = 1
        return np.array(arrows + [
            self.time_since_prev_downpress, 
            self.n_arrows_in_same_line,
            int(self.line_repeats_previous),
            int(self.line_repeats_next),
        ])


@dataclass
class ArrowDataPointWithLimbContext(AbstractArrowDataPoint):
    arrow_pos: int
    time_since_prev_downpress: float
    n_arrows_in_same_line: int
    line_repeats_previous: bool
    line_repeats_next: bool
    limb_annot: float
    singles_or_doubles: str

    def to_array(self) -> NDArray:
        assert self.singles_or_doubles in ['singles', 'doubles']
        sd_to_len = {'singles': 5, 'doubles': 10}
        arrows = [0] * sd_to_len[self.singles_or_doubles]
        assert self.arrow_pos < len(arrows)
        arrows[self.arrow_pos] = 1
        return np.array(arrows + [
            self.time_since_prev_downpress, 
            self.n_arrows_in_same_line,
            int(self.line_repeats_previous),
            int(self.line_repeats_next),
            self.limb_annot
        ])


@dataclass
class LimbLabel:
    limb: int   # 0 for left, 1 for right

    @staticmethod
    def from_limb_annot(annot: str):
        assert annot in list('lr')
        return LimbLabel(limb = 0) if annot == 'l' else LimbLabel(limb = 1)

    def to_array(self) -> NDArray:
        return np.array(self.limb)
    
