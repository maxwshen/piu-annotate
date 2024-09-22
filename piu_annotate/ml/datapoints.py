from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from loguru import logger

from piu_annotate.formats import notelines


@dataclass
class AbstractArrowDataPoint:
    pass


@dataclass
class ArrowDataPoint(AbstractArrowDataPoint):
    """ Datapoint representing a single arrow.
        A line can have multiple arrows.
        This should not use any limb information for any arrow.
    """
    arrow_pos: int
    is_hold: bool
    active_hold_idxs: list[int]
    same_line_as_next_datapoint: bool
    time_since_last_same_arrow_use: float
    time_since_prev_downpress: float
    n_arrows_in_same_line: int
    line_is_bracketable: bool
    line_repeats_previous: bool
    line_repeats_next: bool
    singles_or_doubles: str

    def to_array_onehot(self) -> NDArray:
        """ Featurize into a one-hot array """
        assert self.singles_or_doubles in ['singles', 'doubles']
        sd_to_len = {'singles': 5, 'doubles': 10}
        arrows = [0] * sd_to_len[self.singles_or_doubles]
        assert self.arrow_pos < len(arrows)
        arrows[self.arrow_pos] = 1

        hold_arrows = [0] * sd_to_len[self.singles_or_doubles]
        for idx in self.active_hold_idxs:
            hold_arrows[idx] = 1

        fts = [
            int(self.is_hold),
            int(len(self.active_hold_idxs) > 0),
            int(self.same_line_as_next_datapoint),
            self.time_since_last_same_arrow_use,
            self.time_since_prev_downpress, 
            self.n_arrows_in_same_line,
            int(self.line_is_bracketable),
            int(self.line_repeats_previous),
            int(self.line_repeats_next),
        ]
        return np.concatenate([arrows, hold_arrows, np.array(fts)])

    def to_array_categorical(self) -> NDArray:
        """ Featurize, using int for categorical features """
        assert self.singles_or_doubles in ['singles', 'doubles']
        sd_to_len = {'singles': 5, 'doubles': 10}

        hold_arrows = [0] * sd_to_len[self.singles_or_doubles]
        for idx in self.active_hold_idxs:
            hold_arrows[idx] = 1

        fts = [
            self.arrow_pos,
            int(self.is_hold),
            int(len(self.active_hold_idxs) > 0),
            int(self.same_line_as_next_datapoint),
            self.time_since_last_same_arrow_use,
            self.time_since_prev_downpress, 
            self.n_arrows_in_same_line,
            int(self.line_is_bracketable),
            int(self.line_repeats_previous),
            int(self.line_repeats_next),
        ]
        return np.concatenate([np.array(fts), hold_arrows])

    def get_feature_names_categorical(self) -> list[str]:
        """ Must be aligned with categorical array """
        sord = self.singles_or_doubles
        length = 5 if sord == 'singles' else 10
        hold_arrow_names = [f'active hold in pos {idx}' for idx in range(length)]
        ft_names = [
            'cat.arrow_pos',
            'is_hold',
            'has_active_hold',
            'in_same_line_as_next_datapoint',
            'time_since_last_same_arrow_use',
            'time_since_prev_downpress',
            'num_arrows_in_same_line',
            'line_is_bracketable',
            'line_repeats_previous',
            'line_repeats_next'
        ] + hold_arrow_names
        assert len(ft_names) == len(self.to_array_categorical())
        return ft_names


@dataclass
class LimbLabel:
    limb: int   # 0 for left, 1 for right

    @staticmethod
    def from_limb_annot(annot: str):
        mapper = {'l': 0, 'r': 1, 'h': 0}
        return LimbLabel(limb = mapper[annot])

    def to_array(self) -> NDArray:
        return np.array(self.limb)
    
