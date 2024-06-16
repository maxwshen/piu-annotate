import os
import pickle
import gzip
from dataclasses import dataclass

from .chart import ChartStruct


@dataclass
class ArrowArt:
    arrow_pos: int
    time: float
    limb: str

    def validate(self):
        assert self.arrow_pos in list(range(0, 10))
        assert self.time > 0
        assert self.limb in ['l', 'r', 'e', 'h']


@dataclass
class HoldArt:
    arrow_pos: int
    start_time: float
    end_time: float
    limb: str

    def validate(self):
        assert self.arrow_pos in list(range(0, 10))
        assert self.start_time > 0 and self.end_time > 0
        assert self.limb in ['l', 'r', 'e', 'h']


class ChartJsStruct:
    def __init__(self, arrowarts: list[ArrowArt], holdarts: list[HoldArt]):
        """ Data structure for chart, for javascript visualization use.

            Fields
            - ArrowArts
            - HoldArts
        """
        self.arrowarts = arrowarts
        self.holdarts = holdarts
        pass

    @classmethod
    def from_chartstruct(cs: ChartStruct):
        # return ChartJsStruct(*pcs.get_arrow_hold_arts())
        return
    
    def to_file(self):
        """ Save to file: pkl.gz, for """
        return