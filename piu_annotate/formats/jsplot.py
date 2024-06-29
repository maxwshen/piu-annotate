from __future__ import annotations
import os
from pathlib import Path
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piu_annotate.formats.chart import ChartStruct


@dataclass
class ArrowArt:
    arrow_pos: int
    time: float
    limb: str

    def validate(self):
        assert self.arrow_pos in list(range(0, 10))
        assert self.time > 0
        assert self.limb in set(list('lreh?'))

    def to_tuple(self) -> list:
        return (self.arrow_pos, self.time, self.limb)

@dataclass
class HoldArt:
    arrow_pos: int
    start_time: float
    end_time: float
    limb: str

    def validate(self):
        assert self.arrow_pos in list(range(0, 10))
        assert self.start_time > 0 and self.end_time > 0
        assert self.limb in set(list('lreh?'))

    def to_tuple(self) -> list:
        return (self.arrow_pos, self.start_time, self.end_time, self.limb)


class ChartJsStruct:
    def __init__(self, arrow_arts: list[ArrowArt], hold_arts: list[HoldArt]):
        """ Data structure for chart, for javascript visualization use.

            Fields
            - ArrowArts
            - HoldArts
        """
        self.arrow_arts = arrow_arts
        self.hold_arts = hold_arts
        self.json_struct = self.get_json_struct()
        pass

    @staticmethod
    def from_chartstruct(cs: ChartStruct):
        return ChartJsStruct(*cs.get_arrow_hold_arts())
    
    def get_json_struct(self):
        """ Get representation in terms of lists and basic objects.
            Returns a tuple, containing:
            - list of arrow art tuples
            - list of hold art tuples
        """
        return (
            [art.to_tuple() for art in self.arrow_arts],
            [art.to_tuple() for art in self.hold_arts],
        )

    def to_json(self, filename: str):
        """ Save to json file """
        Path(os.path.dirname(filename)).mkdir(parents = True, exist_ok = True)
        with open(filename, 'w') as f:
            json.dump(self.json_struct, f)
        return