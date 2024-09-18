from __future__ import annotations
import os
from pathlib import Path
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from loguru import logger
import math

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

    @staticmethod
    def from_tuple(tpl: tuple[int, float, str]):
        return ArrowArt(*tpl)

    def matches(self, other: ArrowArt, with_limb_annot: bool) -> bool:
        return all([
            self.arrow_pos == other.arrow_pos,
            math.isclose(self.time, other.time),
            self.limb == other.limb if with_limb_annot else True,
        ])


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

    @staticmethod
    def from_tuple(tpl: tuple[int, float, float, str]):
        return HoldArt(*tpl)

    def matches(self, other: HoldArt, with_limb_annot: bool) -> bool:
        return all([
            self.arrow_pos == other.arrow_pos,
            math.isclose(self.start_time, other.start_time),
            math.isclose(self.end_time, other.end_time),
            self.limb == other.limb if with_limb_annot else True,
        ])


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

    def matches(self, other: ChartJsStruct, with_limb_annot: bool = False) -> bool:
        if not len(self.arrow_arts) == len(other.arrow_arts):
            return False
        if not len(self.hold_arts) == len(other.hold_arts):
            return False
        aas_match = [aa.matches(other_aa, with_limb_annot)
                     for aa, other_aa in zip(self.arrow_arts, other.arrow_arts)]
        has_match = [ha.matches(other_ha, with_limb_annot)
                     for ha, other_ha in zip(self.hold_arts, other.hold_arts)]
        is_match = all(aas_match) and all(has_match)
        if not is_match:
            pass
            # print([i for i, flag in enumerate(aas_match) if not flag])
            # print([i for i, flag in enumerate(has_match) if not flag])
            # import code; code.interact(local=dict(globals(), **locals()))
        return is_match

    @staticmethod
    def from_chartstruct(cs: ChartStruct):
        return ChartJsStruct(*cs.get_arrow_hold_arts())

    @staticmethod
    def from_json(json_file: str):
        with open(json_file) as f:
            json_struct = json.load(f)
        arrow_arts = [ArrowArt.from_tuple(t) for t in json_struct[0]]
        hold_arts = [HoldArt.from_tuple(t) for t in json_struct[1]]
        return ChartJsStruct(arrow_arts, hold_arts)

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