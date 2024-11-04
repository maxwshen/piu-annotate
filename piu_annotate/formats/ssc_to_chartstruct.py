"""
    Code for converting .ssc into chartstruct format, parsing BPM change
    info to annotate each `line` with time and beat
"""
import numpy as np
import copy
from fractions import Fraction
import pandas as pd
from tqdm import tqdm
import math
from loguru import logger
from collections import defaultdict
import itertools
from dataclasses import dataclass

from .sscfile import StepchartSSC
from . import notelines

BEATS_PER_MEASURE = 4


@dataclass
class HoldTick:
    start_time: float
    end_time: float
    ticks: float

    def __repr__(self) -> str:
        return f'{self.to_tuple()}'

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.start_time, self.end_time, self.ticks)

    def hold_length(self) -> float:
        return self.end_time - self.start_time


def merge_holdticks(holdticks: list[HoldTick]) -> list[HoldTick]:
    """ Merge consecutive holdticks that start/end at same time,
        and one hold is short
    """
    if len(holdticks) == 0:
        return holdticks

    def can_merge(ht1: HoldTick, ht2: HoldTick) -> bool:
        LENGTH_THRESHOLD = 0.05 # seconds
        short = any([
            ht1.hold_length() <= LENGTH_THRESHOLD,
            ht2.hold_length() <= LENGTH_THRESHOLD
        ])
        start_end = math.isclose(ht1.end_time, ht2.start_time)
        return short and start_end

    new_holdticks = [holdticks[0]]
    for i in range(1, len(holdticks)):
        ht = holdticks[i]
        q = new_holdticks[-1]

        if can_merge(q, ht):
            merged = HoldTick(q.start_time, ht.end_time, q.ticks + ht.ticks)
            new_holdticks[-1] = merged
        else:
            new_holdticks.append(ht)
    return new_holdticks


def edit_string(s: str, idx: int, c: chr) -> str:
    return s[:idx] + c + s[idx + 1:]


def stepchart_ssc_to_chartstruct(
    stepchart: StepchartSSC,
    debug: bool = False,
) -> tuple[pd.DataFrame | None, list[tuple], str]:
    """ Builds df to create ChartStruct object, converting .ssc fields
        like BPMS, WARPS, DELAYS, STOPS, FAKES into time/beat stamps for lines.

        Output df has one row per "line" and 
        cols = ['Beat', 'Time', 'Line', 'Line with active holds', 'Limb annotation'].

        Output
        ------
        result: pd.DataFrame | None
            Returns None if failed
        List of HoldTick info: list[tuple]
        message: str
            E.g., failure message
    """
    try:
        b2l = BeatToLines(stepchart)
    except Exception as e:
        error_message = str(e)
        return None, f'Error making BeatToLines: {error_message}'

    warps = BeatToValueDict.from_string(stepchart.get('WARPS', ''))
    beat_to_bpm = BeatToValueDict.from_string(stepchart.get('BPMS', ''))
    stops = BeatToValueDict.from_string(stepchart.get('STOPS', ''))
    delays = BeatToValueDict.from_string(stepchart.get('DELAYS', ''))
    fakes = BeatToValueDict.from_string(stepchart.get('FAKES', ''))
    holdticks = BeatToValueDict.from_string(stepchart.get('TICKCOUNTS', ''))
    beat_to_lines = b2l.beat_to_lines

    # aggregate all beats where anything happens
    all_beats = list(beat_to_lines.keys())
    for bd in [warps, beat_to_bpm, stops, delays, fakes, holdticks]:
        all_beats += bd.get_event_times()
    beats = sorted(list(set(all_beats)))

    in_warp = lambda beat: warps.beat_in_any_range(beat, inclusive_end = False)
    in_fake = lambda beat: fakes.beat_in_any_range(beat, inclusive_end = False)
    in_fake_or_warp = lambda beat: in_warp(beat) or in_fake(beat)

    if debug:
        logger.debug(f'In debug mode in ssc to chartstruct - inspect beats, fakes, etc.')
        import code; code.interact(local=dict(globals(), **locals()))

    # setup initial conditions
    beat = 0
    time = 0
    bpm: float = beat_to_bpm[beat]
    hold_ticks_per_beat: float = holdticks.get(beat, 4) / BEATS_PER_MEASURE

    empty_line = b2l.get_empty_line()
    active_holds = set()
    hold_tick_list: list[HoldTick] = []
    curr_hold_tick = None
    num_bad_lines = 0
    dd = defaultdict(list)
    for beat_idx, beat in enumerate(beats):
        """ Iterate over beats where things happen, incrementing time based on bpm.
            Process BPM changes by beat. Track active holds with 4.
            Fake notes exist but are not judged, so we do not include here.
            Note that holds can be split into fake and real sections.
            In warps, time does not increment, and notes are fake.
        """
        next_beat = beats[beat_idx + 1] if beat_idx < len(beats) - 1 else max(beats) + 1
        line = beat_to_lines.get(beat, empty_line)
        comment = ''
        
        # update bpm
        bpm = beat_to_bpm.get(beat, bpm)
        hold_ticks_per_beat = holdticks.get(beat, hold_ticks_per_beat)

        line_towrite = line

        """
            Note logic
        """
        # Add active holds (user must press for judgment) into line as 4
        aug_line = notelines.add_active_holds(line_towrite, active_holds)

        panel_idx_to_action = notelines.panel_idx_to_action(line)
        for panel_idx, action in panel_idx_to_action.items():
            if action == '3':
                if panel_idx not in active_holds:
                    # Tried to release hold that does not exist
                    # this happens when hold starts in fake or warp
                    line_towrite = edit_string(line_towrite, panel_idx, '0')
                    aug_line = edit_string(aug_line, panel_idx, '0')

        # write
        if not in_fake_or_warp(beat) and line_towrite != empty_line:
            d = {
                'Time': time,
                'Beat': beat,
                'Line': line_towrite,
                'Line with active holds': aug_line,
                'Limb annotation': '',
                'Comment': comment,
            }
            for k, v in d.items():
                dd[k].append(v)

        if in_fake_or_warp(beat):
            """ If in fake or warp and line has hold releases,
                write a line with only the hold releases
            """
            end_hold_aug_line = notelines.add_active_holds(empty_line, active_holds)
            for panel_idx, action in panel_idx_to_action.items():
                if action == '3':
                    if panel_idx in active_holds:
                        end_hold_aug_line = edit_string(end_hold_aug_line, panel_idx, '3')
            if '3' in end_hold_aug_line:
                d = {
                    'Time': time,
                    'Beat': beat,
                    'Line': end_hold_aug_line.replace('3', '0'),
                    'Line with active holds': end_hold_aug_line,
                    'Limb annotation': '',
                    'Comment': comment,
                }
                for k, v in d.items():
                    dd[k].append(v)

        # Update active holds
        for panel_idx, action in panel_idx_to_action.items():
            if action == '2':
                if not in_fake_or_warp(beat):
                    # only start holds if not in fake or warp
                    active_holds.add(panel_idx)
            if action == '3':
                if panel_idx in active_holds:
                    active_holds.remove(panel_idx)
        # end note logic

        """
            Hold ticks logic
            11/4/24 - implementation does not match official youtube chart hold counts,
            but it's close -- not sure why 
        """
        if '3' in line:
            # pop and store current hold tick
            if curr_hold_tick is not None:
                if curr_hold_tick.ticks > 0 and time != curr_hold_tick.start_time:
                    curr_hold_tick.end_time = time
                    hold_tick_list.append(curr_hold_tick)

            if len(active_holds) == 0:
                curr_hold_tick = None
            else:
                curr_hold_tick = HoldTick(time, -1, 1)
        if '2' in line:
            if curr_hold_tick is None:
                # init hold tick
                curr_hold_tick = HoldTick(time, -1, 1)
            else:
                # pop and store current hold tick
                if curr_hold_tick.ticks > 0 and time != curr_hold_tick.start_time:
                    curr_hold_tick.end_time = time
                    hold_tick_list.append(curr_hold_tick)

                # reinit hold tick
                curr_hold_tick = HoldTick(time, -1, 1)


        # Update time if not in warp
        beat_increment = next_beat - beat
        if not in_warp(beat):
            time += beat_increment * (60 / bpm)
            time += stops.get(beat, 0)
            time += delays.get(beat, 0)

            if curr_hold_tick is not None:
                curr_hold_tick.ticks += hold_ticks_per_beat * beat_increment

    df = pd.DataFrame(dd)
    return df, [ht.to_tuple() for ht in merge_holdticks(hold_tick_list)], 'success'


class BeatToLines:
    def __init__(self, stepchart: StepchartSSC):
        """ Holds beat_to_lines and beat_to_increments

            beat_to_lines: dict[beat, line (str)]
            beat_to_increments: dict[beat, beat_increment (float)]
        """
        self.stepchart = stepchart
        measures = [s.strip() for s in stepchart.get('NOTES', '').split(',')]

        beat_to_lines = {}
        beat_to_increments = {}
        beat = 0

        for measure_num, measure in enumerate(measures):
            lines = measure.split('\n')
            lines = [line for line in lines if '//' not in line and line != '']
            lines = [line for line in lines if '#NOTE' not in line]
            num_subbeats = len(lines)
            if num_subbeats % 4 != 0:
                raise ValueError(f'{num_subbeats} lines in measure is not divisible by 4')

            for lidx, line in enumerate(lines):
                beat_increment = Fraction(BEATS_PER_MEASURE, num_subbeats)
                try:
                    line = notelines.parse_line(line)
                except Exception as e:
                    raise e

                beat_to_lines[float(beat)] = line
                beat_to_increments[float(beat)] = beat_increment
                beat += beat_increment

        self.beat_to_lines = beat_to_lines
        self.beat_to_increments = beat_to_increments

        self.handle_halfdouble()

    def handle_halfdouble(self):
        """ Add 00 to each side of lines """
        example_line = list(self.beat_to_lines.values())[0]
        if len(example_line) == 6:
            self.beat_to_lines = {k: f'00{v}00' for k, v in self.beat_to_lines.items()}

    def get_empty_line(self) -> str:
        example_line = list(self.beat_to_lines.values())[0]
        return '0' * len(example_line)


"""
    Parse {key}={value} dicts
"""
def parse_beat_value_map(
    data_string: str, 
    rounding = None
) -> dict[float, float]:
    """ Parses comma-delimited {key}={value} dict, with optional rounding """
    d = {}
    if data_string == '':
        return d
    for line in data_string.split(','):
        [beat, val] = line.split('=')
        beat, val = float(beat), float(val)
        if rounding:
            beat = round(beat, 3)
            val = round(val, 3)
        d[beat] = val
    return d


from collections import UserDict
class BeatToValueDict(UserDict):
    def __init__(self, d: dict):
        super().__init__(d)

    @staticmethod
    def from_string(string):
        return BeatToValueDict(parse_beat_value_map(string))

    def beat_in_any_range(self, query_beat: float, inclusive_end: bool = True) -> bool:
        """ Computes whether query_beat is in any range, interpreting
            key as starting beat, and value as length
        """
        for start_beat, length in self.data.items():
            if inclusive_end:
                if start_beat <= query_beat <= start_beat + length:
                    return True
            else:
                if start_beat <= query_beat < start_beat + length:
                    return True
        return False

    def get_event_times(self) -> list[float]:
        """ Get list of all times where anything happens """
        events = []
        for start, length in self.data.items():
            events += [start, start + length]
        return list(set(events))

