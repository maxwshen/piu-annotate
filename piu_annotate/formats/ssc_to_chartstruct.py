"""
    Code for converting .ssc into chartstruct format, parsing BPM change
    info to annotate each `line` with time and beat
"""
import numpy as np
import copy
from fractions import Fraction
import pandas as pd
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from .sscfile import StepchartSSC
from . import notelines

WARP_RELEASE_TIME = Fraction(1 / 1000)


def stepchart_ssc_to_chartstruct(
    stepchart: StepchartSSC,
    debug: bool = False,
) -> tuple[pd.DataFrame, int]:
    """ Builds df to create ChartStruct object.
        df has one row per "line" and 
        cols = ['Beat', 'Time', 'Line', 'Line with active holds', 'Limb annotation']
    """
    assert stepchart.has_4_4_timesig()

    warps = Warps(stepchart.get('WARPS', ''))
    
    beat_to_bpm = BeatToBPM(stepchart.get('BPMS', ''))
    beat_to_bpm.apply_warps(warps)

    # setup
    beat = 0
    time = 0
    bpm = beat_to_bpm.get_init_bpm(beat)

    b2l = BeatToLines(stepchart, warps)

    stops = BeatToValueDict.from_string(stepchart.get('STOPS', ''))
    stops.apply_warp(warps)
    delays = BeatToValueDict.from_string(stepchart.get('DELAYS', ''))
    delays.apply_warp(warps)

    empty_line = b2l.get_empty_line()
    beat_to_lines = b2l.beat_to_lines
    beat_to_increments = b2l.beat_to_increments
    active_holds = set()
    num_bad_lines = 0
    dd = defaultdict(list)
    for beat, line in beat_to_lines.items():
        """ Iterate over beat and lines, incrementing time based on bpm.
            Process BPM changes by beat.
            Track active holds with 4.
        """
        time += delays.get(beat, 0)
        comment = ''

        if notelines.has_notes(line):
            # Add active holds into line as 4
            bad_line = False
            try:
                aug_line = notelines.add_active_holds(line, active_holds)
            except:
                aug_line = line
                bad_line = True
                comment = 'Tried placing 4 onto 1'

            active_panel_to_action = notelines.panel_to_action(line)
            bad_hold_releases = []
            for p in active_panel_to_action:
                a = active_panel_to_action[p]
                if a == '3' and p not in active_holds:
                    # Tried to release hold that doesn't exist
                    pidx = notelines.panel_to_idx[p]
                    bad_hold_releases.append(pidx)
                    # bad_line = True
                    # comment = 'Tried releasing non-existent hold'
                    line = line[:pidx] + '0' + line[pidx+1:]
                    aug_line = aug_line[:pidx] + '0' + aug_line[pidx+1:]


            d = {
                'Time': time,
                'Beat': beat,
                'Line': line,
                'Line with active holds': aug_line,
                'Limb annotation': '',
                'Comment': comment,
            }
            if set(line) != set(['0']):
                for k, v in d.items():
                    dd[k].append(v)

            # Update active holds
            if not bad_line:
                for p in active_panel_to_action:
                    a = active_panel_to_action[p]
                    if a == '2':
                        active_holds.add(p)
                    if a == '3':
                        if p in active_holds:
                            active_holds.remove(p)
            
            if bad_line:
                num_bad_lines += 1

        bi = beat_to_increments[beat]
        time, bpm, beat_to_bpm = update_time(time, beat, bi, bpm, beat_to_bpm)
        time += stops.get(beat, 0)

    df = pd.DataFrame(dd)
    # logger.debug(f'Found {num_bad_lines=}')
    return df, num_bad_lines


class Warps:
    def __init__(self, warp_string: str):
        """ Stores WARPS: on a beat, warp = skip forward some beats
        """
        self.dict = parse_beat_value_map(warp_string, rounding = 3)
    
    def total_warp_beat(self, query_beat: float) -> float:
        """ Get sum number of beats warped up to `beat` """
        total_beats = 0
        for start_beat, warp_length in self.dict.items():
            end_beat = start_beat + warp_length
            if end_beat <= query_beat:
                total_beats += end_beat - start_beat
            elif start_beat <= query_beat <= end_beat:
                total_beats += query_beat - start_beat
        return total_beats

    def beat_in_any_warp(self, query_beat: float) -> bool:
        """ Round to handle beats like 1/3, 2/3
            Ending needs to be in warp: Obliteration S17
            Beginning needs to be in warp: V3 S17
            But sometimes lines are duplicated: Elvis S15
        """
        query = round(query_beat, 3)
        for start_beat, warp_length in self.dict.items():
            if start_beat < query < start_beat + warp_length:
                return True
        return False


class BeatToLines:
    def __init__(self, stepchart: StepchartSSC, warps: Warps):
        """ Holds beat_to_lines and beat_to_increments, which are updated
            for fakes and warps.

            beat_to_lines: dict[beat, line (str)]
            beat_to_increments: dict[beat, beat_increment (float)]
        """
        self.stepchart = stepchart
        measures = [s.strip() for s in stepchart.get('NOTES', '').split(',')]

        beats_per_measure = 4
        beat_to_lines = {}
        beat_to_increments = {}
        beat = 0

        for measure_num, measure in enumerate(measures):
            lines = measure.split('\n')
            lines = [line for line in lines if '//' not in line and line != '']
            num_subbeats = len(lines)

            for lidx, line in enumerate(lines):
                beat_increment = Fraction(beats_per_measure, num_subbeats)
                line = notelines.parse_line(line)

                if any(x not in set(list('01234')) for x in line):
                    logger.error(f'Bad symbol found in {line=}, {stepchart=}')
                    raise ValueError(f'Bad symbol found')
                
                beat_to_lines[float(beat)] = line
                beat_to_increments[float(beat)] = beat_increment
                beat += beat_increment

        self.beat_to_lines = beat_to_lines
        self.beat_to_increments = beat_to_increments

        self.handle_halfdouble()
        self.apply_fakes()
        self.apply_warps(warps)
        self.filter_repeated_hold_releases()
        self.add_empty_lines()

    def handle_halfdouble(self):
        """ Add 00 to each side of lines """
        example_line = list(self.beat_to_lines.values())[0]
        if len(example_line) == 6:
            self.beat_to_lines = {k: f'00{v}00' for k, v in self.beat_to_lines.items()}

    def apply_warps(self, warps: Warps) -> None:
        """ Apply WARPS field, modifying beat_to_lines.
            Remove beats in warps, except:
            - Keep hold release lines compatible with active holds
            Decide to keep start line or end line of warp
            Shift beats after warps 
        """
        beat_to_lines = self.beat_to_lines
        beat_to_incs = self.beat_to_increments

        # Remove lines in warps, except hold releases
        beats = list(beat_to_lines.keys())
        new_beat_to_lines = {}
        new_beat_to_incs = {}
        nonwarp_beats = set([b for b in beats if not warps.beat_in_any_warp(b)])
        for beat, line in beat_to_lines.items():
            if beat in nonwarp_beats:
                new_beat_to_lines[beat] = line
                new_beat_to_incs[beat] = beat_to_incs[beat]
            elif '3' in line:
                # Line in warp with hold release
                new_beat_to_lines[beat] = line
                new_beat_to_incs[beat] = beat_to_incs[beat]
        beat_to_lines = new_beat_to_lines
        beat_to_incs = new_beat_to_incs

        # Decide to keep start or end line of warp
        is_empty = lambda line: set(line) == set(['0'])
        warp_to_line = {}

        for start, length in warps.dict.items():
            end = start + length
            start_line = beat_to_lines.get(start, None)
            end_line = beat_to_lines.get(end, None)
            # print(start, end, start_line, end_line)
            
            if start_line and not end_line:
                warp_to_line[start] = start_line
            elif not start_line and end_line:
                warp_to_line[start] = end_line
            elif start_line and end_line:
                if is_empty(start_line):
                    warp_to_line[start] = end_line
                    # print(f'Replaced {start_line} with {end_line} at warped beat {start}')
                elif is_empty(end_line):
                    warp_to_line[start] = start_line
                    # print(f'Retained {start_line} over {end_line} at warped beat {start}')
                # both start_line and end_line exist and are not empty
                elif start_line.replace('2', '3') != end_line and \
                    start_line.replace('2', '1') != end_line and \
                    start_line.replace('3', '0') != end_line:
                    warp_to_line[start] = end_line
                    # print(f'Replaced {start_line} with {end_line} at warped beat {start}')
                else:
                    warp_to_line[start] = start_line
                    # print(f'Retained {start_line} over {end_line} at warped beat {start}')
            # print(start, end, start_line, end_line, warp_to_line.get(start, None))

        # Shift beats after warps
        new_beat_to_lines = {}
        new_beat_to_incs = {}
        for beat, line in beat_to_lines.items():
            shift = warps.total_warp_beat(beat)
            if beat in nonwarp_beats:
                shifted_beat = beat - shift
            else:
                # hold release line in warp - add very small beat offset
                shifted_beat = beat - shift + WARP_RELEASE_TIME
                while shifted_beat in new_beat_to_lines:
                    shifted_beat += WARP_RELEASE_TIME
                # logger.debug(f'Found hold release in warp; {shifted_beat}, {line}')
            # if beat starts a warp, use warp_to_line, otherwise default to line
            if shifted_beat not in new_beat_to_lines:
                new_beat_to_lines[shifted_beat] = warp_to_line.get(beat, line)
                new_beat_to_incs[shifted_beat] = beat_to_incs[beat]
        
        sorted_beats = sorted(list(new_beat_to_lines.keys()))
        self.beat_to_lines = {b: new_beat_to_lines[b] for b in sorted_beats}
        self.beat_to_increments = {b: new_beat_to_incs[b] for b in sorted_beats}
        return

    def apply_fakes(self) -> None:
        """ Apply FAKES field, modifying beat_to_lines
        """
        fakes = BeatToValueDict.from_string(self.stepchart.get('FAKES', ''))
        empty_line = self.get_empty_line()

        infake = lambda x, fake: fake[0] <= x <= fake[0] + fake[1]
        num_fakes = 0
        for beat, line in self.beat_to_lines.items():
            if fakes.beat_in_any_range(beat):
                if set(line) != set(list('03')):
                    num_fakes += 1
                    self.beat_to_lines[beat] = empty_line
                # else:
                    # logger.debug(f'Kept hold release in fake {beat}, {line}')
        # logger.debug(f'Filtered {num_fakes} fake lines')
        return

    def filter_repeated_hold_releases(self) -> None:
        """
            Ignoring empty lines, filter out duplicated hold release lines
            These can only arise from inserting hold releases during warps
        """
        beat_to_lines = self.beat_to_lines
        beat_to_incs = self.beat_to_increments

        nonempty_btol = {k: v for k, v in beat_to_lines.items() if set(v) != set('0')}
        beats = list(nonempty_btol.keys())
        empty_beats = [b for b, v in beat_to_lines.items() if set(v) == set('0')]
        assert beats == sorted(beats), 'ERROR: Beats are not sorted by default'
        ok_beats = []
        lines = [nonempty_btol[beat] for beat in sorted(beats)]
        for i in range(len(lines) - 1):
            # filter first in repeated hold release: filter one in warp, not one after warp
            line1, line2 = lines[i], lines[i+1]
            if set(line1) == set(list('03')) and line1 == line2:
                pass
            else:
                ok_beats.append(beats[i])
        ok_beats.append(beats[len(lines)-1])
        ok_beats += empty_beats

        # logger.debug(f'Filtered {len(beat_to_lines)-len(ok_beats)} repeated hold releases')
        self.beat_to_lines = {k: v for k, v in beat_to_lines.items() if k in ok_beats}
        self.beat_to_increments = {k: v for k, v in beat_to_incs.items() if k in ok_beats}
        return

    def add_empty_lines(self) -> None:
        """ Every beat + its increment should be a key in both dicts """
        beat_to_lines = self.beat_to_lines
        beat_to_incs = self.beat_to_increments
        empty_line = self.get_empty_line()

        # start at beat 0 
        beat_to_incs[-1] = 1

        add_beat_to_lines = {}
        add_beat_to_incs = {}
        beats = set(beat_to_incs.keys())
        for beat in sorted(beat_to_incs.keys()):
            next_beat = beat + beat_to_incs[beat]
            if next_beat not in beats:
                next_beats = [b for b in beats if b > beat]
                if next_beats:
                    min_next_beat = min(next_beats)
                    min_inc = min([beat_to_incs[beat], beat_to_incs[min_next_beat]])

                    nb = beat + min_inc
                    while nb < min_next_beat:
                        add_beat_to_lines[nb] = empty_line
                        add_beat_to_incs[nb] = min_inc
                        nb += min_inc

        beat_to_lines.update(add_beat_to_lines)
        beat_to_incs.update(add_beat_to_incs)

        sorted_beats = sorted(list(beat_to_lines.keys()))
        sorted_beats = [b for b in sorted_beats if b >= 0]
        self.beat_to_lines = {k: beat_to_lines[k] for k in sorted_beats}
        self.beat_to_increments = {k: beat_to_incs[k] for k in sorted_beats}
        return

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


class BeatToValueDict:
    def __init__(self, d: dict):
        self.dict = d

    @staticmethod
    def from_string(string):
        return BeatToValueDict(parse_beat_value_map(string))

    def get(self, key: str, default_value: any) -> any:
        return self.dict.get(key, default_value)

    def apply_warp(self, warps: Warps):
        warped_data = {beat - warps.total_warp_beat(beat): val
                       for beat, val in self.dict.items()}
        self.dict = warped_data

    def beat_in_any_range(self, query_beat: float) -> bool:
        """ Computes whether query_beat is in any range, interpreting
            key as starting beat, and value as length
        """
        for start_beat, length in self.dict.items():
            if start_beat < query_beat < start_beat + length:
                return True
        return False


class BeatToBPM:
    def __init__(self, bpms_string: str):
        """ Stores a mapping of beat to BPM.

            Parse BPMS:0.000000=67.500000
            ,68.500000=37.500000
            ,76.000000=41.250000
            ,77.000000=48.750000
            ,78.000000=56.250000 ...
            which are comma-delimited {beat}={bpm}.
        """
        self.beat_to_bpm = dict()
        for line in bpms_string.split(','):
            [beat, bpm] = line.split('=')
            self.beat_to_bpm[float(beat)] = float(bpm)
        self.beat_to_bpm[np.inf] = 0
        self.beats = list(self.beat_to_bpm.keys())
        self.validate()

    def validate(self) -> bool:
        return sorted(self.beats) == self.beats

    def __getitem__(self, key: float) -> float:
        return self.beat_to_bpm[key]

    def pop(self) -> tuple[float, float]:
        """ Pops next beat, returns bpm """
        current_beat = self.beats.pop(0)
        bpm = self.beat_to_bpm.pop(current_beat)
        return bpm

    def next_bpm_update_beat(self) -> float:
        """ Returns next beat where bpm will update """
        return self.beats[0]

    def get_init_bpm(self, init_beat: float):
        """ Initialize, getting bpm at `init_beat` """
        while init_beat >= self.next_bpm_update_beat():
            bpm = self.pop()
        return bpm

    def apply_warps(self, warps: Warps):
        """ """
        warped_data = {beat - warps.total_warp_beat(beat): val
                       for beat, val in self.beat_to_bpm.items()}
        self.beat_to_bpm = warped_data
        self.beats = list(self.beat_to_bpm.keys())
        return


"""
    BPM, beat, and time logic
"""
def update_time(
    time: float, 
    beat: float, 
    beat_increment: float, 
    bpm: float, 
    beat_to_bpm: BeatToBPM
) -> tuple[float, float, BeatToBPM]:
    """
        After processing line, update bpm, and time.
        Important: Update time before bpm.
    """
    next_bpm_update_beat = beat_to_bpm.next_bpm_update_beat()
    next_note_beat = beat + beat_increment

    orig_time = copy.copy(time)

    while next_bpm_update_beat <= next_note_beat:
        # 1 or more bpm updates before next note line.
        # For each bpm update, update beat, time (using bpm+beat), and bpm.
        bi = next_bpm_update_beat - beat
        if bi < 0:
            print('Error: Negative beat increment')
            raise Exception('Error: Time decreased')
        time += bi * (60 / bpm)
        beat += bi
        if beat >= beat_to_bpm.next_bpm_update_beat():
            bpm = beat_to_bpm.pop()
            next_bpm_update_beat = beat_to_bpm.next_bpm_update_beat()
        assert bpm is not None, 'Error: Failed to set bpm'

    # No more bpm updates before next note line.
    # Update time. No need to update beat, bpm.
    if beat < next_note_beat:
        bi = next_note_beat - beat
        time += bi * (60 / bpm)
    assert bpm is not None, 'Error: Failed to set bpm'
    # print(beat, bpm)
    if time < orig_time:
        print('Error: Time decreased')
        raise Exception('ERROR: Time decreased')
    return time, bpm, beat_to_bpm

