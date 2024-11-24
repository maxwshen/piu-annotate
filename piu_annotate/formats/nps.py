"""
    Logic for computing effective NPS
"""
import math
import itertools
import numpy as np
import functools

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines

# under this threshold, do not count hold as effective downpress. unit = seconds
HOLD_TIME_THRESHOLD = 0.3

# number of notes with same time_since to 
NUM_NOTES_TO_ANNOTATE_ENPS = 4


def calc_nps(bpm: float, note_type: int = 4) -> float:
    """ 1 beat per quarter note.
        note_type = 4 indicates quarter note.
    """
    bps = bpm / 60
    nps = bps * (note_type / 4)
    return nps


def calc_bpm(time_since: float, display_bpm: float | None) -> tuple[float, str]:
    """ From `time_since`, finds which notetype (quarter, 8th, etc.) 
        at which bpm, favoring the bpm closest to `display_bpm`.
        Returns (bpm, note_type)
    """
    allowed_notetypes = [1, 2, 4, 8, 12, 16, 24, 32]
    note_type_to_str = {
        1: 'Whole notes',
        2: 'Half notes',
        4: 'Quarter notes',
        8: '8th notes',
        12: '12th notes',
        16: '16th notes',
        24: '24th notes',
        32: '32nd notes'
    }
    nps = 1 / time_since

    bpm_notetypes = []
    for note_type in allowed_notetypes:
        bps = nps / (note_type / 4)
        bpm = bps * 60
        bpm_notetypes.append((bpm, note_type))

    # get closest bpm to display_bpm, if available
    if display_bpm is None:
        # default: 150 if missing
        display_bpm = 150

    # get best bpm to show
    def calc_score(bpm, display_bpm, note_type):
        score = np.abs(np.log2(bpm) - np.log2(display_bpm))
        if note_type in [12, 24]:
            score += 0.2
        return score

    dists = [calc_score(bpm, display_bpm, notetype) for (bpm, notetype) in bpm_notetypes]
    best_idx = dists.index(min(dists))
    return bpm_notetypes[best_idx][0], note_type_to_str[bpm_notetypes[best_idx][1]]


@functools.lru_cache
def calc_effective_downpress_times(cs: ChartStruct) -> list[float]:
    """ Calculate times of effective downpresses.
        An effective downpress is 1 or 2, where we do not count lines
        with only hold starts if they repeat the previous line, and occur
        soon after the previous line.
    """
    edp_times = []
    for idx, row in cs.df.iterrows():
        time = row['Time']
        line = row['Line']
        if not notelines.has_downpress(line):
             continue
        if idx == 0:
            edp_times.append(time)
        else:
            if notelines.is_hold_start(line):
                crits = [
                    line['__line repeats previous downpress line'],
                    line['__time since prev downpress'] < HOLD_TIME_THRESHOLD
                ]
                if all(crits):
                    # hold repeats prev downpresses, and occurs soon after - skip
                    continue
        
            if time < edp_times[-1] + 0.005:
                # ignore notes very close together (faster than 200 nps);
                # these are ssc artifacts
                continue
            edp_times.append(time)
    return edp_times


def annotate_enps(cs: ChartStruct) -> tuple[list[float], list[str]]:
    """ Given `cs`, creates a short list of
        string annotations for eNPS at specific times,
        for chart visualization.
        Returns list of times, and list of string annotations.
    """
    cs.annotate_time_since_downpress()
    cs.annotate_line_repeats_previous()

    # get timestamps of effective downpresses
    edp_times = calc_effective_downpress_times(cs)
    edp_times = np.array(edp_times)

    time_since = edp_times[1:] - edp_times[:-1]
    np.insert(time_since, 0, time_since[0])

    # get display bpm
    display_bpm = None
    if 'DISPLAYBPM' in cs.metadata:
        if ':' not in cs.metadata['DISPLAYBPM']:
            display_bpm = float(cs.metadata['DISPLAYBPM'])
            if display_bpm < 0:
                display_bpm = None

    # get enps
    nn = NUM_NOTES_TO_ANNOTATE_ENPS
    annots = []
    annot_times = []
    for i in range(0, len(edp_times) - nn):
        tss = time_since[i : i + nn]
        all_time_since_same = all(math.isclose(x, y) for x, y in itertools.pairwise(tss))
        if all_time_since_same:

            bpm, notetype = calc_bpm(time_since[i], display_bpm)
            nps = 1 / time_since[i]
            annot = f'{nps:.1f} nps\n{notetype}\n{round(bpm) }bpm'
            if not annots or annot != annots[-1]:
                annots.append(annot)
                annot_times.append(edp_times[i])

    return list(zip(annot_times, annots))