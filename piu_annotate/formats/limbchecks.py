import pandas as pd
from tqdm import tqdm
from loguru import logger
import math
import numpy as np
from collections import defaultdict
import functools

from piu_annotate.formats.chart import ChartStruct


def check_unforced_doublestep(cs: ChartStruct) -> list[int]:
    """ Checks for unforced doublesteps:
        - Two lines each with one downpress, not on the same arrow
        - Neither line has an active hold
        - First line is 1
        - Second line can be 1 or 2
        - Faster than 8th note rhythm at 100 bpm (>3.5 nps)
        - Not a staggered bracket (slower than ~13 nps)
        - Same limb used
        If conditions are satisfied, the second line is an unforced doublestep.

        Returns list of row indices with doublestep
    """
    cs.annotate_num_downpresses()

    found_idxs = []
    for idx, row in cs.df.iterrows():
        if idx + 1 == len(cs.df):
            break
        next_row = cs.df.iloc[idx + 1]

        if row['__num downpresses'] != 1 or next_row['__num downpresses'] != 1:
            continue
        
        if row['Limb annotation'] != next_row['Limb annotation']:
            continue

        line = row['Line with active holds'].replace('`', '')
        next_line = next_row['Line with active holds'].replace('`', '')

        has_active_hold = lambda l: '3' in l or '4' in l
        if any(has_active_hold(l) for l in [line, next_line]):
            continue
        if '1' not in line:
            continue
        arrow_pos = line.index('1')
        if next_line[arrow_pos] != '0':
            continue
        
        time_since = next_row['Time'] - row['Time']
        if time_since > 1 / 3.5:
            continue
        if time_since < 1 / 13:
            continue
        
        if cs.get_chart_level() <= 19:            
            import code; code.interact(local=dict(globals(), **locals()))
        found_idxs.append(idx + 1)

    return found_idxs