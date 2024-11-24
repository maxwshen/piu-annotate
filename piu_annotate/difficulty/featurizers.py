"""
    Featurize ChartStruct (whole or segment) for difficulty prediction model
"""
import pandas as pd
import numpy as np
import numpy.typing as npt
from loguru import logger
from hackerargs import args
import functools
import os

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.formats.nps import calc_effective_downpress_times


def calc_max_event_frequency(times: list[float], t: float) -> float:
    """ Given `times`: list of times where event happens (such as downpress),
        with units in seconds, and window size `t` (in seconds), 
        computes the highest event frequency in any sliding window `t`.
        Returns float: units = events per second
    """
    if len(times) == 0:
        return 0.0
    
    n = len(times)
    if n == 1:
        return 1.0 / t
    
    # Use two-pointer technique to maintain a sliding window
    left = 0
    max_frequency = 0.0
    
    for right in range(n):
        # Shrink window from left while it exceeds size t
        while left < right and times[right] - times[left] > t:
            left += 1
            
        # Calculate frequency in current window
        events_in_window = right - left + 1
        frequency = events_in_window / t
        
        max_frequency = max(max_frequency, frequency)
    
    return max_frequency



def featurize(cs: ChartStruct, debug: bool = False) -> npt.NDArray:
    """ Featurize `cs` into (d_ft) np array with floats/ints.
        Features including highest frequency of events in sliding windows
        of length 5s, 10s, 30s, for events:
        - effective downpresses (excluding holds, condensing staggered brackets)
        - twists
        - brackets (including staggered brackets)
    """
    cs.annotate_time_since_downpress()
    cs.annotate_num_downpresses()
    annotate_skills(cs)

    times = np.array(cs.df['Time'])

    edp_times = calc_effective_downpress_times(cs)
    # remove staggered bracket downpresses
    for sbt in times[cs.df['__staggered bracket']]:
        if sbt in edp_times:
            edp_times.remove(sbt)

    bracket_times = np.sort(np.concatenate([
        times[cs.df['__bracket']], times[cs.df['__staggered bracket']]
    ]))
    if debug:
        import code; code.interact(local=dict(globals(), **locals()))

    event_times = {
        'edp': edp_times,
        'bracket': bracket_times,
        'twist90': times[cs.df['__twist 90']],
        'twistover90': times[cs.df['__twist over90']],
    }
    fts = dict()
    for event, times in event_times.items():
        for t in [5, 10, 30, 45]:
            fts[f'{event}-{t}'] = calc_max_event_frequency(times, t)
    
    x = np.array(list(fts.values()))
    return x


if __name__ == '__main__':
    fn = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/VVV_-_ZiGZaG_S18_ARCADE.csv'
    cs = ChartStruct.from_file(fn)

    featurize(cs, debug = True)

