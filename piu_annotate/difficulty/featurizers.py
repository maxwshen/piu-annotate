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
import copy

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.formats.nps import calc_effective_downpress_times
from piu_annotate.segment.segment import Section


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


class DifficultyFeaturizer:
    def __init__(self, cs: ChartStruct, debug: bool = False):
        self.cs = cs
        cs.annotate_time_since_downpress()
        cs.annotate_num_downpresses()
        annotate_skills(cs)
        self.window_times = [2, 5, 7]
        # self.window_times = [5, 10, 30, 45]

        self.times = np.array(cs.df['Time'])
        self.original_edp_times = calc_effective_downpress_times(cs)
        self.edp_times = copy.copy(self.original_edp_times)

        # remove staggered bracket downpresses
        for sbt in self.times[cs.df['__staggered bracket']]:
            if sbt in self.edp_times:
                self.edp_times.remove(sbt)

        self.bracket_times = np.sort(np.concatenate([
            self.times[cs.df['__bracket']], self.times[cs.df['__staggered bracket']]
        ]))

    def get_event_times(self) -> dict[str, npt.NDArray]:
        """ Get list of timestamps during which skills occur in stepchart/section.
        """
        times = self.times
        cs = self.cs
        event_times = {
            'edp': np.array(self.edp_times),
            'bracket': self.bracket_times,
            'twist90': times[cs.df['__twist 90']],
            'twistclose': times[cs.df['__twist close']],
            'twistfar': times[cs.df['__twist far']],
            'run': times[cs.df['__run']],
            'drill': times[cs.df['__drill']],
            'doublestep': times[cs.df['__doublestep']],
            'jump': times[cs.df['__jump']],
            'jack': times[cs.df['__jack']],
            'footswitch': times[cs.df['__footswitch']],
        }
        # reduce to event times for effective downpresses
        for k in event_times:
            event_times[k] = np.array([t for t in event_times[k]
                                       if t in self.original_edp_times])
        return event_times

    def get_feature_dict(self, section: Section | None = None) -> dict[str, float]:
        """ Returns dict of {feature name: value}.

            If section is provided, then trims event times to section.
            For window sizes longer than section length, expand max event frequency
            with discount factor to "repeat" the section to fill up the window.
        """
        event_times = self.get_event_times()

        if section:
            filt_times = lambda ts: ts[(ts >= section.start_time) & (ts < section.end_time)]
            # trim event times to specific section
            for k in event_times:
                event_times[k] = filt_times(event_times[k])

        fts = dict()
        for event, ts in event_times.items():
            for t in self.window_times:
                fq = calc_max_event_frequency(ts, t)

                if section:
                    # if section is shorter than time window,
                    # extrapolate by simulating if section was repeated,
                    # but with discount factor
                    sec_len = section.time_length()
                    if sec_len < t:
                        ratio = t / sec_len
                        # fq *= ratio
                        adj = 0.75
                        fq *= ratio * adj

                feature_name = f'{event}-{t}'
                fts[feature_name] = fq

        return fts

    def get_feature_names(self) -> list[str]:
        return list(self.get_feature_dict().keys())

    def featurize_full_stepchart(self) -> npt.NDArray:
        fts = self.get_feature_dict()
        x = np.array(list(fts.values()))
        return x

    def featurize_sections(self, sections: list[Section]) -> npt.NDArray:
        all_x = []
        for section in sections:
            fts = self.get_feature_dict(section = section)            
            x = np.array(list(fts.values()))
            all_x.append(x)
        return np.stack(all_x)


if __name__ == '__main__':
    folder = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/'
    fn = folder + 'Dement_~After_Legend~_-_Lunatic_Sounds_D26_ARCADE.csv'
    # fn = folder + 'Conflict_-_Siromaru_+_Cranky_D21_ARCADE.csv'
    cs = ChartStruct.from_file(fn)
    fter = DifficultyFeaturizer(cs)
    fter.featurize_full_stepchart()

    sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
    xs = fter.featurize_sections(sections)
    print(xs)
    import code; code.interact(local=dict(globals(), **locals()))

