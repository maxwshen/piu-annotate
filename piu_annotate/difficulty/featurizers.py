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

from scipy.stats import trim_mean

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


def smallest_positive_difference(
    queries: npt.NDArray, 
    refs: npt.NDArray, 
    shift: bool = False
):
    """
    Find the smallest positive difference between each query and the largest 
    reference value less than the query.
    
    Parameters:
    -----------
    queries : numpy.ndarray
        Input array of shape (n,)
    refs : numpy.ndarray
        Reference array of shape (m,)
    shift : bool
        If True, shift found indices left by 1 more. This is used to handle
        staggered brackets.
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (n,) with smallest positive differences
    """
    # Sort the reference array
    sorted_refs = np.sort(refs)
    
    # Find the indices of the largest values less than each query
    # Using searchsorted for efficient lookup
    indices = np.searchsorted(sorted_refs, queries, side='left') - 1

    if shift:
        indices -= 1

    # Handle cases where no reference is less than the query
    valid_mask = indices >= 0
    
    # Initialize result array
    result = np.full_like(queries, np.nan, dtype=float)
    
    # Compute the differences for valid indices
    result[valid_mask] = queries[valid_mask] - sorted_refs[indices[valid_mask]]
    return result


class DifficultyFeaturizer:
    def __init__(self, cs: ChartStruct, debug: bool = False):
        self.cs = cs
        cs.annotate_time_since_downpress()
        cs.annotate_num_downpresses()
        annotate_skills(cs)
        self.window_times = [2, 5, 7]
        # self.window_times = [5, 10, 30, 45]

        self.times = np.array(cs.df['Time'])
        # do not adjust for staggered brackets yet in "original"
        # this ensures that we do not remove staggered brackets when filtering
        # events to timestamps with effective downpresses
        self.original_edp_times = calc_effective_downpress_times(
            cs, 
            adjust_for_staggered_brackets = False
        )
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
            
            This effectively counts "how many" / "how frequent" a skill occurs,
            but not "how fast". E.g. if brackets occur every few lines in a run,
            the event time frequency is much lower than the run NPS.
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

    def get_skill_nps_stats(
        self, 
        event_times: dict[str, npt.NDArray]
    ) -> dict[str, npt.NDArray]:
        """ Summarize statistics on `time_since` for lines with skills.
            
            This counts "how fast" a skill occurs.
            E.g. if brackets occur every few lines in a run,
            the bracket NPS will equal the run NPS.
        """
        times = self.times
        cs = self.cs

        # holds all edp times for full stepchart
        all_edp_times = np.array(self.edp_times)

        skills = {
            'bracket': times[cs.df['__bracket']], 
            'staggered_bracket': times[cs.df['__staggered bracket']],
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

        skill_nps_stats = dict()
        for skill, times in skills.items():
            # calculate time since effective downpress
            # staggered bracket times are on the second arrow of staggered bracket;
            # to calculate time since effective downpress *before* staggered bracket,
            # we use shift
            if len(times) > 0:
                time_since_edp = smallest_positive_difference(
                    times, 
                    all_edp_times,
                    shift = bool(skill == 'staggered_bracket'),
                )
                nps_edp = 1 / time_since_edp
                skill_nps_stats[f'{skill}-trimmedmean'] = trim_mean(nps_edp, 0.1)
                skill_nps_stats[f'{skill}-median'] = np.median(nps_edp)
                skill_nps_stats[f'{skill}-75thpct'] = np.percentile(nps_edp, 75)
            else:
                skill_nps_stats[f'{skill}-trimmedmean'] = 0
                skill_nps_stats[f'{skill}-median'] = 0
                skill_nps_stats[f'{skill}-75thpct'] = 0

        return skill_nps_stats

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

        skill_nps_stats = self.get_skill_nps_stats(event_times)
        fts.update(skill_nps_stats)
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
    # fn = folder + 'Dement_~After_Legend~_-_Lunatic_Sounds_D26_ARCADE.csv'
    fn = folder + 'X-Rave_-_SHORT_CUT_-_-_DM_Ashura_D18_SHORTCUT.csv'
    # fn = folder + 'Conflict_-_Siromaru_+_Cranky_D21_ARCADE.csv'
    cs = ChartStruct.from_file(fn)
    fter = DifficultyFeaturizer(cs)
    fter.featurize_full_stepchart()

    sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
    xs = fter.featurize_sections(sections)
    print(xs)
    import code; code.interact(local=dict(globals(), **locals()))

