"""
    ChartStruct segmentation and description
"""
from dataclasses import dataclass
import numpy as np
from loguru import logger
import itertools

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import nps
from piu_annotate.segment.segment import Section

MIN_SECTION_NOTES = 6
MIN_SECTION_SECONDS = 5


def get_min_section_seconds(level: int) -> float:
    if level < 10:
        return 16
    elif level < 16:
        return 12
    elif level < 23:
        return 8
    return 5


"""
    Annotate basic metadata for segment
"""
def get_segment_metadata(cs: ChartStruct, section: Section) -> dict[str, any]:
    """ Annotate basic metadata for `section` in `cs` like effective NPS.
    """
    edps = nps.calc_effective_downpress_times(cs)
    n_edps = sum([edp in section for edp in edps])
    
    enps = n_edps / section.time_length()

    metadata = {
        'eNPS': enps,
    }
    return metadata


"""
    Perform segmentation
"""
def split_sections_by_breaks(
    times: list[float], 
    time_since: list[float],
    has_holds: list[bool],
    level: int
) -> list[Section]:
    """ Splits `times`, `time_since` into sections separated by breaks,
        which are long time_since values.
        Recursive function.
    """
    # conditions to stop splitting
    if len(times) == 0:
        return []
    
    min_section_seconds = get_min_section_seconds(level)

    stop_splitting_conds = [
        times[-1] - times[0] < min_section_seconds,
        len(times) < MIN_SECTION_NOTES,
        max(time_since) < min(time_since) * 2,
    ]
    if any(stop_splitting_conds):
        return [Section(times[0], times[-1])]
    
    sections = []

    ok_start_idxs = [i for i, t in enumerate(times) if t > times[0] + min_section_seconds]
    ok_end_idxs = [i for i, t in enumerate(times) if t < times[-1] - min_section_seconds]
    if len(ok_start_idxs) == 0 or len(ok_end_idxs) == 0:
        return [Section(times[0], times[-1])]
    ok_start_idx = ok_start_idxs[0]
    ok_end_idx = ok_end_idxs[-1]
    if ok_start_idx >= ok_end_idx:
        return [Section(times[0], times[-1])]

    no_hold_times = [time_since[i] for i in range(ok_start_idx, ok_end_idx) if not has_holds[i]]
    if len(no_hold_times) == 0:
        return [Section(times[0], times[-1])]

    max_time = max(no_hold_times)
    if max_time < 0.1:
        return [Section(times[0], times[-1])]

    max_idxs = [ok_start_idx + i for i, val in enumerate(time_since[ok_start_idx:ok_end_idx])
                if val == max_time and not has_holds[ok_start_idx + i]]
    best_max_idx = np.argmin([times[i] - np.mean([times[0], times[-1]]) for i in max_idxs])
    split_idx = max_idxs[best_max_idx]
    sections += split_sections_by_breaks(times[:split_idx], time_since[:split_idx], has_holds[:split_idx], level)
    sections += split_sections_by_breaks(times[split_idx:], time_since[split_idx:], has_holds[split_idx:], level)
    # logger.debug((len(times), times[0], times[-1]))
    # logger.debug(f'Splitting at {times[split_idx]} using {max_time=}')
    # logger.debug(f'{sections=}')
    return sections


def find_drills(lines: list[str], n_repeats: int = 3) -> list[tuple[int, int]]:
    """ Finds drills or repeated consecutive lines.
        Only looks at lines; assumes that time_since is relatively uniform.
    """
    if len(lines) < 2:
        return []
    
    results = []    
    start = 0
    while start < len(lines) - 1:
        pair = [lines[start], lines[start + 1]]
        
        count = 0
        last_found = start
        
        for i in range(start + 2, len(lines) - 1, 2):
            if [lines[i], lines[i + 1]] == pair:
                count += 1
                last_found = i + 1
            else:
                break

        if count >= n_repeats:
            results.append((start, last_found))
            start = last_found
        else:
            start += 1
    
    # merge
    if len(results) == 0:
        return results

    merged_results = [results[0]]
    for r in results[1:]:
        prev_section = merged_results[-1]
        if r[0] <= prev_section[-1] + 1:
            merged_results.pop()
            merged_results.append((prev_section[0], r[1]))
        else:
            merged_results.append(r)
    return merged_results


def segment_drills(cs: ChartStruct, section: Section) -> list[Section]:
    """ Takes a long section defined by `cs`, `start_idx`, and `end_idx`,
        and attempts to split it into smaller sections by identifying drills.
        
        Does not use `time_since` as a feature, so intended to be run only on
        long sections with similar `time_since`.

        For example, this is intended to pick out drill sections as "rests"
        in Conflict D25, Gargoyle Full D25, etc.
    """
    all_times = list(cs.df['Time'])
    start_idx = all_times.index(section.start_time)
    end_idx = all_times.index(section.end_time)

    lines = list(cs.df['Line'][start_idx : end_idx + 1])
    timespan = cs.df['Time'].loc[end_idx] - cs.df['Time'].loc[start_idx]
    times = list(cs.df['Time'][start_idx : end_idx + 1])

    drills = find_drills(lines)
    if len(drills) == 0:
        return [section]

    print(drills)

    # merge drill sections that are close in time
    if len(drills) > 1:
        merged_drills = [drills[0]]
        for drill_tpl in drills:
            prev_drill = merged_drills[-1]
            prev_end_time = times[prev_drill[1]]
            start_time = times[drill_tpl[0]]
            if start_time < prev_end_time + MIN_SECTION_SECONDS:
                merged_drills.pop()
                merged_drills.append((prev_drill[0], drill_tpl[1]))
            else:
                merged_drills.append(drill_tpl)
        drills = merged_drills
        print(merged_drills)

    # section before first drill
    sections = [Section(section.start_time, times[drills[0][0] - 1])]
    for d1, d2 in itertools.pairwise(drills):
        # append section for drill1
        sections.append(Section(times[d1[0]], times[d1[1]]))

        # append section for non-drill between d1 and d2
        sections.append(Section(times[d1[1] + 1], times[d2[0] - 1]))

    # append last drill
    sections.append(Section(times[drills[-1][0]], times[drills[-1][1]]))

    # append last section after final drill
    sections.append(Section(times[drills[-1][1] + 1], section.end_time))

    return sections


def segment_chart(cs: ChartStruct) -> list[Section]:
    """ Segment `cs` into a list of Sections """
    cs.annotate_time_since_downpress()
    times = list(cs.df['Time'])
    time_since = list(cs.df['__time since prev downpress'])
    time_since = [round(ts, 3) for ts in time_since]
    has_hold = lambda line: '3' in line or '4' in line
    has_holds = [has_hold(line) for line in cs.df['Line with active holds']]
    level = int(cs.metadata['METER'])

    sections = sorted(split_sections_by_breaks(times, time_since, has_holds, level))
    # drop zero-length sections; this can occur if song starts with 1 arrow,
    # then has long wait until next arrows.
    # dropping these means that sections are not guaranteed to cover all notes
    sections = [s for s in sections if s.time_length() > 0]

    # further split very long sections by drills
    # this handles charts that have all 16th notes, like gargoyle or conflict d25
    # which are not properly split into sections when using breaks alone
    # updated_sections = []
    # for section in sections:
    #     if section.time_length() > 45:
    #         if int(cs.metadata['METER']) >= 16:
    #             logger.debug(f'{cs.metadata["shortname"]}, {section.start_time}, {section.end_time}')
    #             drill_sections = segment_drills(cs, section)
    #             print(drill_sections)
    #             import code; code.interact(local=dict(globals(), **locals()))
    #             # updated_sections += segment_drills(cs, section)
    #     else:
    #         updated_sections.append(section)

    return sections