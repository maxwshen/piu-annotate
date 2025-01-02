"""
    Annotate a ChartStruct with skills columns
"""
import math
import re
import itertools
from loguru import logger
import pandas as pd
import numpy as np

from hackerargs import args
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines

# used to filter short runs
# for example, if a run is shorter than 8 notes, do not label as run
MIN_DRILL_LEN = 5
MIN_RUN_LEN = 7
MIN_ANCHOR_RUN_LEN = 7

# used for side3 singles, mid4 doubles, etc.
MIN_POSITION_NOTES_LEN = 8

# faster than 13 nps
STAGGERED_BRACKET_TIME_SINCE_THRESHOLD = 1/13


def extract_consecutive_true_runs(bools: list[bool]) -> list[tuple[int, int]]:
    """ Given a list of bools, returns a list of tuples of start_idx, end_idx
        (inclusive) for each consecutive run of True in `bools`.
    """
    if not bools:
        return []
    
    runs = []
    current_run_start = None
    
    for idx, val in enumerate(bools):
        if val and current_run_start is None:
            # Start of a new True run
            current_run_start = idx
        elif not val and current_run_start is not None:
            # End of a True run
            runs.append((current_run_start, idx - 1))
            current_run_start = None
    
    # Check if the last run goes to the end of the list
    if current_run_start is not None:
        runs.append((current_run_start, len(bools) - 1))
    
    return runs


"""
    Line has ...
"""
def has_bracket(line: str, limb_annot: str) -> bool:
    """ Computes if `limb_annot` for `line` implies that a bracket is performed.
    """
    if limb_annot.count('l') < 2 and limb_annot.count('r') < 2:
        return False
    arrow_positions = [i for i, s in enumerate(line) if s != '0']
    if len(arrow_positions) < 2:
        return False
    valid_limbs = notelines.multihit_to_valid_feet(arrow_positions)
    mapper = {'l': 0, 'r': 1, 'e': 0, 'h': 0}
    return tuple(mapper[l] for l in limb_annot) in valid_limbs


def has_hands(line: str, limb_annot: str) -> bool:
    """ Computes if `limb_annot` for `line` implies that hands are used.
        Forgive if limb annotation has 'e' in it.
    """
    if 'h' in limb_annot:
        return True
    if limb_annot.count('l') < 2 and limb_annot.count('r') < 2:
        return False
    arrow_positions = [i for i, s in enumerate(line) if s != '0']
    if len(arrow_positions) < 2:
        return False
    if 'e' in limb_annot:
        return False
    valid_limbs = notelines.multihit_to_valid_feet(arrow_positions)
    mapper = {'l': 0, 'r': 1, 'e': 0, 'h': 0}
    return not tuple(mapper[l] for l in limb_annot) in valid_limbs


"""
    ChartStruct annotation functions 
"""
def drills(cs: ChartStruct) -> None:
    """ Adds or updates columns to `cs.df` for drills
        
        A drill is:
        - Starts with two lines, which:
            - Have one 1 in them (allow 4)
            - Have alternate feet
        - Additional lines:
            - Have same "time since" as second start line
            - Repeat the first two lines
    """
    df = cs.df
    lines = cs.get_lines()
    limb_annots = list(df['Limb annotation'])
    ts = list(df['__time since prev downpress'])

    drill_idxs = set()
    i, j = 0, 1
    while j < len(df):
        # include bracket drill
        crits = [
            '1' in lines[i],
            '1' in lines[j],
            set(limb_annots[i]) != set(limb_annots[j]),
            len(set(limb_annots[i])) == 1,
            len(set(limb_annots[j])) == 1,
        ]
        if all(crits):
            # k iterates to extend drill
            k = j + 1
            while k < len(df):
                # Must repeat first two lines
                if (k - i) % 2 == 0:
                    same_as = lines[k] == lines[i]
                else:
                    same_as = lines[k] == lines[j]
                consistent_rhythm = math.isclose(ts[k], ts[j])
                if same_as and consistent_rhythm:
                    k += 1
                else:
                    break
        
            # Found drill
            if k - i >= MIN_DRILL_LEN:
                for idx in range(i, k):
                    drill_idxs.add(idx)

        i += 1
        j += 1

    cs.df['__drill'] = [bool(i in drill_idxs) for i in range(len(cs.df))]
    return


def run(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines()
    limb_annots = list(df['Limb annotation'])
    drills = list(df['__drill'])
    ts = list(df['__time since prev downpress'])
    
    idxs = set()
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            set(limb_annots[i]) != set(limb_annots[j]),
            len(set(limb_annots[i])) == 1,
            len(set(limb_annots[j])) == 1,
            '1' in lines[j],
            math.isclose(ts[i], ts[j])
        ]
        if all(crits):
            idxs.add(j)

    # if any run is entirely contained within a drill, do not label as run
    drill_sections = extract_consecutive_true_runs(drills)
    runs = filter_short_runs(idxs, len(df), MIN_RUN_LEN)
    run_sections = extract_consecutive_true_runs(runs)

    def run_in_any_drill(run_start: int, run_end: int):
        return any(
            run_start >= drill_start and run_end <= drill_end
            for drill_start, drill_end in drill_sections
        )

    for run in run_sections:
        run_start, run_end = run
        if run_in_any_drill(run_start, run_end):
            run_len = run_end - run_start + 1
            runs[run_start : run_end + 1] = [False] * run_len

    cs.df['__run'] = runs
    return


def anchor_run(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines()
    limb_annots = list(df['Limb annotation'])
    ts = list(df['__time since prev downpress'])
    drills = list(df['__drill'])

    anchor_run_idxs = set()
    i, j = 0, 1
    while j < len(df):
        # row i, j form two starting lines of anchor run

        # allow for brackets in anchor run
        crits = [
            '1' in lines[i],
            '1' in lines[j],
            set(limb_annots[i]) != set(limb_annots[j]),
            len(set(limb_annots[i])) == 1,
            len(set(limb_annots[j])) == 1,
            not drills[j],
        ]
        if all(crits):
            # k iterates to extend run
            k = j + 1
            odds_same = []
            evens_same = []
            while k < len(df):
                # Must repeat one of first two lines
                if (k - i) % 2 == 0:
                    same_as = lines[k] == lines[i]
                    if len(evens_same) == 0:
                        evens_same.append(same_as)
                    else:
                        if all(evens_same) and not same_as:
                            break
                else:
                    same_as = lines[k] == lines[j]
                    if len(odds_same) == 0:
                        odds_same.append(same_as)
                    else:
                        if all(odds_same) and not same_as:
                            break
                
                if len(odds_same) > 0 and len(evens_same) > 0:
                    if not (all(odds_same) or all(evens_same)):
                        break

                consistent_rhythm = math.isclose(ts[k], ts[j])
                if consistent_rhythm:
                    k += 1
                else:
                    break
        
            # Found anchor run
            if k - i >= MIN_ANCHOR_RUN_LEN:
                for idx in range(i, k):
                    anchor_run_idxs.add(idx)
        i += 1
        j += 1

    cs.df['__anchor run'] = [bool(i in anchor_run_idxs) for i in range(len(cs.df))]
    return


def brackets(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    bracket_annots = [has_bracket(line, la) for line, la in zip(lines, limb_annots)]
    cs.df['__bracket'] = bracket_annots
    return


def staggered_brackets(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines()
    limb_annots = list(df['Limb annotation'])
    ts = list(df['__time since prev downpress'])

    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            notelines.has_one_1(lines[i]),
            notelines.has_one_1(lines[j]),
            limb_annots[i] == limb_annots[j],
            ts[j] < STAGGERED_BRACKET_TIME_SINCE_THRESHOLD,
            notelines.staggered_bracket(lines[i], lines[j])
        ]
        res.append(all(crits))

    cs.df['__staggered bracket'] = res
    return


def doublestep(cs: ChartStruct) -> None:
    """ Doublestep should be detected in 86 during active holds,
        and in End of the World D22 with hold release between double steps.
    """
    df = cs.df
    lines = cs.get_lines()
    line_ahs = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])
    staggered_brackets = list(df['__staggered bracket'])
    jacks = list(df['__jack'])

    prev_dp_limbs = notelines.get_downpress_limbs(line_ahs[0], limb_annots[0])
    prev_dp_arrows = notelines.get_downpress_arrows(line_ahs[0])
    res = [False]
    for i in range(1, len(df)):
        dp_limbs = notelines.get_downpress_limbs(line_ahs[i], limb_annots[i])
        dp_arrows = notelines.get_downpress_arrows(line_ahs[i])

        is_doublestep = False
        if len(dp_limbs) == 1:
            # check if limb is in previous downpress limbs --
            # this means that 10001 -> 00100 considered downpress
            limb = list(dp_limbs)[0]
            crits = [
                limb in prev_dp_limbs,
                dp_arrows != prev_dp_arrows,
                not jacks[i],
                not staggered_brackets[i],
            ]
            is_doublestep = all(crits)

        # has doublestep, so update prev dp limbs
        if len(dp_limbs) > 0:
            prev_dp_limbs = dp_limbs
            prev_dp_arrows = dp_arrows

        res.append(is_doublestep)

    cs.df['__doublestep'] = res
    return


def hands(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])
    cs.df['__hands'] = [has_hands(line, la) for line, la in zip(lines, limb_annots)]
    return


def jump(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])
    hands = list(df['__hands'])

    res = []
    for i in range(len(df)):
        dp_limbs = notelines.get_downpress_limbs(lines[i], limb_annots[i])
        crits = [
            'l' in dp_limbs and 'r' in dp_limbs,
            not hands[i],
        ]
        res.append(all(crits))

    cs.df['__jump'] = res
    return


def twists_90(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    res = [False]
    # i holds the index of the previous line with downpress
    i = 0
    for j in range(1, len(df)):
        if not notelines.has_downpress(lines[j]):
            res.append(False)
            continue

        # j has downpress
        is_twist = False
        if 'r' in limb_annots[i] and 'l' in limb_annots[j]:
            leftmost_r_panel = notelines.get_leftmost_rightfoot_panel(
                lines[i], limb_annots[i]
            )
            rightmost_l_panel = notelines.get_rightmost_leftfoot_panel(
                lines[j], limb_annots[j]
            )
            if leftmost_r_panel is not None and rightmost_l_panel is not None:
                is_twist = any([
                    notelines.is_90_twist(leftmost_r_panel, rightmost_l_panel),
                ])
            
        if 'l' in limb_annots[i] and 'r' in limb_annots[j]:
            rightmost_l_panel = notelines.get_rightmost_leftfoot_panel(
                lines[i], limb_annots[i]
            )
            leftmost_r_panel = notelines.get_leftmost_rightfoot_panel(
                lines[j], limb_annots[j]
            )
            if leftmost_r_panel is not None and rightmost_l_panel is not None:
                is_twist = is_twist or any([
                    notelines.is_90_twist(leftmost_r_panel, rightmost_l_panel),
                ])

        res.append(is_twist)

        # update i because line had downpress
        i = j

    cs.df['__twist 90'] = res
    return


def twists_over90(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    list_is_over90_twist = [False]
    list_is_close_twist = [False]
    list_is_far_twist = [False]

    # i holds the index of the previous line with downpress
    i = 0
    for j in range(1, len(df)):
        if not notelines.has_downpress(lines[j]):
            list_is_over90_twist.append(False)
            list_is_close_twist.append(False)
            list_is_far_twist.append(False)
            continue

        # j has downpress

        is_over90_twist = False
        is_close_twist = False
        is_far_twist = False
        if 'r' in limb_annots[i] and 'l' in limb_annots[j]:
            leftmost_r_panel = notelines.get_leftmost_rightfoot_panel(
                lines[i], limb_annots[i]
            )
            rightmost_l_panel = notelines.get_rightmost_leftfoot_panel(
                lines[j], limb_annots[j]
            )
            if leftmost_r_panel is not None and rightmost_l_panel is not None:
                is_over90_twist = notelines.is_over90_twist(leftmost_r_panel, rightmost_l_panel)
                is_close_twist = notelines.is_close_twist(leftmost_r_panel, rightmost_l_panel)
                is_far_twist = notelines.is_far_twist(leftmost_r_panel, rightmost_l_panel)
            
        if 'l' in limb_annots[i] and 'r' in limb_annots[j]:
            rightmost_l_panel = notelines.get_rightmost_leftfoot_panel(
                lines[i], limb_annots[i]
            )
            leftmost_r_panel = notelines.get_leftmost_rightfoot_panel(
                lines[j], limb_annots[j]
            )
            if leftmost_r_panel is not None and rightmost_l_panel is not None:
                is_over90_twist |= notelines.is_over90_twist(leftmost_r_panel, rightmost_l_panel)
                is_close_twist |= notelines.is_close_twist(leftmost_r_panel, rightmost_l_panel)
                is_far_twist |= notelines.is_far_twist(leftmost_r_panel, rightmost_l_panel)

        list_is_over90_twist.append(is_over90_twist)
        list_is_close_twist.append(is_close_twist)
        list_is_far_twist.append(is_far_twist)

        # update i because line had downpress
        i = j

    cs.df['__twist over90'] = list_is_over90_twist
    cs.df['__twist close'] = list_is_close_twist
    cs.df['__twist far'] = list_is_far_twist
    return


def side3_singles(cs: ChartStruct) -> None:
    df = cs.df
    if cs.singles_or_doubles() == 'doubles':
        cs.df['__side3 singles'] = [False] * len(cs.df)
        return

    cs.annotate_num_downpresses()
    lines = cs.get_lines_with_active_holds()
    if notelines.singlesdoubles(lines[0]) != 'singles':
        cs.df['__side3 singles'] = [False] * len(df)

    left_accept = lambda line: line[-2:] == '00'
    left_idxs = [i for i, line in enumerate(lines) if left_accept(line)]
    left_res = filter_short_runs(left_idxs, len(lines), MIN_POSITION_NOTES_LEN)
    left_res = filter_run_by_num_downpress(df, left_res, MIN_POSITION_NOTES_LEN)

    right_accept = lambda line: line[:2] == '00'
    right_idxs = [i for i, line in enumerate(lines) if right_accept(line)]
    right_res = filter_short_runs(right_idxs, len(lines), MIN_POSITION_NOTES_LEN)
    right_res = filter_run_by_num_downpress(df, right_res, MIN_POSITION_NOTES_LEN)
    cs.df['__side3 singles'] = [bool(l or r) for l, r in zip(left_res, right_res)]
    return 


def mid4_doubles(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    if notelines.singlesdoubles(lines[0]) != 'doubles':
        cs.df['__mid6 doubles'] = [False] * len(df)

    accept = lambda line: re.search('000....000', line) and any(x in line for x in list('1234'))
    idxs = [i for i, line in enumerate(lines) if accept(line)]
    res = filter_short_runs(idxs, len(lines), MIN_POSITION_NOTES_LEN)
    cs.df['__mid4 doubles'] = res
    return


def mid6_doubles(cs: ChartStruct) -> None:
    # Note - can be redundant with mid4; modify chart tags accordingly
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    if notelines.singlesdoubles(lines[0]) != 'doubles':
        cs.df['__mid6 doubles'] = [False] * len(df)

    accept = lambda line: re.search('00......00', line) and any(x in line for x in list('1234'))
    idxs = [i for i, line in enumerate(lines) if accept(line)]
    res = filter_short_runs(idxs, len(lines), MIN_POSITION_NOTES_LEN)
    cs.df['__mid6 doubles'] = res
    return


def splits(cs: ChartStruct) -> None:
    df = cs.df
    lines = list(df['Line with active holds'].apply(lambda l: l.replace('`', '')))
    ts = list(df['__time since prev downpress'])

    if cs.singles_or_doubles() == 'singles':
        cs.df['__split'] = [False] * len(cs.df)
        return

    def has_split(line: str) -> bool:
        return all([
            any(x in line[:2] for x in list('12')),
            any(x in line[-2:] for x in list('12')),
        ])
    # start with splits in any single line
    res = [has_split(line) for line in lines]

    # consider splits in consecutive downpress lines
    i = 0
    for j in range(1, len(df)):
        if not notelines.has_downpress(lines[j]):
            continue
        # j has downpress

        is_split = False

        # if j and prev downpress line (i) are splits
        # i on right, j on left
        if all([
            any(x in lines[i][:2] for x in list('12')),
            any(x in lines[j][-2:] for x in list('12')),
            ts[j] < 0.5,
            notelines.num_downpress(lines[i]) == 1,
            notelines.num_downpress(lines[j]) == 1,
        ]):
            is_split = True

        # i on left, j on right
        if all([
            any(x in lines[j][:2] for x in list('12')),
            any(x in lines[i][-2:] for x in list('12')),
            ts[j] < 0.5,
            notelines.num_downpress(lines[i]) == 1,
            notelines.num_downpress(lines[j]) == 1,
        ]):
            is_split = True

        res[j] |= is_split

        # update i because line had downpress
        i = j

    cs.df['__split'] = res
    return


def jack(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])
    ts = list(df['__time since prev downpress'])

    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            lines[j] == lines[i],
            notelines.num_downpress(lines[i]) == 1,
            limb_annots[i] == limb_annots[j],
            ts[j]
        ]
        res.append(all(crits))

    cs.df['__jack'] = res
    return


def footswitch(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            lines[j] == lines[i],
            notelines.num_downpress(lines[i]) == 1,
            limb_annots[i] != limb_annots[j],
        ]
        res.append(all(crits))

    cs.df['__footswitch'] = res
    return


def hold_footswitch(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    cs.df['__hold footswitch'] = ['e' in la for line, la in zip(lines, limb_annots)]
    return


def hold_footslide(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])
    vals = np.array([False] * len(df))

    def get_panel_downpresses_with_limb(line_ah: str, limb_annot: str, limb: str):
        panels = []
        for panel, action in notelines.panel_idx_to_action(line).items():
            if action in ['1', '2', '4']:
                l = notelines.get_limb_for_arrow_pos(line_ah, limb_annot, panel)
                if l == limb:
                    panels.append(panel)
        return panels
    
    def get_hold_start_idx(panel: int, hold_end_idx: int):
        for i, l in enumerate(lines[:hold_end_idx][::-1]):
            if l[panel] == '2':
                return hold_end_idx - i
        assert False

    active_holds = dict()
    for idx, line in enumerate(lines):
        limb_annot = limb_annots[idx]
        for panel, action in notelines.panel_idx_to_action(line).items():
            if action == '2':
                limb = notelines.get_limb_for_arrow_pos(line, limb_annot, panel)

                # get all panel downpresses with limb

                active_holds[(panel, limb)] = get_panel_downpresses_with_limb(
                    line, limb_annot, limb
                )
            
            elif action == '4':
                possible_keys = [(panel, 'l'), (panel, 'r')]
                keys = [k for k in possible_keys if k in active_holds]
                if len(keys) > 0:
                    key = keys[0]
                    limb = key[1]

                    dp_panels = get_panel_downpresses_with_limb(
                        line, limb_annot, limb
                    )
                    for dp_panel in dp_panels:
                        if dp_panel not in active_holds[key]:
                            active_holds[key].append(dp_panel)
            
            elif action == '3':
                possible_keys = [(panel, 'l'), (panel, 'r')]
                keys = [k for k in possible_keys if k in active_holds]
                if len(keys) > 0:
                    key = keys[0]
                    limb = key[1]

                    dp_panels_during_hold = active_holds.pop(key)
                    if len(set(dp_panels_during_hold)) >= 3:
                        start_idx = get_hold_start_idx(panel, idx)
                        vals[start_idx : idx + 1] = True
        
    cs.df['__hold footslide'] = vals
    return


def stair10(cs: ChartStruct) -> None:
    df = cs.df
    lines = list(df['Line with active holds'].apply(lambda l: l.replace('`', '')))

    col = '__10-stair'
    if cs.singles_or_doubles() == 'singles':
        cs.df[col] = [False] * len(cs.df)
        return

    lines_w_downpress = [line for idx, line in enumerate(lines)
                        if notelines.has_downpress(line)]
    line_idxs = [idx for idx, line in enumerate(lines)
                 if notelines.has_downpress(line)]
    
    patterns = [
        [
            '1000000000',
            '0100000000',
            '0010000000',
            '0001000000',
            '0000100000',
            '0000010000',
            '0000001000',
            '0000000100',
            '0000000010',
            '0000000001',
        ]
    ]
    flipped_patterns = []
    for p in patterns:
        flipped_patterns.append(p[::-1])
    patterns += flipped_patterns

    vals = np.array([False] * len(cs.df))

    # match patterns
    for pattern in patterns:
        for start in range(0, len(lines_w_downpress) - len(pattern)):
            end = start + len(pattern)
            # allow matching 2 as 1
            dp_lines = [l.replace('2', '1') for l in lines_w_downpress[start : end]]

            if all(l == pl for l, pl in zip(dp_lines, pattern)):
                vals[line_idxs[start : end]] = True

    cs.df[col] = vals
    return


def yogwalk(cs: ChartStruct) -> None:
    df = cs.df
    lines = list(df['Line with active holds'].apply(lambda l: l.replace('`', '')))

    col = '__yog walk'
    if cs.singles_or_doubles() == 'singles':
        cs.df[col] = [False] * len(cs.df)
        return

    lines_w_downpress = [line for idx, line in enumerate(lines)
                        if notelines.has_downpress(line)]
    line_idxs = [idx for idx, line in enumerate(lines)
                 if notelines.has_downpress(line)]
    
    patterns = [
        [
            '0010000000',
            '0100000000',
            '0001000000',
            '0010000000',
            '0000100000',
            '0001000000',
            '0000010000',
            '0000100000',
            '0000001000',
            '0000010000',
            '0000000100',
            '0000001000',
            '0000000010',
            '0000000100',
        ],
        [
            '0010000000',
            '0100000000',
            '0000100000',
            '0010000000',
            '0001000000',
            '0000100000',
            '0000001000',
            '0001000000',
            '0000010000',
            '0000001000',
            '0000000100',
            '0000010000',
            '0000000010',
            '0000000100',
        ],
    ]
    flipped_patterns = []
    for p in patterns:
        flipped_patterns.append(p[::-1])
    patterns += flipped_patterns

    vals = np.array([False] * len(cs.df))

    # match patterns
    for pattern in patterns:
        for start in range(0, len(lines_w_downpress) - len(pattern)):
            end = start + len(pattern)
            # allow matching 2 as 1
            dp_lines = [l.replace('2', '1') for l in lines_w_downpress[start : end]]

            if all(l == pl for l, pl in zip(dp_lines, pattern)):
                vals[line_idxs[start : end]] = True

    cs.df[col] = vals
    return


def stair5(cs: ChartStruct) -> None:
    df = cs.df
    lines = list(df['Line with active holds'].apply(lambda l: l.replace('`', '')))

    col = '__5-stair'
    if cs.singles_or_doubles() == 'doubles':
        cs.df[col] = [False] * len(cs.df)
        return

    lines_w_downpress = [line for idx, line in enumerate(lines)
                        if notelines.has_downpress(line)]
    line_idxs = [idx for idx, line in enumerate(lines)
                 if notelines.has_downpress(line)]
    
    patterns = [
        [
            '10000',
            '01000',
            '00100',
            '00010',
            '00001',
        ],
    ]
    flipped_patterns = []
    for p in patterns:
        flipped_patterns.append(p[::-1])
    patterns += flipped_patterns

    vals = np.array([False] * len(cs.df))

    # match patterns
    for pattern in patterns:
        for start in range(0, len(lines_w_downpress) - len(pattern)):
            end = start + len(pattern)
            # allow matching 2 as 1
            dp_lines = [l.replace('2', '1') for l in lines_w_downpress[start : end]]

            if all(l == pl for l, pl in zip(dp_lines, pattern)):
                vals[line_idxs[start : end]] = True

    cs.df[col] = vals
    return


def mid6_pad_transition(cs: ChartStruct) -> None:
    df = cs.df
    lines = list(df['Line with active holds'].apply(lambda l: l.replace('`', '')))
    limb_annots = list(df['Limb annotation'])

    col = '__mid6 pad transition'
    if cs.singles_or_doubles() == 'singles':
        cs.df[col] = [False] * len(cs.df)
        return

    lines_w_downpress = [line for idx, line in enumerate(lines)
                        if notelines.has_downpress(line)]
    line_idxs = [idx for idx, line in enumerate(lines)
                 if notelines.has_downpress(line)]
    
    patterns = [
        [
            '0010000000',
            '0001000000',
            '0000010000',
            '0000000100',
        ],
        [
            '0010000000',
            '0000100000',
            '0000001000',
            '0000000100',
        ],
    ]
    flipped_patterns = []
    for p in patterns:
        flipped_patterns.append(p[::-1])
    patterns += flipped_patterns

    vals = np.array([False] * len(cs.df))

    # match patterns
    for pattern in patterns:
        for start in range(0, len(lines_w_downpress) - len(pattern)):
            end = start + len(pattern)
            # allow matching 2 as 1
            dp_lines = [l.replace('2', '1') for l in lines_w_downpress[start : end]]
            idxs = line_idxs[start:end]
            limbs = [limb_annots[i] for i in idxs]

            matches_pattern = all(l == pl for l, pl in zip(dp_lines, pattern))
            alternates = all([l1 != l2 for l1, l2 in itertools.pairwise(limbs)])

            if matches_pattern and alternates:
                vals[line_idxs[start : end]] = True

    cs.df[col] = vals
    return


def coop_pad_transition(cs: ChartStruct) -> None:
    df = cs.df
    lines = list(df['Line with active holds'].apply(lambda l: l.replace('`', '')))

    col = '__co-op pad transition'
    if cs.singles_or_doubles() == 'singles':
        cs.df[col] = [False] * len(cs.df)
        return

    lines_w_downpress = [line for idx, line in enumerate(lines)
                        if notelines.has_downpress(line)]
    line_idxs = [idx for idx, line in enumerate(lines)
                 if notelines.has_downpress(line)]
    
    patterns = [
        [
            '1000000000',
            '0010000000',
            '0000100000',
            '0000010000',
            '0000000100',
            '0000000001',
        ],
        [
            '0100000000',
            '0010000000',
            '0001000000',
            '0000001000',
            '0000000100',
            '0000000010',
        ],
    ]
    flipped_patterns = []
    for p in patterns:
        flipped_patterns.append(p[::-1])
    patterns += flipped_patterns

    vals = np.array([False] * len(cs.df))

    # match patterns
    for pattern in patterns:
        for start in range(0, len(lines_w_downpress) - len(pattern)):
            end = start + len(pattern)
            # allow matching 2 as 1
            dp_lines = [l.replace('2', '1') for l in lines_w_downpress[start : end]]

            if all(l == pl for l, pl in zip(dp_lines, pattern)):
                vals[line_idxs[start : end]] = True

    cs.df[col] = vals
    return


"""
    Util
"""
def filter_short_runs(
    idxs: list[int] | set[int], 
    n: int, 
    filt_len: int
) -> list[bool]:
    """ From a list of indices, constructs a list of bools
        where an index is True only if it is part of a long run
    """
    flags = []
    idx_set = set(idxs)
    i = 0
    while i < n:
        if i not in idx_set:
            flags.append(False)
            i += 1
        else:
            # extend run
            j = i + 1
            while j in idx_set:
                j += 1

            # if run is long enough, add to flags
            if j - i >= filt_len:
                flags += [True]*(j-i)
            else:
                flags += [False]*(j-i)
            i = j
    return flags


def filter_run_by_num_downpress(
    df: pd.DataFrame, 
    bool_list: list[bool], 
    min_dp: int
) -> list[bool]:
    # Filter runs if they do not have enough downpresses
    ranges = bools_to_ranges(bool_list)
    dp_adjs = list(df['__num downpresses'].astype(bool).astype(int))
    filt = []
    for start, end in ranges:
        num_dp = sum(dp_adjs[start:end])
        if num_dp >= min_dp:
            filt += [i for i in range(start, end)]
    return filter_short_runs(filt, len(df), 1)


def bools_to_ranges(bools: list[bool]) -> list[tuple[int, int]]:
    """ List of bools -> list of idxs of True chains """
    ranges = []
    i = 0
    while i < len(bools):
        if bools[i]:
            j = i + 1
            while j < len(bools) and bools[j]:
                j += 1
            ranges.append((i, j))
            i = j + 1
        else:
            i += 1
    return ranges


"""
    Driver
"""
def annotate_skills(cs: ChartStruct) -> None:
    """ Adds or updates columns to `cs.df` for skills.
        Order of function calls matters -- some annotation functions
        require other annotations to be called first. 
    """
    # general skills
    drills(cs)
    jack(cs)
    footswitch(cs)
    run(cs)
    anchor_run(cs)
    brackets(cs)
    staggered_brackets(cs)
    doublestep(cs)

    # positions and specific patterns
    hands(cs)
    side3_singles(cs)
    mid4_doubles(cs)
    mid6_doubles(cs)
    splits(cs)
    hold_footswitch(cs)

    jump(cs)
    twists_90(cs)
    twists_over90(cs)

    stair5(cs)
    stair10(cs)
    yogwalk(cs)
    mid6_pad_transition(cs)
    coop_pad_transition(cs)
    hold_footslide(cs)

    # compound skills
    cs.df['__bracket run'] = cs.df['__bracket'] & cs.df['__run']
    cs.df['__bracket drill'] = cs.df['__bracket'] & cs.df['__drill']
    cs.df['__bracket jump'] = cs.df['__bracket'] & cs.df['__jump']
    cs.df['__bracket twist'] = cs.df['__bracket'] & (
        cs.df['__twist 90'] | cs.df['__twist over90']
    )
    cs.df['__run without twists'] = cs.df['__run'] & ~(
        cs.df['__twist 90'] | cs.df['__twist over90']
    )

    if args.setdefault('debug', False):
        import numpy as np
        df = cs.df
        import code; code.interact(local=dict(globals(), **locals()))

    return

