"""
    Annotate a ChartStruct with skills columns
"""
import math
import re
import itertools
from loguru import logger
import pandas as pd

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines

# used to filter short runs
# for example, if a run is shorter than 8 notes, do not label as run
MIN_DRILL_LEN = 5
MIN_RUN_LEN = 8
MIN_ANCHOR_RUN_LEN = 7

# used for side3 singles, mid4 doubles, etc.
MIN_POSITION_NOTES_LEN = 8

# faster than 13 nps
STAGGERED_BRACKET_TIME_SINCE_THRESHOLD = 1/13


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
            not drills[j],
            math.isclose(ts[i], ts[j])
        ]
        if all(crits):
            idxs.add(i-1)
            idxs.add(i)

    cs.df['__run'] = filter_short_runs(idxs, len(df), MIN_RUN_LEN)  
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
    df = cs.df
    lines = cs.get_lines()
    limb_annots = list(df['Limb annotation'])
    staggered_brackets = list(df['__staggered bracket'])
    jacks = list(df['__jack'])

    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            notelines.has_one_1(lines[i]),
            notelines.has_one_1(lines[j]),
            limb_annots[i] == limb_annots[j],
            not staggered_brackets[j],
            not jacks[j],
        ]
        res.append(all(crits))

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
    for i, j in itertools.pairwise(range(len(df))):
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

    cs.df['__twist 90'] = res
    return


def twists_over90(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
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
                    notelines.is_over90_twist(leftmost_r_panel, rightmost_l_panel)
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
                    notelines.is_over90_twist(leftmost_r_panel, rightmost_l_panel)
                ])

        res.append(is_twist)

    cs.df['__twist over90'] = res
    return


def side3_singles(cs: ChartStruct) -> None:
    df = cs.df
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

    def has_split(line: str) -> bool:
        return all([
            any(x in line[:2] for x in list('12')),
            any(x in line[-2:] for x in list('12')),
        ])

    cs.df['__split'] = [has_split(line) for line in lines]
    return


def jack(cs: ChartStruct) -> None:
    df = cs.df
    lines = cs.get_lines_with_active_holds()
    limb_annots = list(df['Limb annotation'])

    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            lines[j] == lines[i],
            notelines.num_downpress(lines[i]) == 1,
            limb_annots[i] == limb_annots[j],
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

    jump(cs)
    twists_90(cs)
    twists_over90(cs)
    return


def check_hands(cs: ChartStruct):
    """ Use to detect unexpected hands in `cs`, which can indicate prediction errors
    """
    ok_hands = [
        'Ugly_Dee_-_Banya_Production_D18_ARCADE',
        'Ugly_Dee_-_Banya_Production_D17_ARCADE',
        'Come_to_Me_-_Banya_S17_INFOBAR_TITLE_ARCADE',
        'ESCAPE_-_D_AAN_D26_ARCADE',
        'Chimera_-_YAHPP_S23_ARCADE',
        'Chimera_-_YAHPP_D26_ARCADE',
        'Come_to_Me_-_Banya_S13_ARCADE',
        'Naissance_2_-_BanYa_D16_ARCADE',
        'Achluoias_-_D_AAN_D26_ARCADE',
        'Jump_-_BanYa_S16_ARCADE',
        'Gun_Rock_-_Banya_Production_S20_ARCADE',
        'Uh-Heung_-_DKZ_S22_ARCADE',
        'Love_is_a_Danger_Zone_2_-_FULL_SONG_-_-_Yahpp_S20_FULLSONG',
        'Love_is_a_Danger_Zone_2_-_FULL_SONG_-_-_Yahpp_D21_FULLSONG',
        'Love_is_a_Danger_Zone_2_Try_To_B.P.M_-_BanYa_D23_INFOBAR_TITLE_REMIX',
        'Love_is_a_Danger_Zone_2_Try_To_B.P.M_-_BanYa_S21_REMIX',
        'Hi-Bi_-_BanYa_D21_ARCADE',
        'Fire_Noodle_Challenge_-_Memme_S23_REMIX',
        'Slam_-_Novasonic_S18_INFOBAR_TITLE_ARCADE',
        'Bee_-_BanYa_D23_ARCADE',
        'Bee_-_BanYa_S17_INFOBAR_TITLE_ARCADE',
        'Another_Truth_-_Novasonic_D19_ARCADE',
    ]

    hands(cs)
    has_hands = any(cs.df['__hands'])

    if has_hands:
        shortname = cs.metadata["shortname"]
        if shortname not in ok_hands:
            logger.debug(f'Found hands')
            logger.debug(f'{cs.metadata["shortname"]}')
            print(cs.df[cs.df['__hands']])
            import code; code.interact(local=dict(globals(), **locals()))
    return
