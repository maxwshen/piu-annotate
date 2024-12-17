"""
    Logic re note lines from .ssc file
"""
import re
import itertools
import functools


def get_limb_for_arrow_pos(
    line_with_active_holds: str, 
    limb_annot: str,
    arrow_pos: int,
) -> str:
    """ Get limb from `limb_annot` for `arrow_pos` in `line_with_active_holds`
    """
    limb_idx = get_limb_idx_for_arrow_pos(line_with_active_holds, arrow_pos)
    return limb_annot[limb_idx]


def get_limb_idx_for_arrow_pos(
    line_with_active_holds: str,
    arrow_pos: int
) -> int:
    line = line_with_active_holds.replace('`', '')
    n_active_symbols_before = arrow_pos - line[:arrow_pos].count('0')
    return n_active_symbols_before


def panel_idx_to_action(line: str) -> dict[int, str]:
    idx_to_action = dict()
    for idx, action in enumerate(line):
        if action != '0':
            idx_to_action[idx] = action
    return idx_to_action


def get_downpress_limbs(line_with_active_holds: str, limb_annot: str) -> set[str]:
    """ Returns tuple of limbs ('l', 'r') used for all downpresses (1/2)
        in `line` according to `limb_annot`.
    """
    dp_limbs = [get_limb_for_arrow_pos(line_with_active_holds, limb_annot, arrow_pos)
                for arrow_pos, symbol in enumerate(line_with_active_holds)
                if symbol == '1' or symbol == '2']
    return set(dp_limbs)


def get_downpress_arrows(line: str) -> list[int]:
    line = line.replace('`', '')
    return [i for i, s in enumerate(line) if s in '12']


def get_leftmost_rightfoot_panel(line_with_active_holds: str, limb_annot: str) -> int | None:
    """ Returns index of left-most panel used by right foot for 1/2/4.
        Returns None if right foot is not used for 1/2/4.
    """
    ok_symbols = set(list('124'))
    panels = []
    for panel, symbol in enumerate(line_with_active_holds):
        if symbol in ok_symbols:
            limb = get_limb_for_arrow_pos(line_with_active_holds, limb_annot, panel)
            if limb == 'r':
                panels.append(panel)
    return min(panels) if panels else None


def get_rightmost_leftfoot_panel(line_with_active_holds: str, limb_annot: str) -> int | None:
    """ Returns index of right-most panel used by left foot for 1/2/4.
        Returns None if left foot is not used for 1/2/4.
    """
    ok_symbols = set(list('124'))
    panels = []
    for panel, symbol in enumerate(line_with_active_holds):
        if symbol in ok_symbols:
            limb = get_limb_for_arrow_pos(line_with_active_holds, limb_annot, panel)
            if limb == 'l':
                panels.append(panel)
    return max(panels) if panels else None


def is_90_twist(panel1: int, panel2: int) -> bool:
    """ Whether panels 1 and 2 form a 90-degree twist """
    ok_90_twists = [[0, 1], [3, 4], [5, 6], [8, 9]]
    return sorted([panel1, panel2]) in ok_90_twists


def is_over90_twist(rightfoot_panel: int, leftfoot_panel: int) -> bool:
    return all([
        rightfoot_panel < leftfoot_panel,
        not is_90_twist(rightfoot_panel, leftfoot_panel)
    ])


def is_close_twist(rightfoot_panel: int, leftfoot_panel: int) -> bool:
    if not is_over90_twist(rightfoot_panel, leftfoot_panel):
        return False
    # right foot is on a panel left of panel that left foot is on
    # neighbor dict maps left-side panel to panels neighboring, on right side
    neighbors = {
        0: [2],
        1: [2],
        2: [3, 4],
        3: [5, 6],
        4: [5, 6],
        5: [7],
        6: [7],
        7: [8, 9],
        8: [],
        9: [],
    }
    return leftfoot_panel in neighbors[rightfoot_panel]


def is_far_twist(rightfoot_panel: int, leftfoot_panel: int) -> bool:
    return all([
        is_over90_twist(rightfoot_panel, leftfoot_panel),
        not is_close_twist(rightfoot_panel, leftfoot_panel)
    ])


def singlesdoubles(line: str) -> str:
    if len(line.replace('`', '')) == 5:
        return 'singles'
    elif len(line.replace('`', '')) == 10:
        return 'doubles'
    raise Exception(f'Bad line {line}')


def has_downpress(line: str) -> bool:
    return num_downpress(line) > 0


def has_one_1(line: str) -> bool:
    return line.count('1') == 1 and (line.count('0') in [4, 9])


def has_one_2(line: str) -> bool:
    return line.count('2') == 1 and (line.count('0') in [4, 9])


def has_one_3(line: str) -> bool:
    return line.count('3') == 1 and (line.count('0') in [4, 9])


def has_center_arrow(line: str) -> bool:
    if '`' in line:
        line = line.replace('`', '')
    has_p2_center = False
    if len(line) == 10:
        has_p2_center = (line[7] in list('12'))
    has_p1_center = line[2] in list('12')
    return has_p1_center or has_p2_center


def num_downpress(line: str) -> int:
    return sum(line.count(x) for x in list('12'))


def num_pressed(line: str) -> int:
    return sum(line.count(x) for x in list('124'))


def is_hold_release(line: str) -> bool:
    return bool(set(line.replace('`', '')) == set(list('03')))


def is_hold_start(line: str) -> bool:
    return bool(set(line.replace('`', '')) == set(list('02')))


def has_active_hold(line: str) -> bool:
    return '4' in line


def staggered_bracket(line1: str, line2: str) -> bool:
    """ Whether `line1` and `line2` can be executed as a staggered bracket.
        Only considers 1 in lines.
    """
    f = lambda c1, c2: '1' if bool(c1 == '1' or c2 == '1') else '0'
    merged_line = ''.join([f(c1, c2) for c1, c2 in zip(line1, line2)])
    return line_is_bracketable(merged_line)


bracketable_lines = set([
    '10100', '01100', '00110', '00101',
    '1010000000',
    '0110000000',
    '0011000000',
    '0010100000',
    '0000010100',
    '0000001100',
    '0000000110',
    '0000000101',
    '0000110000',
    '0001001000',
])
def frac_two_arrows_bracketable(lines: list[str]) -> float:
    tlines = [x.replace('2', '1') for x in lines]
    two_arrow_lines = [line for line in tlines if line.count('1') == 2]
    num_bracketable = sum([l in bracketable_lines for l in two_arrow_lines])
    return num_bracketable / len(two_arrow_lines)


bracketable_arrow_positions = [
    [0, 2], [1, 2], [2, 3], [2, 4],
    [5, 7], [6, 7], [7, 8], [7, 9],
    [4, 5], [3, 6],
]
quads = [(b1, b2) for b1, b2 in itertools.combinations(bracketable_arrow_positions, 2)
         if len(set(b1 + b2)) == 4]
def one_foot_multihit_possible(arrow_positions: list[int]) -> bool:
    """ Returns whether one foot can be used to hit all `arrow_positions`
        at the same time.
    """
    if len(arrow_positions) > 2:
        return False
    if len(arrow_positions) <= 1:
        return True
    return sorted(arrow_positions) in bracketable_arrow_positions


def multihit_to_valid_feet(arrow_positions: list[int]) -> list[tuple[int]]:
    """ Given `arrow_positions`, returns a list of valid feet assignments
        represented as a tuple of ints.
        Each output tuple has the same length as `arrow_positions`, and has elements
        0 = left, 1 = right, for the i-th arrow position.

        This logic can be used to detect hands: hands are implied to be used
        if the limb annotation for a multihit is not in this function's set of
        valid feet assignments.
    """
    assert arrow_positions == sorted(arrow_positions), 'Must be sorted'
    if len(arrow_positions) == 2:
        ok = [(0, 1), (1, 0)]
        if arrow_positions in bracketable_arrow_positions:
            ok += [(0, 0), (1, 1)]
        return ok
    if len(arrow_positions) == 3:
        assignments = []
        for b in bracketable_arrow_positions:
            if all(pos in arrow_positions for pos in b):
                assign = [0, 0, 0]
                for pos in b:
                    assign[arrow_positions.index(pos)] = 1
                flipped = [1 - x for x in assign]
                assignments += [tuple(assign), tuple(flipped)]
        return assignments
    if len(arrow_positions) == 4:
        # Valid quad must be two brackets
        for b1, b2 in quads:
            if sorted(b1 + b2) == arrow_positions:
                assign = [0, 0, 0, 0]
                for pos in b1:
                    assign[arrow_positions.index(pos)] = 1
                flipped = [1 - x for x in assign]
                return [tuple(assign), tuple(flipped)]
        return []
    return []


@functools.lru_cache
def line_is_bracketable(line: str) -> bool:
    """ Returns whether `line` is bracketable, counting all downpresses (1-4).
        If only two downpresses, returns whether line can be bracketed with one foot.
        If three+ downpresses, returns whether line can be executed with one or more brackets
        with two feet only.
    """
    line = line.replace('`', '')
    downpress_idxs = [i for i, x in enumerate(line) if x != '0']
    if len(downpress_idxs) == 2:
        arrow_positions = sorted(downpress_idxs)
        return bool(arrow_positions in bracketable_arrow_positions)
    elif len(downpress_idxs) > 2:
        return bool(len(multihit_to_valid_feet(downpress_idxs)))
    return False


def add_active_holds(line: str, active_hold_idxs: set[str]) -> str:
    """ Add active holds into line as '4'. 01000 -> 01040 """
    aug_line = list(line)
    for panel_idx in active_hold_idxs:
        if aug_line[panel_idx] == '0':
            aug_line[panel_idx] = '4'
        elif aug_line[panel_idx] in ['1', '2']:
            raise Exception('Error: Tried to place active hold 4 onto 1/2')
    return ''.join(aug_line)


def parse_line(line: str) -> str:
    """ Parse notes in stepmania [0/1/2/3] and stepf2 {2|n|1|0} format.
        Return line in standardized format using note types 0/1/2/3.
        https://github.com/rhythmlunatic/stepmania/wiki/Note-Types#stepf2-notes
        https://github.com/stepmania/stepmania/wiki/Note-Types
        Handle lines like:
            0000F00000
            00{2|n|1|0}0000000    
            0000{M|n|1|0} -> 0
    """
    ws = re.split('{|}', line)
    nl = ''
    for w in ws:
        if '|' not in w:
            nl += w
        else:
            [note_type, attribute, fake_flag, reserved_flag] = w.split('|')
            if fake_flag == '1':
                nl += '0'
            else:
                nl += note_type
    line = nl

    replace = {
        'F': '0',
        'M': '0',
        'K': '0',
        'V': '0',
        'v': '0',
        'S': '0',
        's': '0',
        'E': '0',
        'I': '1',
        '4': '2',
        '6': '2',
        'L': '1',
    }
    line = line.translate(str.maketrans(replace))

    if any(x not in set(list('01234')) for x in line):
        raise ValueError(f'Bad symbol found in {line}')
    return line


def excel_refmt(string):
    return f'`{string}'


def hd_to_fulldouble(line: str):
    return '00' + line + '00'