"""
    Logic re note lines from .ssc file
"""
import re


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


def singlesdoubles(line: str) -> str:
    if len(line.replace('`', '')) == 5:
        return 'singles'
    elif len(line.replace('`', '')) == 10:
        return 'doubles'
    raise Exception(f'Bad line {line}')


def has_downpress(line: str) -> bool:
    return num_downpress(line) > 0


def has_one_arrow(line: str) -> bool:
    return line.count('1') == 1 and (line.count('0') in [4, 9])


def num_downpress(line: str) -> int:
    return sum(line.count(x) for x in list('12'))


def num_pressed(line: str) -> int:
    return sum(line.count(x) for x in list('124'))


def has_notes(line: str) -> bool:
    if '`' in line:
        return bool(set(line) != set(['`0']))
    else:
        return bool(set(line) != set(['0']))


def is_hold_release(line: str) -> bool:
    if '`' in line:
        return bool(set(line) != set(['`03']))
    else:
        return bool(set(line) != set(['03']))


def has_active_hold(line: str) -> bool:
    return '4' in line


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