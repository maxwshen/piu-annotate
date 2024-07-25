"""
    Logic re note lines from .ssc file
"""
import re, functools

idx_to_panel = {
    0: 'p1,1',
    1: 'p1,7',
    2: 'p1,5',
    3: 'p1,9',
    4: 'p1,3',
    5: 'p2,1',
    6: 'p2,7',
    7: 'p2,5',
    8: 'p2,9',
    9: 'p2,3',
}
panel_to_idx = {panel: idx for idx, panel in idx_to_panel.items()}


def panel_to_action(line: str) -> dict[str, str]:
    panel_to_action = dict()
    for idx, action in enumerate(line):
        panel = idx_to_panel[idx]
        if action != '0':
            panel_to_action[panel] = action
    return panel_to_action


def singlesdoubles(line: str) -> str:
    if len(line.replace('`', '')) == 5:
        return 'singles'
    elif len(line.replace('`', '')) == 10:
        return 'doubles'
    raise Exception(f'Bad line {line}')


def has_downpress(line: str) -> bool:
    return num_downpress(line) > 0


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


def add_active_holds(
    line: str, 
    active_holds: set[str], 
) -> str:
    """
        Add active holds into line as '4'
        01000 -> 01040
    """
    aug_line = list(line)
    for panel in active_holds:
        idx = panel_to_idx[panel]
        if aug_line[idx] == '0':
          aug_line[idx] = '4'
        elif aug_line[idx] == '1':
            # print('Error: Tried to place active hold 4 onto 1')
            # import code; code.interact(local=dict(globals(), **locals()))
            raise Exception('Error: Tried to place active hold 4 onto 1')
    return ''.join(aug_line)


@functools.lru_cache(maxsize = None)
def parse_line(line: str) -> str:
    """
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
            [note_type, attribute, fake_flag, x_offset] = w.split('|')
            if fake_flag == '1':
                nl += '0'
            else:
                if attribute in ['v', 'h']:
                    nl += '0'
                else:
                    nl += note_type
    line = nl

    # F is fake note
    replace = {
        'F': '0',
        'M': '0',
        'K': '0',
        'L': '0',
        'V': '0',
        'v': '0',
        's': '0',
        'S': '0',
        'E': '0',
        'I': '1',
        '4': '2',
        '6': '2',
    }
    line = line.translate(str.maketrans(replace))
    return line


def excel_refmt(string):
    return f'`{string}'


def hd_to_fulldouble(line: str):
    return '00' + line + '00'