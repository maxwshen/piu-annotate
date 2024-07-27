import os
from collections import UserDict
from loguru import logger
from pathlib import Path


def parse_ssc_to_dict(string: str) -> dict[str, str]:
    """ Parses string into `#{KEY}:{VALUE}; dict """
    kvs = string.split(';')

    d = dict()
    for kv in kvs:
        if ':' in kv:
            [k, *v] = kv.strip().split(':')
            v = ':'.join(v)
            assert k[0] == '#'
            k = k[1:]
            d[k] = v
    return d


class HeaderSSC(UserDict):
    def __init__(self, d: dict):
        """ Dict for header in ssc file """
        super().__init__(d)

    @staticmethod
    def from_string(string: str):
        """ Parse from string like
            #VERSION:0.81;
            #TITLE:Galaxy Collapse;
            #SUBTITLE:;
            #ARTIST:Kurokotei;
            #TITLETRANSLIT:;
        """
        return HeaderSSC(parse_ssc_to_dict(string))

    def validate(self) -> bool:
        required_keys = [
            'TITLE',
            'SONGTYPE',
        ]
        return all(key in self.data for key in required_keys)


class StepchartSSC(UserDict):
    def __init__(self, d: dict):
        """ Dict for stepchart in ssc file """
        super().__init__(d)

    @staticmethod
    def from_string_and_header(
        string: str, 
        header: HeaderSSC,
        ssc_file: str,
        pack: str,
    ):
        """ Parse from string like
            #CHARTNAME:Mobile Edition;
            #STEPSTYPE:pump-single;
            #DESCRIPTION:S4 HIDDEN INFOBAR;
            #DIFFICULTY:Edit;
            #METER:4;

            Loads items from header first, then overwrites with stepchart string.
        """
        d = {'ssc_file': ssc_file, 'pack': pack}
        d.update(dict(header))
        d.update(parse_ssc_to_dict(string))
        return StepchartSSC(d)

    @staticmethod
    def from_file(filename: str):
        with open(filename, 'r') as f:
            string = '\n'.join(f.readlines())
        return StepchartSSC(parse_ssc_to_dict(string))

    def to_file(self, filename: str) -> None:
        """ Writes to file """
        Path(os.path.dirname(filename)).mkdir(parents = True, exist_ok = True)
        string_repr = '\n'.join(f'#{k}:{v.replace("\n\n", "\n")};' for k, v in self.data.items())
        with open(filename, 'w') as f:
            f.write(string_repr)
        return

    def __repr__(self) -> str:
        return '\n'.join(f'{k}: {v[:15].replace("\n", "")}' for k, v in self.data.items())

    def shortname(self) -> str:
        shortname = '_'.join([
            f'{self.data["TITLE"]} - {self.data["ARTIST"]}',
            self.data["DESCRIPTION"],
            self.data["SONGTYPE"],
        ]) 
        return shortname.replace(' ', '_')

    def validate(self) -> bool:
        required_keys = [
            'STEPSTYPE',
            'DESCRIPTION',
            'BPMS',
            'METER',
            'NOTES',
        ]
        return all(key in self.data for key in required_keys)

    """
        Attributes
    """
    def is_nonstandard(self) -> bool:
        return any([
            self.is_ucs(),
            self.is_quest(),
            self.is_hidden(),
            self.is_infinity(),
            self.is_train(),
            self.is_coop(),
            self.has_99_meter(),
            not self.has_4_4_timesig(),
            self.has_nonstandard_notes(),
            not self.standard_stepstype(),
            not self.standard_songtype(),
        ])

    def describe(self) -> dict[str, any]:
        d = {k: v[:15].replace('\n', ';') for k, v in self.data.items()}
        d.update({
            'nonstandard': self.is_nonstandard(),
            'ucs': self.is_ucs(),
            'quest': self.is_quest(),
            'hidden': self.is_hidden(),
            'infinity': self.is_infinity(),
            'train': self.is_train(),
            'coop': self.is_coop(),
            'meter 99': self.has_99_meter(),
            'not 4/4': not self.has_4_4_timesig(),
            'nonstandard notes': self.has_nonstandard_notes(),
            'standard stepstype': self.standard_stepstype(),
            'standard songtype': self.standard_songtype(),
        })
        return d

    def has_99_meter(self) -> bool:
        return self.data['METER'] == '99'

    def is_ucs(self) -> bool:
        return 'UCS' in self.data['DESCRIPTION'].upper()
    
    def is_coop(self) -> bool:
        return 'COOP' in self.data['DESCRIPTION'].upper()

    def has_nonstandard_notes(self) -> bool:
        """ Has notes other than 0, 1, 2, 3 """
        notes = self.data['NOTES'].replace('\n', '').replace(',', '')
        note_set = set(notes)
        for ok_char in list('0123'):
            if ok_char in note_set:
                note_set.remove(ok_char)
        return bool(len(note_set))

    def is_quest(self) -> bool:
        return 'QUEST' in self.data['DESCRIPTION'].upper()

    def is_hidden(self) -> bool:
        return 'HIDDEN' in self.data['DESCRIPTION'].upper()

    def is_infinity(self) -> bool:
        return 'INFINITY' in self.data['DESCRIPTION'].upper()

    def is_train(self) -> bool:
        return 'TRAIN' in self.data['DESCRIPTION'].upper()

    def has_4_4_timesig(self) -> bool:
        timesig = self.data['TIMESIGNATURES']
        return all([
            timesig[-3:] == '4=4',
            '\n\n' not in timesig
        ])
    
    def standard_stepstype(self) -> bool:
        ok = ['pump-single', 'pump-double', 'pump-halfdouble']
        return self.data['STEPSTYPE'] in ok

    def standard_songtype(self) -> bool:
        ok = ['ARCADE', 'FULLSONG', 'REMIX', 'SHORTCUT']
        return self.data['SONGTYPE'] in ok


class SongSSC:
    def __init__(self, ssc_file: str, pack: str):
        """ Parses song .ssc file into 1 HeaderSSC and multiple StepchartSSC objects.

            .ssc file format resources
            https://github.com/stepmania/stepmania/wiki/ssc
            https://github.com/stepmania/stepmania/wiki/sm
        """
        self.ssc_file = ssc_file
        self.pack = pack

        header, stepcharts = self.parse_song_ssc_file(self.ssc_file)
        self.header = header
        self.stepcharts = stepcharts
        self.validate()

    def parse_song_ssc_file(
        self, 
        file_lines: str
    ) -> tuple[HeaderSSC, list[StepchartSSC]]:
        """ Parse song ssc file. Sections in file are delineated by #NOTEDATA:; """
        with open(self.ssc_file, 'r') as f:
            file_lines = f.readlines()

        all_lines = '\n'.join(file_lines)
        sections = all_lines.split('#NOTEDATA:;')

        header = HeaderSSC.from_string(sections[0])
        stepcharts = [StepchartSSC.from_string_and_header(section, header, self.ssc_file, self.pack)
                      for section in sections[1:]]
        return header, stepcharts
    
    def validate(self) -> bool:
        return self.header.validate() and all(sc.validate() for sc in self.stepcharts)