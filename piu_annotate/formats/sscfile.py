from collections import UserDict
from loguru import logger


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
    def from_string_and_header(string: str, header: HeaderSSC):
        """ Parse from string like
            #CHARTNAME:Mobile Edition;
            #STEPSTYPE:pump-single;
            #DESCRIPTION:S4 HIDDEN INFOBAR;
            #DIFFICULTY:Edit;
            #METER:4;

            Loads items from header first, then overwrites with stepchart string.
        """
        d = dict(header)
        d.update(parse_ssc_to_dict(string))
        return StepchartSSC(d)

    def __repr__(self) -> str:
        return '\n'.join(f'{k}: {v[:15].replace("\n", "")}' for k, v in self.data.items())

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
            self.is_train(),
            self.is_coop(),
            not self.has_4_4_timesig(),
            self.has_nonstandard_notes(),
        ])

    def is_ucs(self) -> bool:
        return 'UCS' in self.data['DESCRIPTION']
    
    def is_coop(self) -> bool:
        return 'COOP' in self.data['DESCRIPTION']

    def has_nonstandard_notes(self) -> bool:
        notes = self.data['NOTES'].replace('\n', '').replace(',', '')
        note_set = set(notes)
        ok = list('0123')
        for ok_char in list('0123'):
            if ok_char in note_set:
                note_set.remove(ok_char)
        return bool(len(note_set))

    def is_quest(self) -> bool:
        return 'QUEST' in self.data['DESCRIPTION']

    def is_hidden(self) -> bool:
        return 'HIDDEN' in self.data['DESCRIPTION']
    
    def is_train(self) -> bool:
        return 'TRAIN' in self.data['DESCRIPTION']

    def has_4_4_timesig(self) -> bool:
        timesig = self.data['TIMESIGNATURES']
        return all([
            timesig[-3:] == '4=4',
            '\n\n' not in timesig
        ])


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
        stepcharts = [StepchartSSC.from_string_and_header(section, header)
                      for section in sections[1:]]
        return header, stepcharts
    
    def validate(self) -> bool:
        return self.header.validate() and all(sc.validate() for sc in self.stepcharts)