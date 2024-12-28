
import json

from piu_annotate.formats.sscfile import StepchartSSC

class ArrowEclipseChartInfo:
    def __init__(self, data: dict[str, any]):
        """ d in format like:
            {
            "id": "78a5a68c-6510-43bb-8433-4545ed395f2b",
            "type": "Double",
            "shorthand": "D20",
            "level": 20,
            "song": {
                "name": "Baroque Virus - FULL SONG -",
                "type": "FullSong",
                "imagePath": "https://piuimages.arroweclip.se/songs/acef9bbe-4cbe-40ef-b0e0-00084836a7ea.png"
            }
            },
        """
        self.data = data

    def matches_song_name(self, sc: StepchartSSC) -> bool:
        return self.data['song']['name'] == sc['TITLE']

    def matches_stepchart_ssc(self, sc: StepchartSSC) -> bool:
        return all([
            self.data['song']['name'] == sc['TITLE'],
            self.data['song']['type'].lower() == sc['SONGTYPE'].lower(),
            self.match_stepstype(sc['STEPSTYPE']),
            int(self.data['level']) == int(sc['METER']),
        ])

    def matches_stepchart_ssc_partial(self, sc: StepchartSSC) -> bool:
        """ Matches on songtype (shortcut/arcade/remix),
            stepstype (double/single), and level.
            Does not match on song title.
        """
        return all([
            self.data['song']['type'].lower() == sc['SONGTYPE'].lower(),
            self.match_stepstype(sc['STEPSTYPE']),
            int(self.data['level']) == int(sc['METER']),
        ])

    def match_stepstype(self, ssc_stepstype: str) -> bool:
        """ ArrowEclipse type: {'Single': 2356, 'Double': 1490, 'CoOp': 126}
            ssc_stepstype
            {'pump-single': 7694, 'pump-double': 5206, 'pump-halfdouble': 897, 'pump-routine': 216, 'pump-couple': 136}
        """
        mapping = {
            'pump-single': 'Single',
            'pump-double': 'Double',
            'pump-halfdouble': 'Double',
            'pump-couple': 'CoOp',
        }
        return self.data['type'] == mapping.get(ssc_stepstype, ssc_stepstype)

    def is_coop(self) -> bool:
        return self.data['type'] == 'CoOp'

    def level(self) -> int:
        return self.data['level']

    def is_singles(self) -> int:
        return self.data['type'] == 'Single'

    def is_doubles(self) -> int:
        return self.data['type'] == 'Double'

    def __repr__(self) -> str:
        return '\n'.join(f'{k}: {v}' for k, v in self.data.items())

    def shortname(self) -> str:
        return ' '.join([
            self.data['song']['name'],
            self.data['song']['type'],
            self.data['shorthand'],
        ])

class ArrowEclipseStepchartListJson:
    def __init__(self, json_file: str):
        """ Loads json in format like:

            {
            "page": 1,
            "count": 3972,
            "totalResults": 3972,
            "results": [
                {
                "id": "78a5a68c-6510-43bb-8433-4545ed395f2b",
                "type": "Double",
                "shorthand": "D20",
                "level": 20,
                "song": {
                    "name": "Baroque Virus - FULL SONG -",
                    "type": "FullSong",
                    "imagePath": "https://piuimages.arroweclip.se/songs/acef9bbe-4cbe-40ef-b0e0-00084836a7ea.png"
                }
                },
        """
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.d = json.load(f)
        self.charts = [ArrowEclipseChartInfo(rd) for rd in self.d['results']]

    def __len__(self) -> int:
        return len(self.charts)