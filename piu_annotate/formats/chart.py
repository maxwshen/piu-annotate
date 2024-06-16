import pandas as pd
from tqdm import tqdm
from loguru import logger

from .piucenterdf import PiuCenterDataFrame


class ChartStruct:
    def __init__(self, df: pd.DataFrame):
        """ Primary dataframe representation of a chart.
            One row per "line"

            Columns
            -------
            Beat: float
            Time: float
            Line
                concatenated string of 0, 1, 2, 3, where 0 = no note,
                1 = arrow, 2 = hold start, 3 = hold end.
                Must start with `
                Length must be 6 (singles) or 11 (doubles).
            Line with active holds:
                concatenated string of "actions" 0, 1, 2, 3, 4,
                where 0 = no note, 1 = arrow, 2 = hold start,
                3 = hold end, 4 = active hold.
                Must start with `
                Length must be 6 (singles) or 11 (doubles).
            Limb annotation (optional, can be incomplete):
                Concatenated string of l (left foot), r (right foot),
                e (either foot), h (either hand), ? (unknown).
                Length must be:
                    = Number of non-0 symbols in `Line with active holds`:
                      limb per symbol in same order.
                    = 0: (blank) equivalent to ? * (n. non-0 symbols)
        
            Intended features
            - Load from old d_annotate format (with foot annotations)
            - Load from .ssc file (with or without foot annotations)
            - Convert ChartStruct to ChartJSStruct for visualization
            - Featurize ChartStruct, for ML annotation of feet
        """
        self.df = df
        self.validate()

    @staticmethod
    def from_file(csv_file: str):
        return ChartStruct(pd.read_csv(csv_file))

    @staticmethod
    def from_piucenterdataframe(pc_df: PiuCenterDataFrame):
        """ Make ChartStruct from old PiuCenter d_annotate df.
        """
        dfs = pc_df.df[['Beat', 'Time', 'Line', 'Line with active holds']].copy()
        dfs['Limb annotation'] = pc_df.get_limb_annotations()
        return ChartStruct(dfs)
    
    @classmethod
    def from_ssc(ssc_file: str):
        raise NotImplementedError
    
    def validate(self):
        """ Validate format -- see docstring. """
        logger.debug('Verifying ChartStruct ...')
        cols = ['Beat', 'Time', 'Line', 'Line with active holds', 'Limb annotation']
        for col in cols:
            assert col in self.df.columns
        assert self.df['Beat'].dtype == float
        assert self.df['Time'].dtype == float

        line_symbols = set(list('`0123'))
        line_w_active_holds_symbols = set(list('`01234'))
        limb_symbols = set(list('lreh?'))

        for idx, row in tqdm(self.df.iterrows(), total = len(self.df)):
            line = row['Line']
            lineah = row['Line with active holds']
            limb_annot = row['Limb annotation']

            # check starts with `
            assert line[0] == '`'
            assert lineah[0] == '`'

            # check lengths
            assert len(line) == len(lineah)
            length = len(line)
            assert len(line) == 6 or len(line) == 11
            n_active_symbols = len(lineah[1:].replace('0', ''))
            assert len(limb_annot) == 0 or len(limb_annot) == n_active_symbols

            # check characters
            assert set(line).issubset(line_symbols)
            assert set(lineah).issubset(line_w_active_holds_symbols)
            assert set(limb_annot).issubset(limb_symbols)
        return
