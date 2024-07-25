import pandas as pd
from tqdm import tqdm
from loguru import logger

from .piucenterdf import PiuCenterDataFrame
from .sscfile import StepChartSSC
from .ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate.formats.jsplot import ArrowArt, HoldArt


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
        # self.validate()

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
    def from_stepchart_ssc(stepchart_ssc: StepChartSSC):
        df = stepchart_ssc_to_chartstruct(stepchart_ssc)
        return ChartStruct(df)
    
    def validate(self):
        """ Validate format -- see docstring. """
        # logger.debug('Verifying ChartStruct ...')
        cols = ['Beat', 'Time', 'Line', 'Line with active holds', 'Limb annotation']
        for col in cols:
            assert col in self.df.columns
        assert self.df['Beat'].dtype == float
        assert self.df['Time'].dtype == float

        line_symbols = set(list('`0123'))
        line_w_active_holds_symbols = set(list('`01234'))
        limb_symbols = set(list('lreh?'))

        for idx, row in self.df.iterrows():
        # for idx, row in tqdm(self.df.iterrows(), total = len(self.df)):
            line = row['Line']
            lineah = row['Line with active holds']
            limb_annot = row['Limb annotation']

            # check starts with `
            assert line[0] == '`'
            assert lineah[0] == '`'

            # check lengths
            assert len(line) == len(lineah)
            assert len(line) == 6 or len(line) == 11
            n_active_symbols = len(lineah[1:].replace('0', ''))
            try:
                assert len(limb_annot) == 0 or len(limb_annot) == n_active_symbols
            except:
                logger.error('error')
                import code; code.interact(local=dict(globals(), **locals()))

            # check characters
            assert set(line).issubset(line_symbols)
            assert set(lineah).issubset(line_w_active_holds_symbols)
            assert set(limb_annot).issubset(limb_symbols)
        return

    def get_arrow_hold_arts(self) -> tuple[list[ArrowArt], list[HoldArt]]:
        arrow_arts = []
        hold_arts = []
        is_active_symbol = lambda sym: sym in list('1234')
        get_limb = lambda limbs, idx: limbs[idx] if len(limbs) > 0 else '?'
        active_holds = {}   # arrowpos: (time start, limb)
        for _, row in self.df.iterrows():
            line = row['Line with active holds']
            limb_annot = row['Limb annotation']
            time = row['Time']

            n_active_symbols_seen = 0
            for arrow_pos, sym in enumerate(line[1:]):
                if is_active_symbol(sym):
                    limb = get_limb(limb_annot, n_active_symbols_seen)

                    if sym == '1':
                        arrow_arts.append(ArrowArt(arrow_pos, time, limb))
                    if sym == '2':
                        # add to active holds
                        if arrow_pos in active_holds:
                            logger.warning(f'WARNING: {arrow_pos=} in {active_holds=}')
                            logger.warning(f'Skipping line ...')
                            continue
                        # try:
                        #     assert arrow_pos not in active_holds
                        # except:
                        #     print(f'{arrow_pos=} not in {active_holds=}')
                        #     import code; code.interact(local=dict(globals(), **locals()))
                        active_holds[arrow_pos] = (time, limb)
                    if sym == '4':
                        # if limb changes, pop active hold and add new hold
                        active_limb = active_holds[arrow_pos][1]
                        if limb != active_limb:
                            start_time, start_limb = active_holds.pop(arrow_pos)
                            hold_arts.append(
                                HoldArt(arrow_pos, start_time, time, start_limb)
                            )
                            active_holds[arrow_pos] = (time, limb)
                    if sym == '3':
                        # pop from active holds
                        start_time, start_limb = active_holds.pop(arrow_pos)
                        hold_arts.append(
                            HoldArt(arrow_pos, start_time, time, limb)
                        )

                    n_active_symbols_seen += 1
        return arrow_arts, hold_arts

    