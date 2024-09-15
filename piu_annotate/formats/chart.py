import pandas as pd
from tqdm import tqdm
from loguru import logger
import math
from dataclasses import dataclass

from .piucenterdf import PiuCenterDataFrame
from .sscfile import StepchartSSC
from .ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate.formats.jsplot import ArrowArt, HoldArt, ChartJsStruct
from piu_annotate.formats import notelines


def is_active_symbol(sym: str) -> bool: 
    return sym in set('1234')


def right_index(items: list[any], query: any) -> int:
    return len(items) - 1 - items[::-1].index(query)


@dataclass
class PredictionCoordinate:
    row_idx: int
    arrow_pos: int
    limb_idx: int

    def __hash__(self):
        return hash((self.row_idx, self.arrow_pos, self.limb_idx))


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

    def to_csv(self, filename: str):
        self.df.to_csv(filename)

    @staticmethod
    def from_piucenterdataframe(pc_df: PiuCenterDataFrame):
        """ Make ChartStruct from old PiuCenter d_annotate df.
        """
        dfs = pc_df.df[['Beat', 'Time', 'Line', 'Line with active holds']].copy()
        dfs['Limb annotation'] = pc_df.get_limb_annotations()
        return ChartStruct(dfs)
    
    @staticmethod
    def from_stepchart_ssc(stepchart_ssc: StepchartSSC):
        df, num_bad_lines = stepchart_ssc_to_chartstruct(stepchart_ssc)
        df['Line'] = [f'`{line}' for line in df['Line']]
        df['Line with active holds'] = [f'`{line}' for line in df['Line with active holds']]
        return ChartStruct(df)
    
    def validate(self) -> None:
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

    """
        Properties
    """
    def singles_or_doubles(self) -> str:
        """ Returns 'singles' or 'doubles' """
        line = self.df['Line'].iloc[0].replace('`', '')
        assert len(line) in [5, 10]
        if len(line) == 5:
            return 'singles'
        elif len(line) == 10:
            return 'doubles'

    """
        Prediction
    """
    def get_prediction_coordinates(self) -> list[PredictionCoordinate]:
        """ Find arrows to predict limb annotation: focus on 1/2.
            Returns list of (row_idx, arrow_pos, limb_idx)
        """
        pred_coords = []
        for idx, row in self.df.iterrows():
            line = row['Line'].replace('`', '')
            for arrow_pos, action in enumerate(line):
                if action in list('12'):
                    limb_idx = notelines.get_limb_idx_for_arrow_pos(
                        row['Line with active holds'],
                        arrow_pos
                    )
                    pred_coord = PredictionCoordinate(idx, arrow_pos, limb_idx)
                    pred_coords.append(pred_coord)
        return pred_coords

    def get_time_since_last_same_arrow_use(self) -> dict[PredictionCoordinate, float]:
        """ For each PredictionCoordinate, calculates the time since
            that arrow was last used by any limb (1 or 3).
        """
        pc_to_time = dict()
        last_time_used = [None] * 10
        for idx, row in self.df.iterrows():
            line = row['Line'].replace('`', '')
            time = row['Time']
            for arrow_pos, action in enumerate(line):
                if action in list('12'):
                    limb_idx = notelines.get_limb_idx_for_arrow_pos(
                        row['Line with active holds'],
                        arrow_pos
                    )
                    pred_coord = PredictionCoordinate(idx, arrow_pos, limb_idx)
                    if last_time_used[arrow_pos] is not None:
                        pc_to_time[pred_coord] = time - last_time_used[arrow_pos]
                    else:
                        pc_to_time[pred_coord] = -1
                if action in list('13'):
                    last_time_used[arrow_pos] = time
        return pc_to_time

    def get_previous_used_pred_coord_for_arrow(self) -> dict[int, int | None]:
        """ Compute dict mapping PredictionCoordinate index to the index of
            the PredictionCoordinate for the most recent previous time the
            arrow was used.
            If None, then None.
            Supports limb featurization that annotates the most recent
            (predicted) limb used for a given PredictionCoordinate. 
        """
        last_idx_used = [None] * 10
        pcs = self.get_prediction_coordinates()
        pc_to_prev_idx = dict()
        for idx, pc in enumerate(pcs):            
            pc_to_prev_idx[pc] = last_idx_used[pc.arrow_pos]
            last_idx_used[pc.arrow_pos] = idx
        return {idx: pc_to_prev_idx[pc] for idx, pc in enumerate(pcs)}

    def add_limb_annotations(
        self, 
        pred_coords: list[PredictionCoordinate],
        limb_annots: list[str],
        new_col: str,
    ) -> None:
        """ Populates `new_col` in df with `limb_annots` at `pred_coords`,
            for example predicted limb annotations.
        """
        assert len(pred_coords) == len(limb_annots)
        self.init_limb_annotations(new_col)

        # update
        for pred_coord, new_limb in zip(pred_coords, limb_annots):
            row_idx = pred_coord.row_idx
            limb_idx = pred_coord.limb_idx

            prev_annot = self.df.iloc[row_idx][new_col]
            new_annot = prev_annot[:limb_idx] + new_limb + prev_annot[limb_idx + 1:]
            self.df.loc[row_idx, new_col] = new_annot
        return

    def init_limb_annotations(self, new_col: str) -> None:
        """ Initializes limb annotations to `new_col` as all ? """
        limb_annots = []
        for idx, row in self.df.iterrows():
            line = row['Line with active holds']
            n_active_symbols = sum(is_active_symbol(s) for s in line)
            limb_annots.append('?' * n_active_symbols)
        self.df[new_col] = limb_annots
        return

    """
        Annotate
    """
    def annotate_time_since_downpress(self):
        """ Adds column `__time since prev downpress` to df
        """
        has_dps = [notelines.has_downpress(line) for line in self.df['Line']]
        recent_downpress_idx = None
        time_since_dp = []
        for idx, row in self.df.iterrows():
            
            if recent_downpress_idx is None:
                time_since_dp.append(-1)
            else:
                prev_dp_time = self.df.iloc[recent_downpress_idx]['Time']
                time_since_dp.append(row['Time'] - prev_dp_time)

            has_dp = has_dps[idx]
            if has_dp:
                recent_downpress_idx = idx

        self.df['__time since prev downpress'] = time_since_dp
        return

    def annotate_line_repeats_previous(self):
        """ Adds column `__line repeats previous` to df,
            which is True if current line is the same as previous or next line
            with downpress.
        """
        has_dps = [notelines.has_downpress(line) for line in self.df['Line']]
        lines = list(self.df['Line'])
        line_repeats = []
        for idx in range(len(self.df)):
            repeats = False

            prev = has_dps[:idx]
            prev_downpress_idx = None
            if any(prev):
                prev_downpress_idx = right_index(prev, True)
                prev_line_std = lines[prev_downpress_idx].replace('2', '1')
                curr_line_std = lines[idx].replace('2', '1')
                if prev_line_std == curr_line_std:
                    repeats = True

            line_repeats.append(repeats)

        self.df['__line repeats previous'] = line_repeats
        return

    def annotate_line_repeats_next(self):
        """ Adds column `__line repeats next` to df,
            which is True if current line is the same as previous or next line
            with downpress.
        """
        has_dps = [notelines.has_downpress(line) for line in self.df['Line']]
        lines = list(self.df['Line'])
        line_repeats = []
        for idx in range(len(self.df)):
            repeats = False
            
            next = has_dps[idx + 1:]
            next_downpress_idx = None
            if any(next):
                next_downpress_idx = idx + 1 + next.index(True)
                next_line_std = lines[next_downpress_idx].replace('2', '1')
                curr_line_std = lines[idx].replace('2', '1')
                if next_line_std == curr_line_std:
                    repeats = True

            line_repeats.append(repeats)

        self.df['__line repeats next'] = line_repeats
        return

    """
        Search
    """
    def __time_to_df_idx(self, query_time: float) -> list[int]:
        """ Finds df row idx by query_time """
        index = self.df.index[self.df['Time'].apply(lambda t: math.isclose(query_time, t))]
        idxs = index.to_list()
        if len(idxs) > 1:
            logger.warning(f'... matched multiple lines at {query_time=}')
        return idxs

    """
        Interaction with chart json: check match, update
    """
    def matches_chart_json(
        self, 
        chartjs: ChartJsStruct, 
        with_limb_annot: bool = True
    ) -> bool:
        self_cjs = ChartJsStruct.from_chartstruct(self)
        return self_cjs.matches(chartjs, with_limb_annot = with_limb_annot)

    def update_from_manual_json(self, chartjs: ChartJsStruct, verbose: bool = False) -> None:
        """ Updates ChartStruct given chart json, which can be manually
            annotated in step editor web app.
        """
        is_compatible = self.matches_chart_json(chartjs, with_limb_annot = False)
        if not is_compatible:
            logger.error('Tried to update chartstruct with non-matching chart json')
            return

        # update arrow arts
        num_arrow_arts_updated = 0
        for aa in chartjs.arrow_arts:
            df_idxs = self.__time_to_df_idx(aa.time)

            n_updates_made = 0
            for df_idx in df_idxs:
                update_made = self.__update_row_with_limb_annot(
                    df_idx, 
                    aa.arrow_pos, 
                    aa.limb,
                    expected_symbols = ['1'],
                )
                if update_made:
                    n_updates_made += 1
            
            num_arrow_arts_updated += n_updates_made
            if n_updates_made > 1:
                logger.warning(f'Used one arrow art to update multiple lines')
        
        if verbose:
            logger.info(f'Updated {num_arrow_arts_updated} arrows with limb annotations')

        # update hold arts
        num_hold_arts_updated = 0
        num_hold_art_lines_updated = 0
        for ha in chartjs.hold_arts:
            df_start_idx = self.__time_to_df_idx(ha.start_time)[0]
            df_end_idx = self.__time_to_df_idx(ha.end_time)[-1]

            n_lines_updated = 0
            for row_idx in range(df_start_idx, df_end_idx + 1):
                update_made = self.__update_row_with_limb_annot(
                    row_idx, 
                    ha.arrow_pos, 
                    ha.limb,
                    expected_symbols = ['2', '3', '4'],
                )

                if update_made:
                    n_lines_updated += 1
            if n_lines_updated:
                num_hold_art_lines_updated += n_lines_updated
                num_hold_arts_updated += 1
        
        if verbose:
            logger.success(f'Updated {num_hold_arts_updated} holds with limb annotations')
            logger.success(f'Updated {num_hold_art_lines_updated} lines updated with hold art limb annotations')
        return

    def __update_row_with_limb_annot(
        self, 
        row_idx: int, 
        new_arrow_pos: int, 
        new_limb: str,
        expected_symbols: list[str],
    ) -> bool:
        """ Update limb annotation for `row_idx` in self.df,
            to use `new_limb` for `new_arrow_pos`.

            Returns whether an update was made, or whether requested limb annotation
            was already in use (so no update made).
        """
        line = self.df.iloc[row_idx]['Line with active holds'].replace('`', '')

        if line[new_arrow_pos] not in expected_symbols:
            return False

        curr_limb_annot = self.df.iloc[row_idx]['Limb annotation']          

        if curr_limb_annot == '':
            n_active_symbols = sum(is_active_symbol(s) for s in line)
            new_annot = '?' * n_active_symbols
            self.df.loc[row_idx, 'Limb annotation'] = new_annot
            curr_limb_annot = new_annot

        arrow_pos_to_limb_annot_idx = {
            arrow_pos: sum(is_active_symbol(s) for s in line[:arrow_pos])
            for arrow_pos in range(len(line))
        }
        limb_annot_idx = arrow_pos_to_limb_annot_idx[new_arrow_pos]
        if curr_limb_annot[limb_annot_idx] != new_limb:
            curr_limb_annot_list = list(curr_limb_annot)
            curr_limb_annot_list[limb_annot_idx] = new_limb
            curr_limb_annot = ''.join(curr_limb_annot_list)
            self.df.loc[row_idx, 'Limb annotation'] = curr_limb_annot
            return True
        return False

    """
        Arts
    """
    def get_arrow_hold_arts(self) -> tuple[list[ArrowArt], list[HoldArt]]:
        arrow_arts = []
        hold_arts = []
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
                    elif sym == '2':
                        # add to active holds
                        if arrow_pos not in active_holds:
                            active_holds[arrow_pos] = (time, limb)
                        else:
                            logger.warning(f'WARNING: {arrow_pos=} in {active_holds=}')
                            # logger.warning(f'Skipping line ...')
                            continue
                        # try:
                        #     assert arrow_pos not in active_holds
                        # except:
                        #     print(f'{arrow_pos=} not in {active_holds=}')
                        #     import code; code.interact(local=dict(globals(), **locals()))
                    elif sym == '4':
                        # if limb changes, pop active hold and add new hold
                        active_limb = active_holds[arrow_pos][1]
                        if limb != active_limb:
                            start_time, start_limb = active_holds.pop(arrow_pos)
                            hold_arts.append(
                                HoldArt(arrow_pos, start_time, time, start_limb)
                            )
                            active_holds[arrow_pos] = (time, limb)
                    elif sym == '3':
                        # pop from active holds
                        start_time, start_limb = active_holds.pop(arrow_pos)
                        hold_arts.append(
                            HoldArt(arrow_pos, start_time, time, limb)
                        )

                    n_active_symbols_seen += 1
        return arrow_arts, hold_arts

    