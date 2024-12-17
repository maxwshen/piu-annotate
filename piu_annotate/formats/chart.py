import pandas as pd
from tqdm import tqdm
from loguru import logger
import math
from dataclasses import dataclass
import numpy as np
import json
import os
from collections import defaultdict
import functools

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.sscfile import StepchartSSC
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate.formats.jsplot import ArrowArt, HoldArt, ChartJsStruct
from piu_annotate.formats import notelines


def is_active_symbol(sym: str) -> bool: 
    return sym in set('1234')


def right_index(items: list[any], query: any) -> int:
    """ Gets the index of the right-most item in `items` matching `query`. """
    return len(items) - 1 - items[::-1].index(query)


@dataclass
class ArrowCoordinate:
    row_idx: int
    arrow_pos: int
    limb_idx: int
    is_downpress: bool
    line_with_active_holds: str

    def __hash__(self):
        return hash((self.row_idx, self.arrow_pos, self.limb_idx, self.is_downpress))


class ChartStruct:
    def __init__(self, df: pd.DataFrame, source_file: str = ''):
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
            Metadata (optional)
                One entry only; json text of metadata dict.
                    
            Features and uses
            - Load from old d_annotate format (with foot annotations)
            - Load from .ssc file (with or without foot annotations)
            - Convert ChartStruct to ChartJSStruct for visualization
            - Featurize ChartStruct, for ML annotation of feet
        """
        self.df = df
        self.source_file = source_file
        # self.validate()
        if 'Metadata' in self.df.columns:
            self.metadata = json.loads(self.df['Metadata'][0])
        else:
            self.metadata = dict()

    @staticmethod
    def from_file(csv_file: str):
        df = pd.read_csv(csv_file, dtype = {'Limb annotation': str})
        if type(df['Limb annotation'].iloc[0]) != str:
            df['Limb annotation'] = ['' for i in range(len(df))]
        df['Limb annotation'] = [x if type(x) != float else ''
                                 for x in df['Limb annotation']]
        df = df.drop([col for col in df.columns if 'Unnamed: ' in col], axis=1)
        return ChartStruct(df, source_file = csv_file)

    def to_csv(self, filename: str) -> None:
        metadata_json = json.dumps(self.metadata)
        if 'Metadata' not in self.df.columns or self.df['Metadata'][0] != metadata_json:
            self.df['Metadata'] = [metadata_json] + ['' for line in range(len(self.df)-1)]
        self.df.to_csv(filename, index = False)
        return

    @staticmethod
    def from_piucenterdataframe(pc_df: PiuCenterDataFrame):
        """ Make ChartStruct from old PiuCenter d_annotate df.
        """
        dfs = pc_df.df[['Beat', 'Time', 'Line', 'Line with active holds']].copy()
        dfs['Limb annotation'] = pc_df.get_limb_annotations()
        return ChartStruct(dfs)
    
    @staticmethod
    def from_stepchart_ssc(stepchart_ssc: StepchartSSC):
        df, holdticks, message = stepchart_ssc_to_chartstruct(stepchart_ssc)
        df['Line'] = [f'`{line}' for line in df['Line']]
        df['Line with active holds'] = [f'`{line}' for line in df['Line with active holds']]
        df['Limb annotation'] = ['' for line in df['Line']]

        metadata_dict = stepchart_ssc.get_metadata()
        metadata_dict['Hold ticks'] = holdticks
        metadata_json = json.dumps(metadata_dict)
        df['Metadata'] = [metadata_json] + ['' for line in range(len(df)-1)]
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

    def get_chart_level(self) -> int:
        if 'METER' in self.metadata:
            level_str = int(self.metadata['METER'])
        else:
            # attempt to get level from filename
            basename = os.path.basename(self.source_file)
            maybe_level = basename.split('_')[-2]
            for prefix in ['S', 'HD', 'D']:
                maybe_level = maybe_level.replace(prefix, '')
            level_str = int(maybe_level)
        try:
            return int(level_str)
        except:
            logger.warning(f'Failed to parse chart level')
            return -1

    def get_sord_chartlevel(self) -> str:
        """ Outputs 'S20' / 'D7' etc. """
        return f'{self.singles_or_doubles()[0].upper()}{self.get_chart_level()}'

    @functools.lru_cache
    def get_lines(self) -> list[str]:
        """ Return list of line strings, with ` removed """
        return list(self.df['Line'].apply(lambda l: l.replace('`', '')))

    @functools.lru_cache
    def get_lines_with_active_holds(self) -> list[str]:
        """ Return list of lines with active holds, with ` removed """
        return list(self.df['Line with active holds'].apply(lambda l: l.replace('`', '')))

    """
        Prediction
    """
    @functools.lru_cache
    def get_arrow_coordinates(self) -> list[ArrowCoordinate]:
        """ Get coordinates of arrows: 1, 2, 3"""
        arrow_coords = []
        for idx, row in self.df.iterrows():
            line = row['Line with active holds'].replace('`', '')
            for arrow_pos, action in enumerate(line):
                if action in list('123'):
                    limb_idx = notelines.get_limb_idx_for_arrow_pos(
                        row['Line with active holds'],
                        arrow_pos
                    )
                    is_downpress = action in list('12')
                    coord = ArrowCoordinate(idx, arrow_pos, limb_idx, is_downpress, line)
                    arrow_coords.append(coord)
        return arrow_coords

    @functools.lru_cache
    def get_prediction_coordinates(self) -> list[ArrowCoordinate]:
        """ Get arrow coordinates with downpresses for limb prediction """
        return [ac for ac in self.get_arrow_coordinates() if ac.is_downpress]

    def get_time_since_last_same_arrow_use(self) -> dict[ArrowCoordinate, float]:
        """ For each ArrowCoordinate with downpress, calculates the time since
            that arrow was last used by any limb (1, 2, or 3).
        """
        ac_to_time = dict()
        last_time_used = [None] * 10
        for idx, row in self.df.iterrows():
            line = row['Line'].replace('`', '')
            time = row['Time']
            for arrow_pos, action in enumerate(line):
                if action in list('123'):
                    limb_idx = notelines.get_limb_idx_for_arrow_pos(
                        row['Line with active holds'],
                        arrow_pos
                    )
                    line_ah = row['Line with active holds'].replace('`', '')
                    has_downpress = action in list('12')
                    coord = ArrowCoordinate(idx, arrow_pos, limb_idx, has_downpress, line_ah)
                    if last_time_used[arrow_pos] is not None:
                        ac_to_time[coord] = time - last_time_used[arrow_pos]
                    else:
                        ac_to_time[coord] = -1

                    # update last time used
                    last_time_used[arrow_pos] = time
        return ac_to_time

    def get_previous_used_pred_coord_for_arrow(self) -> dict[int, int | None]:
        """ Compute dict mapping index of (ArrowCoordinate with downpress)
            in pred_coords
            to the index of
            the (ArrowCoordinate with downpress) most recently used for
            the same arrow, which can be None.
            Supports limb featurization that annotates the most recent
            (predicted) limb used for a given ArrowCoordinate. 
        """

        last_idx_used = [None] * 10
        pred_coords = self.get_prediction_coordinates()
        pc_to_prev_idx = dict()
        for idx, pc in enumerate(pred_coords):
            pc_to_prev_idx[pc] = last_idx_used[pc.arrow_pos]
            last_idx_used[pc.arrow_pos] = idx
        return {idx: pc_to_prev_idx[pc] for idx, pc in enumerate(pred_coords)}

    def get_previous_used_pred_coord(self) -> dict[int, list[int | None]]:
        """ Compute dict mapping row index to a list of indices of
            the (ArrowCoordinate with downpress) most recently used
            for each arrow, which can be None.
            Used by tactician to check for impossible lines with holds.
        """
        last_idx_used = [None] * 10
        pcs = self.get_prediction_coordinates()
        acs = self.get_arrow_coordinates()
        row_idx_to_prev = dict()

        row_idx_to_pcs = defaultdict(list)
        for pc in pcs:
            row_idx_to_pcs[pc.row_idx].append(pc)

        all_row_idxs = sorted(list(set(ac.row_idx for ac in acs)))
        for row_idx in all_row_idxs:
            row_idx_to_prev[row_idx] = tuple(last_idx_used)

            # update last idx used
            for pc in row_idx_to_pcs[row_idx]:
                last_idx_used[pc.arrow_pos] = pcs.index(pc)
        return row_idx_to_prev

    def add_limb_annotations(
        self,
        pred_coords: list[ArrowCoordinate],
        limb_annots: list[str],
        new_col: str
    ) -> None:
        """ Populates `new_col` in df with `limb_annots` at `arrow_coords`,
            for example predicted limb annotations.

            Holds started with a limb will use that same limb throughout the hold
            duration.
        """
        assert len(pred_coords) == len(limb_annots)
        df_limb_annots = self.init_limb_annotations()

        row_arrow_to_pcidx = {}
        for pc_idx, ac in enumerate(pred_coords):
            row_arrow_to_pcidx[(ac.row_idx, ac.arrow_pos)] = pc_idx

        def update_limb_annot(row_idx: int, limb_idx: int, new_limb: str):
            prev_annot = df_limb_annots[row_idx]
            new_annot = prev_annot[:limb_idx] + new_limb + prev_annot[limb_idx + 1:]
            df_limb_annots[row_idx] = new_annot
            return

        active_holds = {}   # arrowpos: limb
        for row_idx, row in self.df.iterrows():
            line = row['Line with active holds']

            limb_idx = 0
            for arrow_pos, sym in enumerate(line[1:]):
                if is_active_symbol(sym):

                    if sym in list('12'):
                        # get new limb
                        new_limb = limb_annots[row_arrow_to_pcidx[(row_idx, arrow_pos)]]

                    if sym == '1':
                        update_limb_annot(row_idx, limb_idx, new_limb)
                    elif sym == '2':
                        update_limb_annot(row_idx, limb_idx, new_limb)
                        # add to active holds
                        if arrow_pos not in active_holds:
                            active_holds[arrow_pos] = new_limb
                        else:
                            logger.warning(f'WARNING: {arrow_pos=} in {active_holds=}')
                            continue
                    elif sym == '4':
                        new_limb = active_holds[arrow_pos]
                        update_limb_annot(row_idx, limb_idx, new_limb)
                    elif sym == '3':
                        new_limb = active_holds.pop(arrow_pos)
                        update_limb_annot(row_idx, limb_idx, new_limb)

                    limb_idx += 1
        self.df[new_col] = df_limb_annots
        return

    def init_limb_annotations(self) -> list[str]:
        """ Initializes limb annotations to `new_col` as all ? """
        limb_annots = []
        for idx, row in self.df.iterrows():
            line = row['Line with active holds']
            n_active_symbols = sum(is_active_symbol(s) for s in line)
            limb_annots.append('?' * n_active_symbols)
        return limb_annots

    """
        Annotate
    """
    def annotate_time_since_downpress(self) -> None:
        """ Adds column `__time since prev downpress` to df.
            Uses value -1 for first line (has no prev downpress).
        """
        if '__time since prev downpress' in self.df.columns:
            return
        has_dps = [notelines.has_downpress(line) for line in self.df['Line']]
        recent_downpress_idx = None
        time_since_dp = []
        for idx, row in self.df.iterrows():
            
            if recent_downpress_idx is None:    
                time_since_dp.append(-1)
            else:
                prev_dp_time = self.df.at[recent_downpress_idx, 'Time']
                time_since_dp.append(row['Time'] - prev_dp_time)

            has_dp = has_dps[idx]
            if has_dp:
                recent_downpress_idx = idx

        self.df['__time since prev downpress'] = time_since_dp
        return

    def annotate_time_to_next_downpress(self) -> None:
        """ Adds column `__time to next downpress` to df
        """
        if '__time to next downpress' in self.df.columns:
            return
        has_dps = [notelines.has_downpress(line) for line in self.df['Line']]
        time_to_dps = []
        for idx, row in self.df.iterrows():
            next_dp_idxs = has_dps[idx + 1:]
            if True in next_dp_idxs:
                next_dp_idx = idx + 1 + next_dp_idxs.index(True)
                
                time_to_dp = self.df.at[next_dp_idx, 'Time'] - row['Time']
                time_to_dps.append(time_to_dp)
            else:
                time_to_dps.append(-1)

        self.df['__time to next downpress'] = time_to_dps
        return

    def annotate_line_repeats_previous(self) -> None:
        """ Adds column `__line repeats previous downpress line` to df,
            which is True if current line has the same downpress
            as the previous line with downpress (treating 1 and 2 the same).
        """
        if '__line repeats previous downpress line' in self.df.columns:
            return
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

        self.df['__line repeats previous downpress line'] = line_repeats
        return

    def annotate_line_repeats_next(self) -> None:
        """ Adds column `__line repeats next downpress line` to df,
            which is True if current line has the same downpress
            as the next line with downpress (treating 1 and 2 the same).
        """
        if '__line repeats next downpress line' in self.df.columns:
            return
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

        self.df['__line repeats next downpress line'] = line_repeats
        return

    def annotate_num_downpresses(self) -> None:
        """ Adds column `__num downpresses` to df """
        self.df['__num downpresses'] = self.df['Line'].str.count('1') + \
            self.df['Line'].str.count('2')
        return

    def annotate_single_hold_ends_immediately(self) -> None:
        """ Adds column `__single hold ends immediately` to df """
        if '__single hold ends immediately' in self.df.columns:
            return
        values = []
        for idx, row in self.df.iterrows():
            val = False
            line = row['Line with active holds'].replace('`', '')
            if notelines.has_one_2(line):
                next_line = self.df.at[idx + 1, 'Line with active holds'].replace('`', '')
                if line.replace('2', '3') == next_line:
                    val = True
            values.append(val)

        self.df['__single hold ends immediately'] = values
        return

    """
        Search
    """
    @functools.lru_cache
    def __get_rounded_time(self) -> list[float]:
        return [np.round(t, decimals = 4) for t in self.df['Time']]

    def __time_to_df_idx(self, query_time: float) -> list[int]:
        """ Finds df row idx by query_time """
        rounded_time = self.__get_rounded_time()
        q_time = np.round(query_time, decimals = 4)
        idxs = [i for i, t in enumerate(rounded_time) if math.isclose(q_time, t, abs_tol = 1.1e-4)]
        if len(idxs) > 1:
            pass
        elif len(idxs) == 0:
            logger.error(f'... failed to match lines at {query_time=}')
            import code; code.interact(local=dict(globals(), **locals()))
        return idxs

    """
        Interaction with chart json: check match, update
    """
    def matches_chart_json(
        self, 
        chartjs: ChartJsStruct, 
        with_limb_annot: bool = True
    ) -> bool:
        """ Computes if ChartStruct matches `chartjs` at all arrow times and positions.
            If `with_limb_annot`, requires matching limb annotations too.
        """
        self_cjs = ChartJsStruct.from_chartstruct(self)
        return self_cjs.matches(chartjs, with_limb_annot = with_limb_annot)

    def update_from_manual_json(self, chartjs: ChartJsStruct, verbose: bool = False) -> None:
        """ Updates ChartStruct given chart json, which can be manually
            annotated in step editor web app.
            First checks that `chartjs` is compatible with ChartStruct, and throws
            error if not. 
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
            if n_updates_made == 0:
                logger.warning(f'Failed to update any lines')

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
        line = self.df.at[row_idx, 'Line with active holds'].replace('`', '')

        if line[new_arrow_pos] not in expected_symbols:
            return False

        curr_limb_annot = self.df.at[row_idx, 'Limb annotation']          

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

        def get_limb(limbs: str | float, idx: int) -> str:
            if limbs == '':
                return '?'
            return limbs[idx]

        active_holds = {}   # arrowpos: (time start, limb)
        for row_idx, row in self.df.iterrows():
            line = row['Line with active holds'].replace('`', '')
            limb_annot = row['Limb annotation']
            time = row['Time']

            n_active_symbols_seen = 0
            for arrow_pos, sym in enumerate(line):
                if is_active_symbol(sym):
                    limb = get_limb(limb_annot, n_active_symbols_seen)

                    if sym == '1':
                        arrow_arts.append(ArrowArt(arrow_pos, time, limb))
                        if arrow_pos in active_holds:
                            raise Exception(f'1 in active hold at {row_idx=}, {arrow_pos=}')
                    elif sym == '2':
                        # add to active holds
                        if arrow_pos not in active_holds:
                            active_holds[arrow_pos] = (time, limb)
                        else:
                            logger.warning(f'WARNING: {arrow_pos=} in {active_holds=}')
                            continue
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

    