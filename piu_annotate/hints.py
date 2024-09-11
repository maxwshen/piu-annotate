import pandas as pd
import numpy as np
import math
import itertools

from piu_annotate.formats import notelines
from piu_annotate.formats.chart import ChartStruct


class Hint:
    def preprocess(self, chart_struct_df: pd.DataFrame) -> pd.DataFrame:
        """ Annotate helper columns in chart_struct_df for hint rule,
            such as time elapsed, line repeats, etc.
        """
        raise NotImplementedError

    def apply_rule_on_section(
        self,
        chart_struct_df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> bool:
        """ Returns True or False if hint rule is valid on `chart_struct_df`
            from `start_idx` to `end_idx`
        """
        raise NotImplementedError
    
    def validate(
        self, 
        chart_struct_df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> tuple[bool, str]:
        """ Evaluates whether limb annotations are consistent with hint.
            Return string can be error message or for debugging.
        """
        raise NotImplementedError


def apply_hint(
    chart_struct: ChartStruct, 
    hint: Hint, 
    min_lines: int
) -> list[tuple[int, int]]:
    """ Apply hint on chart_struct dataframe, 
        returning a list of (start_idx, end_idx) for sections matching hint.
    """
    chart_struct.df = hint.preprocess(chart_struct.df)
    df = chart_struct.df

    start_idx = None
    sections = []

    for idx in range(len(df)):
        if start_idx is None:
            if hint.apply_rule_on_section(df, idx, idx + 1):
                start_idx = idx
                continue

        if start_idx is not None:
            if not hint.apply_rule_on_section(df, start_idx, idx + 1):
                end_idx = idx
                length = end_idx - start_idx

                if length >= min_lines:
                    sections.append((start_idx, end_idx))
                
                start_idx = None
    return sections


class AlternateSoloArrows(Hint):
    def __init__(self):
        pass

    def preprocess(self, chart_struct_df: pd.DataFrame) -> pd.DataFrame:
        df = chart_struct_df.copy()
        lines = df['Line with active holds']

        time_since = list(np.array(df['Time'][1:]) - np.array(df['Time'][:-1]))
        time_since.insert(0, np.nan)
        df['__time since'] = time_since

        df['__line has one arrow'] = lines.apply(notelines.has_one_arrow)

        repeats = [l1 == l2 for l1, l2 in zip(lines[1:], lines[:-1])]
        repeats.insert(0, False)
        df['__line repeats'] = repeats

        df['__line has one arrow and not repeat'] = df['__line has one arrow'] & ~df['__line repeats']
        return df

    def apply_rule_on_section(
        self,
        chart_struct_df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> bool:
        df = chart_struct_df
        if end_idx == len(df):
            return False

        dfs = df.iloc[start_idx : end_idx]
        length = len(dfs)

        crits = []
        crits.append(dfs['__line has one arrow and not repeat'].all())

        # time since should be same
        if length > 1:
            time_since_set = set(dfs.iloc[1:]['__time since'])
            time_since_similar = all(
                math.isclose(t1, t2)
                for t1, t2 in itertools.combinations(time_since_set, 2)
            )
            crits.append(time_since_similar)
        return all(crits)
    
    def validate(
        self, 
        chart_struct_df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> tuple[bool, str]:
        length = end_idx - start_idx
        if length < 2:
            return False
        dfs = chart_struct_df.iloc[start_idx : end_idx]
        limbs = list(dfs['Limb annotation'])
        evens = limbs[::2]
        odds = limbs[1::2]
        unique = lambda l: len(set(l)) == 1
        ok = unique(evens) and unique(odds) and evens[0] != odds[0]
        if ok:
            return True, ''
        else:
            nps = 1 / dfs['__time since'].iloc[1]
            return False, f'{nps=}'