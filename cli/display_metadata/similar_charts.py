import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.segment import Section
from piu_annotate.utils import make_basename_url_safe
from cli.page_content.skills_page import make_skills_dataframe


def guess_level_from_shortname(shortname: str) -> int:
    """ """
    shortname = shortname.replace('_INFOBAR_TITLE_', '_')
    shortname = shortname.replace('_HALFDOUBLE_', '_')
    try:
        return int(shortname.split('_')[-2][1:])
    except:
        logger.error(shortname)
        return ''

def guess_sord_from_shortname(shortname: str) -> str:
    """ """
    shortname = shortname.replace('_INFOBAR_TITLE_', '_')
    shortname = shortname.replace('_HALFDOUBLE_', '_')
    try:
        return shortname.split('_')[-2][0]
    except:
        logger.error(shortname)
        return ''


class ChartSimilarity:
    def __init__(
        self,
        dataset_fn: str = '/home/maxwshen/piu-annotate/artifacts/skills/stepchart-skills.csv'
    ):
        """ Depends on running skills_page.py first, to build stepchart skill df
        """
        assert os.path.exists(dataset_fn)
        self.dataset_fn = dataset_fn
        self.dataset = pd.read_csv(dataset_fn)
        logger.info(f'Loaded dataset from {dataset_fn}')

        # info
        self.shortnames = list(self.dataset['shortname'])
        self.shortname_to_level = {name: level for name, level in zip(self.dataset['shortname'], self.dataset['chart level'])}
        self.shortname_to_sord = {name: level for name, level in zip(self.dataset['shortname'], self.dataset['sord'])}

        not_ft_cols = ['shortname', 'sord', 'chart level']
        ft_cols = [col for col in self.dataset if col not in not_ft_cols]
        all_xs = np.array(self.dataset[ft_cols])
        # shape: (total_num_stepcharts, num_features)

        # normalize each skill ft in xs
        self.all_xs = (all_xs - np.nanmean(all_xs, axis = 0)) / np.nanstd(all_xs, axis = 0)

        sords = np.array(self.dataset['sord'])
        self.sord_to_knns: dict[str, NearestNeighbors] = dict()
        self.sord_to_charts = dict()
        # build k-nn separately for singles/doubles
        for sord in ['singles', 'doubles']:
            xs = self.all_xs[np.where(sords == sord)]

            # default = L2 norm distance
            self.sord_to_knns[sord] = NearestNeighbors(n_neighbors = 100).fit(xs)
            self.sord_to_charts[sord] = [s for i, s in enumerate(self.shortnames) if sords[i] == sord]

    def find_closest(
        self, 
        shortname: str, 
        desired_level: int,
        num_closest: int = 3,
    ) -> list[str]:
        """ From a query (shortname), returns a list of
            `num_closest` charts,
            within `level_threshold` of query level, and matching `sord`.
        """
        query_sord = self.shortname_to_sord[shortname]
        knn: NearestNeighbors = self.sord_to_knns[query_sord]
        charts: list[str] = self.sord_to_charts[query_sord]
        
        query_idx = self.shortnames.index(shortname)
        query_vector = self.all_xs[query_idx]

        _, idx_array = knn.kneighbors(query_vector.reshape(1, -1))
        closest_idxs = list(idx_array[0])

        j = 0
        found_charts = []
        while len(found_charts) < num_closest and j < len(closest_idxs):
            found_idx = closest_idxs[j]
            found_name = charts[found_idx]
            crits = [
                self.shortname_to_level[found_name] == desired_level,
                self.shortname_to_sord[found_name] == query_sord,
                found_name != shortname,
                found_name not in found_charts,
            ]
            if all(crits):
                found_charts.append(found_name)
            j += 1

        return found_charts


def annotate_chart_similarity():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    ss = ChartSimilarity()

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
            'GLORIA_-_Croire_D21_ARCADE.csv',
            # 'Final_Audition_2__-_SHORT_CUT_-_-_Banya_S17_SHORTCUT.csv',
        ]
        chartstruct_files = [os.path.join(cs_folder, f) for f in chartstruct_files]

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        shortname = make_basename_url_safe(cs.metadata['shortname'])
        chart_level = cs.get_chart_level()

        similar_chart_dict = dict()
        for level in range(chart_level - 2, chart_level + 3):
            closest_charts = ss.find_closest(shortname, desired_level = level)
            similar_chart_dict[level] = closest_charts

        cs.metadata['Similar charts'] = similar_chart_dict
        if debug:
            import code; code.interact(local=dict(globals(), **locals()))
        cs.to_csv(inp_fn)

    return


def main():
    annotate_chart_similarity()

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Update ChartStruct metadata Segment metadata with similar segments.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524',
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/Nyarlathotep_-_nato_S21_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/BOOOM!!_-_RiraN_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()