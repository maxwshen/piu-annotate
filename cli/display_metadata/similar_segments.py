import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys

from sklearn.neighbors import NearestNeighbors

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.segment import Section
from piu_annotate.utils import make_basename_url_safe


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


class SegmentSimilarity:
    def __init__(
        self,
        dataset_fn: str = '/home/maxwshen/piu-annotate/artifacts/difficulty/segments/feature-store-segment.pkl'
    ):
        assert os.path.exists(dataset_fn)
        self.dataset_fn = dataset_fn
        with open(dataset_fn, 'rb') as f:
            dataset = pickle.load(f)
        logger.info(f'Loaded dataset from {dataset_fn}')
        self.dataset = dataset
        self.ft_names = dataset['feature names']

        sections = []
        xs = []
        section_sords = []
        logger.info(f'Preparing dataset ...')
        from tqdm import tqdm
        for shortname, mat in tqdm(dataset.items()):
            if shortname == 'feature names':
                continue
            if 'fullstepchart' in shortname:
                continue
            xs.append(mat)
            sections += [(shortname, section_idx) for section_idx in range(len(mat))]
            section_sords += [guess_sord_from_shortname(shortname)] * len(mat)

        # section info
        self.sections = sections
        shortnames = [s[0] for s in sections]
        self.shortname_to_level = {name: guess_level_from_shortname(name)
                            for name in shortnames}
        self.shortname_to_sord = {name: guess_sord_from_shortname(name)
                            for name in shortnames}

        # form data
        all_xs = np.concatenate(xs, axis = 0)
        all_xs = np.nan_to_num(all_xs)
        # shape: (total_num_sections, num_features)
        # normalize each skill ft in xs
        self.all_xs = (all_xs - np.nanmean(all_xs, axis = 0)) / np.nanstd(all_xs, axis = 0)

        section_sords = np.array(section_sords)
        self.sord_to_knns: dict[str, NearestNeighbors] = dict()
        self.sord_to_sections = dict()
        # build k-nn separately for singles/doubles
        for sord in ['S', 'D']:
            xs = self.all_xs[np.where(section_sords == sord)]

            # default = L2 norm distance
            self.sord_to_knns[sord] = NearestNeighbors(n_neighbors = 100).fit(xs)
            self.sord_to_sections[sord] = [s for i, s in enumerate(sections) if section_sords[i] == sord]

    def find_closest(
        self, 
        shortname: str, 
        section_idx: int,
        level_threshold: int = 0,
        num_closest: int = 5,
    ) -> list[tuple[str, int]]:
        """ From a query (shortname, section_idx), returns a list of
            `num_closest` sections (shortname, section_idx),
            within `level_threshold` of query level, and matching `sord`.
        """
        query_sord = self.shortname_to_sord[shortname]
        query_level = self.shortname_to_level[shortname]
        knn: NearestNeighbors = self.sord_to_knns[query_sord]
        sections: list[tuple[str, int]] = self.sord_to_sections[query_sord]
        
        query_idx = self.sections.index((shortname, section_idx))
        query_vector = self.all_xs[query_idx]

        _, idx_array = knn.kneighbors(query_vector.reshape(1, -1))
        closest_idxs = list(idx_array[0])

        j = 0
        found_sections = []
        found_names = set()
        while len(found_sections) < num_closest and j < len(closest_idxs):
            found_idx = closest_idxs[j]
            found_section = sections[found_idx]
            found_name = found_section[0]
            crits = [
                self.shortname_to_level[found_name] <= query_level + level_threshold,
                self.shortname_to_sord[found_name] == query_sord,
                found_name != shortname,
                found_name not in found_names,
            ]
            if all(crits):
                # logger.debug(found_section, '--', f'{dists[found_idx]:.2f}')
                found_sections.append(found_section)
                found_names.add(found_name)
            j += 1

        return found_sections


def annotate_segment_similarity():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    ss = SegmentSimilarity()

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
            # 'GLORIA_-_Croire_D21_ARCADE.csv',
            'Final_Audition_2__-_SHORT_CUT_-_-_Banya_S17_SHORTCUT.csv',
        ]
        chartstruct_files = [os.path.join(cs_folder, f) for f in chartstruct_files]

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        shortname = cs.metadata['shortname']
        chart_level = cs.get_chart_level()

        assert 'Segment metadata' in cs.metadata, 'Expected segment metadata dicts to already be created'
        meta_dicts = cs.metadata['Segment metadata']
        # one dict per section
        segment_pred_levels = [meta_dicts[i]['level'] for i in range(len(meta_dicts))]
        max_segment_pred_level = max(segment_pred_levels)

        for section_idx in range(len(sections)):
            segment_pred_level = segment_pred_levels[section_idx]
            closest_sections = []

            # skip similar sections for very easy segments
            crits = [
                segment_pred_level >= max_segment_pred_level - 4,
                segment_pred_level >= chart_level - 4,
                len(meta_dicts[section_idx]['rare skills']) > 0,
            ]
            if any(crits):
                closest_sections = ss.find_closest(
                    shortname, 
                    section_idx,
                )
                # if no sections are found, try with increased level threshold
                if len(closest_sections) == 0:
                    closest_sections = ss.find_closest(
                        shortname, 
                        section_idx,
                        level_threshold = 3,
                    )

            # convert to url safe
            urlsafe_sections = [
                (make_basename_url_safe(shortname), section_idx)
                for shortname, section_idx in closest_sections
            ]
            meta_dicts[section_idx]['Closest sections'] = urlsafe_sections

        cs.metadata['Segment metadata'] = meta_dicts
        if debug:
            import code; code.interact(local=dict(globals(), **locals()))
        cs.to_csv(inp_fn)

    return


def main():
    annotate_segment_similarity()

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