import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import train_test_split

from piu_annotate.formats.chart import ChartStruct
from piu_annotate import utils
from piu_annotate.difficulty import featurizers
from piu_annotate.difficulty.models import DifficultyModelPredictor
from piu_annotate.segment.segment import Section
from piu_annotate.segment.segment_breaks import get_segment_metadata


def annotate_segment_difficulty():
    """ Runs pretrained stepchart difficulty prediction model on segments.
    """
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    debug = args.setdefault('debug', False)
    if debug:
        folder = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/'
        chartstruct_files = [
            'X-Rave_-_SHORT_CUT_-_-_DM_Ashura_D18_SHORTCUT.csv',
            'Altale_-_sakuzyo_D19_ARCADE.csv',
            'Native_-_SHK_S20_ARCADE.csv',
            'Kimchi_Fingers_-_Garlic_Squad_D21_ARCADE.csv',
            'Life_is_PIANO_-_Junk_D21_ARCADE.csv',
            '8_6_-_DASU_D21_ARCADE.csv',
            'The_End_of_the_World_ft._Skizzo_-_MonstDeath_D22_ARCADE.csv',
            'Super_Fantasy_-_SHK_S16_INFOBAR_TITLE_ARCADE.csv',
            'HTTP_-_Quree_S21_ARCADE.csv',
            'GOODBOUNCE_-_EBIMAYO_D21_ARCADE.csv',
            'Dement_~After_Legend~_-_Lunatic_Sounds_D26_ARCADE.csv',
            'My_Dreams_-_Banya_Production_D22_ARCADE.csv',
            'Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
            'Conflict_-_Siromaru_+_Cranky_D21_ARCADE.csv',
            'BOOOM!!_-_RiraN_D22_ARCADE.csv'
        ]
        chartstruct_files = [folder + f for f in chartstruct_files]

    # Load models
    dmp = DifficultyModelPredictor()
    dmp.load_models()

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]

        preds = dmp.predict_segment_difficulties(cs)

        # update segment metadata dicts with level
        meta_dicts = [get_segment_metadata(cs, s) for s in sections]
        for md, pred_level in zip(meta_dicts, list(preds)):
            md['level'] = np.round(pred_level, 2)
        cs.metadata['Segment metadata'] = meta_dicts

        cs.to_csv(inp_fn)

    return



def main():
    annotate_segment_difficulty()
    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Runs pretrained stepchart difficulty prediction model on segments.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    parser.add_argument(
        '--full_stepchart_difficulty_model_singles',
        default = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/full-stepchart-model-singles.txt'
    )
    parser.add_argument(
        '--full_stepchart_difficulty_model_doubles',
        default = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/full-stepchart-model-doubles.txt'
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Nyarlathotep_-_nato_S21_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/BOOOM!!_-_RiraN_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()