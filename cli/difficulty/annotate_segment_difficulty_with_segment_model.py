import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys
from collections import defaultdict

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import train_test_split

from piu_annotate.formats.chart import ChartStruct
from piu_annotate import utils
from piu_annotate.difficulty import featurizers
from piu_annotate.difficulty.models import DifficultySegmentModelPredictor
from piu_annotate.segment.segment import Section
from piu_annotate.segment.segment_breaks import get_segment_metadata
from cli.difficulty.train_segment_difficulty import build_segment_feature_store


def annotate_segments(dataset: dict):
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    ft_names = dataset['feature names']

    # Load models
    dmp = DifficultySegmentModelPredictor()
    dmp.load_models()

    if args.setdefault('debug', False):
        chartstruct_files = [
            'Winter_-_BanYa_D21_ARCADE.csv',
            'Sudden_Appearance_Image_-_Blacklolita_D23_ARCADE.csv',
            'Conflict_-_Siromaru_+_Cranky_S6_ARCADE.csv',
            # 'Phantom_-Intermezzo-_-_Banya_Production_S7_ARCADE.csv',
            'GLORIA_-_Croire_D21_ARCADE.csv',
            # 'The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog_-_Doin_D24_ARCADE.csv',
            # 'Good_Night_-_Dreamcatcher_D22_ARCADE.csv',
            # 'PRiMA_MATERiA_-_xi_D24_ARCADE.csv',
            # 'Full_Moon_-_Dreamcatcher_D20_ARCADE.csv',
            # 'Full_Moon_-_Dreamcatcher_D24_ARCADE.csv',
            'Mopemope_-_LeaF_D25_ARCADE.csv',
            'Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
            'My_Dreams_-_Banya_Production_D22_ARCADE.csv',
            # 'Conflict_-_Siromaru_+_Cranky_D21_ARCADE.csv',
            # 'GLORIA_-_Croire_D25_ARCADE.csv',
            # 'Life_is_PIANO_-_Junk_D21_ARCADE.csv',
            # '8_6_-_FULL_SONG_-_-_DASU_S21_FULLSONG.csv',
            'BOOOM!!_-_RiraN_D22_ARCADE.csv',
            # 'Papasito_(feat.__KuTiNA)_-_FULL_SONG_-_-_Yakikaze_&_Cashew_S19_FULLSONG.csv',
            # 'Conflict_-_Siromaru_+_Cranky_S15_ARCADE.csv',
            # 'X-Rave_-_SHORT_CUT_-_-_DM_Ashura_D18_SHORTCUT.csv',
            # 'Dement_~After_Legend~_-_Lunatic_Sounds_D26_ARCADE.csv',
            # 'Altale_-_sakuzyo_D19_ARCADE.csv',
            # 'Native_-_SHK_S20_ARCADE.csv',
            # 'Kimchi_Fingers_-_Garlic_Squad_D21_ARCADE.csv',
            # '8_6_-_DASU_D21_ARCADE.csv',
            # 'The_End_of_the_World_ft._Skizzo_-_MonstDeath_D22_ARCADE.csv',
            # 'Super_Fantasy_-_SHK_S16_INFOBAR_TITLE_ARCADE.csv',
            # 'HTTP_-_Quree_S21_ARCADE.csv',
            'GOODBOUNCE_-_EBIMAYO_D21_ARCADE.csv',
        ]
        chartstruct_files = [os.path.join(cs_folder, f) for f in chartstruct_files]

    rerun_all_preds = args.setdefault('rerun_all_preds', False)
    stats = defaultdict(int)

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        shortname = cs.metadata['shortname']

        if not rerun_all_preds:
            ds = cs.metadata['Segment metadata']
            if 'level' in ds[0]:
                # we already predicted section level
                stats['skipped'] += 1
                continue

        xs = dataset[shortname]        
        segment_dicts = dmp.predict_segments(cs, xs, ft_names)
        stats['predicted'] += 1

        # update segment metadata dicts with level
        meta_dicts = [get_segment_metadata(cs, s) for s in sections]
        for md, sd in zip(meta_dicts, segment_dicts):
            for k, v in sd.items():
                md[k] = v
        cs.metadata['Segment metadata'] = meta_dicts

        cs.to_csv(inp_fn)

    logger.debug(stats)
    return


def main():
    dataset = build_segment_feature_store('segment')
    annotate_segments(dataset)

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Runs pretrained segment difficulty prediction model on segments.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/Nyarlathotep_-_nato_S21_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/BOOOM!!_-_RiraN_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()