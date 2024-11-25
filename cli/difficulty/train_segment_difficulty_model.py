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
from piu_annotate.segment.segment import Section


def build_crux_segment_dataset():
    """ Builds dataset of crux segments.
        Uses pretrained stepchart difficulty prediction model to find cruxes.
    """
    # load from file if exists
    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/cruxes/datasets/temp.pkl'
    if not args.setdefault('rerun', False):
        if os.path.exists(dataset_fn):
            with open(dataset_fn, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded dataset from {dataset_fn}')
            return dataset

    model = lgb.Booster(model_file = args['full_stepchart_difficulty_model_singles'])

    # run on folder
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    chartstruct_files = [
        '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
        '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Conflict_-_Siromaru_+_Cranky_D21_ARCADE.csv',
        '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/BOOOM!!_-_RiraN_D22_ARCADE.csv'
    ]

    X = []
    Y = []
    files = []
    singles_or_doubles = []
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        sord = cs.singles_or_doubles()
        # model = lgb.Booster(model_file = args[f'full_stepchart_difficulty_model_{sord}'])
        model_fn = f'/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/full-stepchart-hist-{sord}.pkl'
        with open(model_fn, 'rb') as f:
            model = pickle.load(f)

        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        fter = featurizers.DifficultyFeaturizer(cs)
        xs = fter.featurize_sections(sections)

        # feature_subset_idxs = [0, 4, 8, 12]
        # xs = xs[:, feature_subset_idxs]

        print(cs_file)
        section_preds = model.predict(xs)
        print(section_preds)

        x = fter.featurize_full_stepchart()
        # x = x[feature_subset_idxs]
        x = x.reshape(1, -1)
        sc_pred = model.predict(x)
        print(sc_pred)

        scale = sc_pred / max(section_preds)
        print(section_preds * scale)
        import code; code.interact(local=dict(globals(), **locals()))

        # # featurize
        # x = featurizers.featurize(cs)
        # X.append(x)

        # Y.append(int(cs.metadata['METER']))
        # files.append(cs_file)
        # singles_or_doubles.append(cs.singles_or_doubles())

    # dataset = {'x': np.array(X), 'y': np.array(Y),
    #            'files': files, 'singles_or_doubles': singles_or_doubles}

    # with open(dataset_fn, 'wb') as f:
    #     pickle.dump(dataset, f)
    # logger.info(f'Saved dataset to {dataset_fn}')
    return dataset



def main():
    """ Featurize full stepcharts and train difficulty prediction model.
    """
    dataset = build_crux_segment_dataset()

    # model_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/cruxes/segment-model.txt'
    # bst.save_model(model_fn)
    # logger.info(f'Saved model to {model_fn}')

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Use pretrained stepchart difficulty prediction model to extract
        crux segments, by predicting difficulty on chart segments.
        
        Crux segments form dataset for training a segment difficulty prediction
        model, using stepchart difficulty as label.
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