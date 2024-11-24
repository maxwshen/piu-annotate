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


def build_dataset():
    # load from file if exists
    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/datasets/temp.pkl'
    if not args.setdefault('rerun', False):
        if os.path.exists(dataset_fn):
            with open(dataset_fn, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded dataset from {dataset_fn}')
            return dataset

    # run on folder
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    from collections import defaultdict
    X = []
    Y = []
    files = []
    singles_or_doubles = []
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        # featurize
        x = featurizers.featurize(cs)
        X.append(x)

        Y.append(int(cs.metadata['METER']))
        files.append(cs_file)
        singles_or_doubles.append(cs.singles_or_doubles())

    dataset = {'x': np.array(X), 'y': np.array(Y),
               'files': files, 'singles_or_doubles': singles_or_doubles}

    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/datasets/temp.pkl'
    with open(dataset_fn, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f'Saved dataset to {dataset_fn}')
    return dataset


def train_model(dataset: dict):
    # train/test split
    points = dataset['x']
    labels = dataset['y']

    train_x, test_x, train_y, test_y = train_test_split(
        points, labels, test_size = 0.1, random_state = 0
    )
    
    train_data = lgb.Dataset(train_x, label = train_y)
    test_data = lgb.Dataset(test_x, label = test_y)
    params = {'objective': 'regression'}
    bst = lgb.train(params, train_data, valid_sets = [test_data])

    # train pred
    train_pred = bst.predict(train_x).round()
    logger.info(sum(train_pred == train_y) / len(train_y))

    test_pred = bst.predict(test_x).round()
    logger.info(sum(test_pred == test_y) / len(test_y))

    from scipy.stats import pearsonr, spearmanr
    logger.info(pearsonr(test_pred, test_y))
    import code; code.interact(local=dict(globals(), **locals()))
    return bst


def main():
    dataset = build_dataset()

    bst = train_model(dataset)


    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Segments ChartStruct CSVs, updating metadata field
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
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