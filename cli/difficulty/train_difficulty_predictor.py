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


def build_full_stepchart_dataset():
    # load from file if exists
    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/datasets/temp.pkl'
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

    X = []
    Y = []
    files = []
    singles_or_doubles = []
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        # featurize
        fter = featurizers.DifficultyFeaturizer(cs)
        x = fter.featurize_full_stepchart()
        X.append(x)

        Y.append(int(cs.metadata['METER']))
        files.append(cs_file)
        singles_or_doubles.append(cs.singles_or_doubles())

    dataset = {
        'x': np.array(X), 
        'y': np.array(Y),
        'files': files, 
        'singles_or_doubles': singles_or_doubles,
        'feature_names': fter.get_feature_names()
    }

    with open(dataset_fn, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f'Saved dataset to {dataset_fn}')
    return dataset


def train_hist(
    dataset: dict, 
    singles_or_doubles: str,
    feature_subset: str = 'all',
):
    """ Trains a monotonic HistGradientBoostingRegressor
        to predict stepchart difficulty
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    # train/test split
    sd_selector = np.where(np.array(dataset['singles_or_doubles']) == singles_or_doubles)
    points = dataset['x'][sd_selector]
    labels = dataset['y'][sd_selector]
    feature_names = dataset['feature_names']

    if feature_subset != 'all':
        ok_idxs = [i for i, nm in enumerate(feature_names) if feature_subset in nm]
        logger.info(f'Using {feature_subset=} with {len(ok_idxs)} features ...')
        points = points[:, ok_idxs]

    train_x, test_x, train_y, test_y = train_test_split(
        points, labels, test_size = 0.05, random_state = 0
    )
    
    model = HistGradientBoostingRegressor(monotonic_cst = [1] * points.shape[-1])
    model.fit(train_x, train_y)

    from scipy.stats import linregress
    test_pred = model.predict(test_x)
    train_pred = model.predict(train_x)
    logger.info('hist')
    logger.info(model.score(train_x, train_y))
    logger.info(model.score(test_x, test_y))
    logger.info(singles_or_doubles)
    logger.info(f'train set: {linregress(train_pred, train_y)}')
    logger.info(f'val set: {linregress(test_pred, test_y)}')

    import pickle
    model_dir = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/'
    model_fn = os.path.join(model_dir, f'{singles_or_doubles}-{feature_subset}.pkl')
    with open(model_fn, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Saved model to {model_fn}')
    return


def train_lgbm(
    dataset: dict, 
    singles_or_doubles: str,
    feature_subset: str = 'all',
):
    """ Trains a monotonic HistGradientBoostingRegressor
        to predict stepchart difficulty
    """
    import lightgbm as lgb
    from lightgbm import Booster
    # train/test split
    sd_selector = np.where(np.array(dataset['singles_or_doubles']) == singles_or_doubles)
    points = dataset['x'][sd_selector]
    labels = dataset['y'][sd_selector]
    feature_names = dataset['feature_names']

    if feature_subset != 'all':
        ok_idxs = [i for i, nm in enumerate(feature_names) if feature_subset in nm]
        logger.info(f'Using {feature_subset=} with {len(ok_idxs)} features ...')
        points = points[:, ok_idxs]

    train_x, test_x, train_y, test_y = train_test_split(
        points, labels, test_size = 0.05, random_state = 0
    )
    train_data = lgb.Dataset(train_x, label = train_y)
    test_data = lgb.Dataset(test_x, label = test_y)

    params = {
        'objective': 'regression', 
        'force_col_wise': True,
        'monotone_constraints': [1] * points.shape[-1],
        'monotone_constraints_method': 'advanced',
        'verbose': -1,
    }
    model = lgb.train(params, train_data, valid_sets = [test_data])

    from scipy.stats import linregress
    from sklearn.metrics import r2_score
    test_pred = model.predict(test_x)
    train_pred = model.predict(train_x)
    logger.info('hist')
    logger.info(r2_score(train_pred, train_y))
    logger.info(r2_score(test_pred, test_y))
    logger.info(singles_or_doubles)
    logger.info(f'train set: {linregress(train_pred, train_y)}')
    logger.info(f'val set: {linregress(test_pred, test_y)}')

    # import pickle
    model_dir = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/'
    model_fn = os.path.join(model_dir, f'lgbm-{singles_or_doubles}-{feature_subset}.txt')
    model.save_model(model_fn)
    logger.info(f'Saved model to {model_fn}')
    return


def main():
    """ Featurize full stepcharts and train difficulty prediction model.
    """
    dataset = build_full_stepchart_dataset()

    for sd in ['singles', 'doubles']:
        # for feature_subset in ['all']:
        for feature_subset in ['all', 'bracket', 'edp']:
            train_hist(dataset, sd, feature_subset = feature_subset)
            train_lgbm(dataset, sd, feature_subset = feature_subset)

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Trains difficulty prediction model on ChartStruct CSVs.
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