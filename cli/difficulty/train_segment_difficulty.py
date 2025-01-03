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
from piu_annotate.difficulty.models import DifficultyStepchartModelPredictor
from piu_annotate.segment.segment import Section
from piu_annotate.segment.segment_breaks import get_segment_metadata


def worker(inp_fn: str, feature_type: str) -> dict:
    cs = ChartStruct.from_file(inp_fn)
    sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]

    if feature_type == 'segment':
        fter = featurizers.DifficultySegmentFeaturizer(cs)
    else:
        fter = featurizers.DifficultyStepchartFeaturizer(cs)

    shortname = cs.metadata['shortname']
    dataset = {}
    dataset[shortname] = fter.featurize_sections(sections)
    dataset[f'{shortname}-fullstepchart'] = fter.featurize_full_stepchart()
    dataset['feature names'] = fter.get_feature_names()
    return dataset


def build_segment_feature_store(feature_type: str) -> dict:
    """ Featurizes segments using segment or stepchart featurization,
        storing into pkl. If pkl already exists, loads from file.
    """
    assert feature_type in ['segment', 'stepchart']

    if feature_type == 'segment':
        dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/segments/feature-store-segment.pkl'
    else:
        dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/segments/feature-store-stepchart.pkl'

    rerun_all = args.setdefault('rerun_all', False)
    rerun_ftstore = args.setdefault(f'rerun_ftstore_{feature_type}', False)
    if not rerun_all and not rerun_ftstore:
        if os.path.exists(dataset_fn):
            with open(dataset_fn, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded feature store from {dataset_fn}')
            return dataset

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    dataset = dict()

    do_parallel = args.setdefault('parallel', True)
    if not do_parallel:
        logger.info(f'Building featurized segments for {feature_type=}, serially ...')
        for cs_file in tqdm(chartstruct_files):
            inp_fn = os.path.join(cs_folder, cs_file)
            d = worker(inp_fn, feature_type)
            for k, v in d.items():
                dataset[k] = v

    # do parallel
    elif do_parallel:
        logger.info(f'Building featurized segments for {feature_type=}, parallelizing ...')
        inputs = [[os.path.join(cs_folder, cs_file), feature_type]
                  for cs_file in chartstruct_files]
        import multiprocessing as mp
        with mp.Pool(num_processes := 6) as pool:
            dataset_dicts = pool.starmap(
                worker,
                tqdm(inputs, total = len(inputs))
            )

        # combine into one dataset dict
        dataset = {}
        for d in tqdm(dataset_dicts):
            for k, v in d.items():
                dataset[k] = v

    with open(dataset_fn, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f'Saved dataset to {dataset_fn}')
    return dataset


def build_dataset(ft_store_segment: dict, ft_store_stepchart: dict) -> dict:
    """ From ft_store (xs), return a dataset of xs and ys.
        
        ys are from stepchart difficulty predictor model to obtain segment difficulties, which will be used to train a segment difficulty model.
    """
    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/segments/dataset.pkl'
    rerun_all = args.setdefault('rerun_all', False)
    rerun_ys = args.setdefault('rerun_ys', False)
    if not rerun_all and not rerun_ys:
        if os.path.exists(dataset_fn):
            with open(dataset_fn, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded dataset from {dataset_fn}')
            return dataset

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    # Load models
    dmp = DifficultyStepchartModelPredictor()
    dmp.load_models()

    if args.setdefault('debug', False):
        chartstruct_files = [

        ]

    all_xs = []
    all_ys = []
    singles_or_doubles = []
    logger.info(f'Creating dataset with ys ... ')
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        shortname = cs.metadata['shortname']

        # use stepchart featurizer to form x, to use stepchart model to predict y, on segments
        stepchart_features_segment_xs = ft_store_stepchart[shortname]
        segment_dicts = dmp.predict_segments(
            cs, 
            stepchart_features_segment_xs, 
            ft_store_stepchart['feature names']
        )

        y = np.array([d['level'] for d in segment_dicts])

        # get cruxes by level close to hardest predicted segment
        crux_level_pickup = 1.5
        idxs = [i for i, lv in enumerate(y) if lv >= max(y) - crux_level_pickup]

        # include sections with long 16th note runs
        # todo?

        # subset to cruxes
        segment_xs = ft_store_segment[shortname]
        segment_xs = segment_xs[idxs]

        # use chart level as target level
        y = np.array([cs.get_chart_level()] * len(idxs))

        all_xs.append(segment_xs)
        all_ys.append(y)
        singles_or_doubles.append(cs.singles_or_doubles())

        # include featurized full stepchart
        full_stepchart_x = ft_store_segment[f'{shortname}-fullstepchart']
        all_xs.append(full_stepchart_x)
        all_ys.append(np.array([cs.get_chart_level()]))
        singles_or_doubles.append(cs.singles_or_doubles())

    logger.info(f'Created dataset with {len(all_xs)=}, {len(all_ys)=}')
    dataset = {
        'x': all_xs,
        'y': all_ys,
        'singles_or_doubles': singles_or_doubles,
    }

    with open(dataset_fn, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f'Saved dataset to {dataset_fn}')

    return dataset


def train_lgbm(
    dataset: dict, 
    singles_or_doubles: str,
):
    """ Trains a monotonic HistGradientBoostingRegressor
        to predict stepchart difficulty
    """
    import lightgbm as lgb
    from lightgbm import Booster

    # train/test split
    sd_idxs = np.where(np.array(dataset['singles_or_doubles']) == singles_or_doubles)[0]

    # each item in points is: [n_sections (variable), n_features]
    points = [x for i, x in enumerate(dataset['x']) if i in sd_idxs]
    labels = [y for i, y in enumerate(dataset['y']) if i in sd_idxs]

    # concatenate across sections
    points = np.concatenate(points, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    # shape: (total_n_sections, ft_dim)

    logger.info(f'Found dataset shape: {points.shape}, {labels.shape}')

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
    model_dir = '/home/maxwshen/piu-annotate/artifacts/difficulty/segments/'
    model_fn = os.path.join(model_dir, f'lgbm-{singles_or_doubles}.txt')
    model.save_model(model_fn)
    logger.info(f'Saved model to {model_fn}')
    return


def main():
    ft_store_segment = build_segment_feature_store(feature_type = 'segment')
    ft_store_stepchart = build_segment_feature_store(feature_type = 'stepchart')

    if args.setdefault('retrain_model', False):
        dataset = build_dataset(ft_store_segment, ft_store_stepchart)
        for sd in ['singles', 'doubles']:
            train_lgbm(dataset, sd)

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Runs pretrained stepchart difficulty prediction model on segments.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    args.parse_args(parser)
    main()