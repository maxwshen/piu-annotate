"""
    Featurize
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import hashlib
import gzip

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.datapoints import ArrowDataPoint


def md5_hash(tup) -> str:
    return hashlib.md5(pickle.dumps(tup)).hexdigest()


def guess_singles_or_doubles_from_filename(filename: str) -> str:
    """ """
    basename = os.path.basename(filename)
    sord = basename.split('_')[-2][0]
    if sord == 'S':
        return 'singles'
    elif sord == 'D':
        return 'doubles'
    return 'unsure'


def create_dataset(
    csvs: list[str],
    get_label_func: callable,
    dataset_name: str,
    use_limb_features: bool = False,
):
    singles_doubles = args.setdefault('singles_or_doubles', 'singles')

    # get hash
    hashid = md5_hash(( tuple(sorted(csvs)), dataset_name, use_limb_features, singles_doubles ))

    # try to load dataset
    dataset_storage = args.setdefault('dataset_storage', '/home/maxwshen/piu-annotate/cli/temp/dataset-storage/')
    storage_file = os.path.join(dataset_storage, f'{hashid}.pkl.gz')
    if os.path.isfile(storage_file) and not args.setdefault('rebuild_datasets', False):
        logger.info(f'Loading from {storage_file} ...')
        with gzip.open(storage_file, 'rb') as f:
            return pickle.load(f)

    # subset csvs to singles or doubles
    csv_sord = [csv for csv in csvs
                if guess_singles_or_doubles_from_filename(csv) in [singles_doubles, 'unsure']]

    all_points, all_labels = [], []
    n_csvs = 0
    for csv in tqdm(csv_sord):
        cs = ChartStruct.from_file(csv)
        if cs.singles_or_doubles() != singles_doubles:
            continue

        fcs = featurizers.ChartStructFeaturizer(cs)
        # labels = fcs.get_labels_from_limb_col('Limb annotation')
        # labels = fcs.get_label_matches_next('Limb annotation')
        # labels = fcs.get_label_matches_prev('Limb annotation')
        labels = get_label_func(fcs)

        if use_limb_features:
            points = fcs.featurize_arrowlimbs_with_context(labels)
            feature_names = fcs.get_arrowlimb_context_feature_names()
        else:
            points = fcs.featurize_arrows_with_context()
            feature_names = fcs.get_arrow_context_feature_names()

        all_points.append(points)
        all_labels.append(labels)
        n_csvs += 1
    logger.info(f'Featurized {n_csvs} ChartStruct csvs ...')

    points = np.concatenate(all_points)
    labels = np.concatenate(all_labels)
    logger.info(f'Found dataset shape {points.shape}')

    result = (points, labels, feature_names)
    # with gzip.open(storage_file, 'wb') as f:
    #     pickle.dump(result, f)
    # logger.info(f'Stored dataset into {storage_file}')
    return result


def train_categorical_model(
    points: NDArray, 
    labels: NDArray, 
    feature_names: list[str]
):
    # train/test split
    train_x, test_x, train_y, test_y = train_test_split(
        points, labels, test_size = 0.1, random_state = 0
    )
    
    train_data = lgb.Dataset(
        train_x, 
        label = train_y, 
        feature_name = feature_names,
        categorical_feature = [fn for fn in feature_names if fn.startswith('cat.')],
    )
    test_data = lgb.Dataset(
        test_x, 
        label = test_y,
        feature_name = feature_names,
        categorical_feature = [fn for fn in feature_names if fn.startswith('cat.')],
    )
    params = {'objective': 'binary', 'metric': 'binary_logloss'}
    bst = lgb.train(params, train_data, valid_sets = [test_data])

    # train pred
    train_pred = bst.predict(train_x).round()
    print(sum(train_pred == train_y) / len(train_y))

    test_pred = bst.predict(test_x).round()
    print(sum(test_pred == test_y) / len(test_y))
    return bst


def save_model(bst: Booster, name):
    out_dir = args.setdefault('out_dir', '/home/maxwshen/piu-annotate/artifacts/models/temp')
    singles_doubles = args.setdefault('singles_or_doubles', 'singles')
    out_fn = os.path.join(out_dir, f'{singles_doubles}-{name}.txt')

    bst.save_model(out_fn)

    logger.info(f'Saved model to {out_fn}')
    return


def main():
    folder = args['manual_chart_struct_folder']
    
    # crawl all subdirs for csvs
    csvs = []
    dirpaths = set()
    for dirpath, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv') and 'exclude' not in dirpath:
                csvs.append(os.path.join(dirpath, file))
                dirpaths.add(dirpath)
    logger.info(f'Found {len(csvs)} csvs in {len(dirpaths)} directories ...')

    label_func = lambda fcs: fcs.get_labels_from_limb_col('Limb annotation')
    points, labels, feature_names = create_dataset(csvs, label_func, 'arrows_to_limb')
    model = train_categorical_model(points, labels, feature_names)
    save_model(model, 'arrows_to_limb')

    label_func = lambda fcs: fcs.get_labels_from_limb_col('Limb annotation')
    points, labels, feature_names = create_dataset(csvs, label_func, 'arrowlimbs_to_limb', use_limb_features = True)
    model = train_categorical_model(points, labels, feature_names)
    save_model(model, 'arrowlimbs_to_limb')

    label_func = lambda fcs: fcs.get_label_matches_next('Limb annotation')
    points, labels, feature_names = create_dataset(csvs, label_func, 'matchnext')
    model = train_categorical_model(points, labels, feature_names)
    save_model(model, 'arrows_to_matchnext')

    label_func = lambda fcs: fcs.get_label_matches_prev('Limb annotation')
    points, labels, feature_names = create_dataset(csvs, label_func, 'matchprev')
    model = train_categorical_model(points, labels, feature_names)
    save_model(model, 'arrows_to_matchprev')

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Train LightGBM model suite on chart struct with manual limb annotations.
    """)
    parser.add_argument(
        '--manual_chart_struct_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/',
    )
    parser.add_argument(
        '--singles_or_doubles', 
        default = 'singles',
    )
    args.parse_args(parser)
    main()