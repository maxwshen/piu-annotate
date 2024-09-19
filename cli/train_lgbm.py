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

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers


def create_dataset(
    csvs: list[str],
    get_label_func: callable,
    use_limb_features: bool = False,
):
    singles_doubles = args.setdefault('singles_or_doubles', 'singles')

    all_points, all_labels = [], []
    n_csvs = 0
    for csv in tqdm(csvs):
        cs = ChartStruct.from_file(csv)
        if cs.singles_or_doubles() == singles_doubles:

            fcs = featurizers.ChartStructFeaturizer(cs)
            # labels = fcs.get_labels_from_limb_col('Limb annotation')
            # labels = fcs.get_label_matches_next('Limb annotation')
            # labels = fcs.get_label_matches_prev('Limb annotation')
            labels = get_label_func(fcs)

            if use_limb_features:
                points = fcs.featurize_arrowlimbs_with_context(labels)
            else:
                points = fcs.featurize_arrows_with_context()

            all_points.append(points)
            all_labels.append(labels)
            n_csvs += 1
    logger.info(f'Featurized {n_csvs} ChartStruct csvs ...')

    points = np.concatenate(all_points)
    labels = np.concatenate(all_labels)
    logger.info(f'Found dataset shape {points.shape}')
    return points, labels


def train_model(points: NDArray, labels: NDArray):
    # train/test split
    train_x, test_x, train_y, test_y = train_test_split(points, labels, test_size = 0.1)

    train_data = lgb.Dataset(train_x, label = train_y)
    test_data = lgb.Dataset(test_x, label = test_y)
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
    folder = args['chart_struct_folder']
    csvs = [os.path.join(folder, fn) for fn in os.listdir(folder)
            if fn.endswith('.csv')]

    label_func = lambda fcs: fcs.get_labels_from_limb_col('Limb annotation')
    points, labels = create_dataset(csvs, label_func)
    model = train_model(points, labels)
    save_model(model, 'arrows_to_limb')

    label_func = lambda fcs: fcs.get_labels_from_limb_col('Limb annotation')
    points, labels = create_dataset(csvs, label_func, use_limb_features = True)
    model = train_model(points, labels)
    save_model(model, 'arrowlimbs_to_limb')

    label_func = lambda fcs: fcs.get_label_matches_next('Limb annotation')
    points, labels = create_dataset(csvs, label_func)
    model = train_model(points, labels)
    save_model(model, 'arrows_to_matchnext')

    label_func = lambda fcs: fcs.get_label_matches_prev('Limb annotation')
    points, labels = create_dataset(csvs, label_func)
    model = train_model(points, labels)
    save_model(model, 'arrows_to_matchprev')

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--chart_struct_csv', 
    #     default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/Conflict_-_Siromaru___Cranky_S11_arcade.csv',
    # )
    parser.add_argument(
        '--chart_struct_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/',
    )
    args.parse_args(parser)
    main()