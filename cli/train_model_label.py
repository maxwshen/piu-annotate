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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers


def main():
    folder = args['chart_struct_folder']
    csvs = [os.path.join(folder, fn) for fn in os.listdir(folder)
            if fn.endswith('.csv')]

    singles_doubles = args.setdefault('singles_or_doubles', 'singles')

    all_points, all_labels = [], []
    n_csvs = 0
    for csv in tqdm(csvs):
        cs = ChartStruct.from_file(csv)
        if cs.singles_or_doubles() == singles_doubles:
            
            fcs = featurizers.ChartStructFeaturizer(cs)
            labels = fcs.get_labels_from_limb_col('Limb annotation')
            points = fcs.featurize_arrowlimbs_with_context(labels)

            all_points.append(points)
            all_labels.append(labels)
            n_csvs += 1
    logger.info(f'Featurized {n_csvs} ChartStruct csvs ...')

    points = np.concatenate(all_points)
    labels = np.concatenate(all_labels)
    logger.info(f'Found dataset shape {points.shape}')

    # cs = ChartStruct.from_file(args['chart_struct_csv'])
    # points, labels = featurizers.featurize_chart_struct(cs)
    # logger.info(f'Found {len(points)=}')

    # train/test split
    train_x, test_x, train_y, test_y = train_test_split(points, labels)

    model = GradientBoostingClassifier()
    model.fit(train_x, train_y)

    print(model.score(train_x, train_y))
    print(model.score(test_x, test_y))

    out_dir = args.setdefault('out_dir', '/home/maxwshen/piu-annotate/artifacts/models/temp')
    out_fn = os.path.join(out_dir, f'{singles_doubles}-arrowlimbs_to_limb.pkl')
    with open(out_fn, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Saved model to {out_fn}')

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