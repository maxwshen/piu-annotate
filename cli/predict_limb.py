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
    cs = ChartStruct.from_file(args['chart_struct_csv'])
    singles_doubles = cs.singles_or_doubles()

    # load models
    with open(args['model_pkl'], 'rb') as f:
        model_nolimb: GradientBoostingClassifier = pickle.load(f)

    with open(args['model_limbcontext_pkl'], 'rb') as f:
        model_withlimb: GradientBoostingClassifier = pickle.load(f)

    """
        First prediction pass: Without limb context
    """
    # featurize
    fcs = featurizers.FeaturizedChartStruct(cs)
    points = fcs.get_prediction_input_with_context()
    labels = fcs.labels_to_array()
    logger.info(f'Found dataset shape {points.shape}')

    pred_limbs = model_nolimb.predict(points)
    int_to_limb = {0: 'l', 1: 'r'}
    pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
    pred_limb_probs = model_nolimb.predict_proba(points)[:, -1]

    # compare
    accuracy = np.sum(labels == pred_limbs) / len(labels)
    logger.info(f'{accuracy=}')

    """
        Second prediction pass: Use predicted limb context
    """
    pred_coords = cs.get_prediction_coordinates()

    num_iters = 5
    for __i in range(num_iters):
        cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb v1')

        # weight
        weight = (__i + 1) / (num_iters + 1)

        adj_limb_probs = weight * pred_limb_probs + (1 - weight) * 0.5 * np.ones(pred_limb_probs.shape)

        # fcs.update_with_pred_limb_probs(pred_limb_probs, pred_limb_probs < 0.9)
        fcs.update_with_pred_limb_probs(adj_limb_probs, adj_limb_probs < 0.9)
        points_withlimb = fcs.get_prediction_input_with_context(
            use_limb_features = True
        )

        logger.info(f'Found dataset shape {points_withlimb.shape}')

        pred_limbs = model_withlimb.predict(points_withlimb)
        pred_limb_probs = model_withlimb.predict_proba(points_withlimb)[:, -1]

        # compare
        accuracy = np.sum(labels == pred_limbs) / len(labels)
        logger.info(f'{accuracy=}')
        pred_limb_strs = [int_to_limb[i] for i in pred_limbs]

    cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

    print('Error indices:')
    print(np.where(labels != pred_limbs))

    cs.df.to_csv('temp/conflict-s11.csv')
    import code; code.interact(local=dict(globals(), **locals()))
    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chart_struct_csv', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/Conflict_-_Siromaru___Cranky_S11_arcade.csv',
    )
    parser.add_argument(
        '--model_pkl', 
        # default = '/home/maxwshen/piu-annotate/artifacts/models/091024/gbc-singles.pkl',
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/line-repeat/temp-singles.pkl'
    )
    parser.add_argument(
        '--model_limbcontext_pkl', 
        # default = '/home/maxwshen/piu-annotate/artifacts/models/091024/gbc-singles-withlimbcontext.pkl',
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/line-repeat/temp-withlimb-singles.pkl'
    )
    args.parse_args(parser)
    main()