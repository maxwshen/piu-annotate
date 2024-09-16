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
from numpy.typing import NDArray
from operator import itemgetter
import itertools
from collections import defaultdict
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.actors import Actor
from piu_annotate.ml.models import ModelSuite


def predict(
    cs: ChartStruct, 
    model_suite: ModelSuite,
    verbose: bool = False,
) -> ChartStruct:
    """ Use actor to predict limb annotations for `cs` """
    actor = Actor(cs, model_suite)
    fcs = featurizers.ChartStructFeaturizer(cs)

    true_labels = fcs.get_labels_from_limb_col('Limb annotation')

    score_to_limbs = dict()

    # score true labels
    if verbose:
        logger.info(f'Score of true labels: {actor.score(true_labels):.3f}')

    pred_limbs = actor.iterative_refine()
    score_to_limbs[actor.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, iterative refine: {actor.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs)

    pred_limbs = actor.flip_labels_by_score(pred_limbs)
    score_to_limbs[actor.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip: {actor.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs)

    pred_limbs = actor.flip_jack_sections(pred_limbs)
    score_to_limbs[actor.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip jacks: {actor.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs)

    pred_limbs = actor.beam_search(score_to_limbs[max(score_to_limbs)], width = 5, n_iter = 3)
    score_to_limbs[actor.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, beam search: {actor.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    for _ in range(1):
        pred_limbs = actor.fix_double_doublestep(pred_limbs)
        score_to_limbs[actor.score(pred_limbs)] = pred_limbs.copy()
        if verbose:
            logger.info(f'Score, fix double doublestep: {actor.score(pred_limbs):.3f}')
            fcs.evaluate(pred_limbs, verbose = True)

    # best score
    if verbose:
        best_score = max(score_to_limbs.keys())
        logger.success(f'Found {best_score=:.3f}')

    pred_limbs = actor.detect_impossible_multihit(score_to_limbs[max(score_to_limbs)])
    if verbose:
        logger.info(f'Score, fix impossible multihit: {actor.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    return cs, fcs, pred_limbs


def accuracy(fcs: featurizers.ChartStructFeaturizer, pred_limbs: NDArray):
    eval_d = fcs.evaluate(pred_limbs, verbose = False)
    return eval_d['accuracy']


def main():
    model_suite = ModelSuite()

    if not args['run_folder']:
        cs = ChartStruct.from_file(args['chart_struct_csv'])
        cs, fcs, pred_limbs = predict(cs, model_suite, verbose = True)

        # annotate
        pred_coords = cs.get_prediction_coordinates()
        int_to_limb = {0: 'l', 1: 'r'}
        pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
        cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

        cs.df['Error'] = (
            cs.df['__pred limb final'] != cs.df['Limb annotation']
        ).astype(int)

        basename = os.path.basename(args['chart_struct_csv'])
        out_fn = f'temp/{basename}'
        cs.to_csv(out_fn)
        logger.info(f'Saved to {out_fn}')

    else:
        csv_folder = args['chart_struct_csv_folder']
        singles_or_doubles = args['singles_or_doubles']
        csvs = [os.path.join(csv_folder, fn) for fn in os.listdir(csv_folder)
                if fn.endswith('.csv')]
        
        dd = defaultdict(list)
        for csv in tqdm(csvs):
            cs = ChartStruct.from_file(csv)
            if cs.singles_or_doubles() != singles_or_doubles:
                continue
            # logger.info(csv)
            cs, fcs, pred_limbs = predict(cs, model_suite)
            
            dd['File'].append(os.path.basename(csv))
            dd['Accuracy'].append(accuracy(fcs, pred_limbs))

        stats_df = pd.DataFrame(dd)
        stats_df.to_csv('temp/stats.csv')

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chart_struct_csv', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/Rising_Star_-_M2U_S17_arcade.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/Conflict_-_Siromaru___Cranky_S11_arcade.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/Headless_Chicken_-_r300k_S21_arcade.csv'
    )
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/',
    )
    parser.add_argument(
        '--singles_or_doubles', 
        default = 'singles',
    )
    parser.add_argument(
        '--run_folder', 
        default = False,
    )
    args.parse_args(
        parser, 
        '/home/maxwshen/piu-annotate/artifacts/models/091324/singles/model-config.yaml'
    )
    main()