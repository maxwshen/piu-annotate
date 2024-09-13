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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.actors import Actor
from piu_annotate.ml.models import ModelSuite


def main():
    model_suite = ModelSuite()
    cs = ChartStruct.from_file(args['chart_struct_csv'])
    actor = Actor(cs, model_suite)

    fcs = featurizers.ChartStructFeaturizer(cs)

    true_labels = fcs.get_labels_from_limb_col('Limb annotation')

    # score true labels
    print(actor.score(true_labels))

    pred_limbs = actor.iterative_refine()
    fcs.evaluate(pred_limbs)

    pred_limbs = actor.flip_labels_by_score(pred_limbs)
    fcs.evaluate(pred_limbs)

    pred_limbs = actor.flip_jack_sections(pred_limbs)
    fcs.evaluate(pred_limbs)

    # annotate
    pred_coords = cs.get_prediction_coordinates()
    int_to_limb = {0: 'l', 1: 'r'}
    pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
    cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

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
        '--model.arrows_to_limb', 
        # default = '/home/maxwshen/piu-annotate/artifacts/models/091024/gbc-singles.pkl',
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/line-repeat/temp-singles.pkl'
    )
    parser.add_argument(
        '--model.arrowlimbs_to_limb', 
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/line-repeat/temp-withlimb-singles.pkl'
    )
    parser.add_argument(
        '--model.arrows_to_matchnext', 
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/label-matches-next/temp-singles.pkl'
    )
    parser.add_argument(
        '--model.arrows_to_matchprev', 
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/label-matches-prev/temp-singles.pkl'
        # default = '/home/maxwshen/piu-annotate/artifacts/models/091024/gbc-singles-withlimbcontext.pkl',
    )
    args.parse_args(parser)
    main()