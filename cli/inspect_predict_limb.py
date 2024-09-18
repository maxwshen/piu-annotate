import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import itertools
from collections import defaultdict
import pandas as pd
import yaml

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.tactics import Tactician
from piu_annotate.ml.models import ModelSuite
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.utils import make_dir

def predict(
    cs: ChartStruct, 
    model_suite: ModelSuite,
    verbose: bool = False,
) -> ChartStruct:
    """ Use tactician to predict limb annotations for `cs` """
    tactics = Tactician(cs, model_suite)
    fcs = featurizers.ChartStructFeaturizer(cs)

    score_to_limbs = dict()

    pred_limbs = tactics.initial_predict()
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, initial pred: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.flip_labels_by_score(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.flip_jack_sections(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip jacks: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.beam_search(score_to_limbs[max(score_to_limbs)], width = 5, n_iter = 3)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, beam search: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.fix_double_doublestep(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, fix double doublestep: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    # best score
    if verbose:
        best_score = max(score_to_limbs.keys())
        logger.success(f'Found {best_score=:.3f}')

    pred_limbs = tactics.detect_impossible_multihit(score_to_limbs[max(score_to_limbs)])
    if verbose:
        logger.info(f'Score, fix impossible multihit: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    return cs, fcs, pred_limbs


def main():
    csv = args['chart_struct_csv']
    logger.info(f'Using {csv=} ...')

    cs: ChartStruct = ChartStruct.from_file(csv)

    singles_or_doubles = cs.singles_or_doubles()
    model_suite = ModelSuite(singles_or_doubles)
    logger.info(f'Using {args["model.name"]} ...')

    # load __cs_to_manual_json.yaml
    csv_folder = args['chart_struct_csv_folder']
    cs_to_manual_fn = os.path.join(csv_folder, '__cs_to_manual_json.yaml')
    with open(cs_to_manual_fn, 'r') as f:
        cs_to_manual = yaml.safe_load(f)
    logger.info(f'Found cs_to_manual with {len(cs_to_manual)} entries ...')

    try:
        if csv in cs_to_manual:
            logger.debug(f'updating with manual - {csv}')
            # if existing manual, load that json, and update cs with json
            manual_json = cs_to_manual[csv]
            cjs = ChartJsStruct.from_json(manual_json)
            cs.update_from_manual_json(cjs)
        else:
            logger.debug(f'predicting - {csv}')
            cs, fcs, pred_limbs = predict(cs, model_suite)
        
            # annotate
            pred_coords = cs.get_prediction_coordinates()
            int_to_limb = {0: 'l', 1: 'r'}
            pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
            cs.add_limb_annotations(pred_coords, pred_limb_strs, 'Limb annotation')
    except Exception as e:
        logger.error(str(e))
        logger.error(csv)
        import code; code.interact(local=dict(globals(), **locals()))

    import code; code.interact(local=dict(globals(), **locals()))

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Debug predict limbs on chart struct without existing limb annotation
    """)
    parser.add_argument(
        '--chart_struct_csv', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/rayden-072924-arroweclipse-072824/Over_The_Horizon_-_Yamajet_S11_ARCADE.csv',
    )
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/rayden-072924-arroweclipse-072824/',
    )
    args.parse_args(
        parser, 
        '/home/maxwshen/piu-annotate/artifacts/models/091624/model-config.yaml'
    )
    main()