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
from piu_annotate.ml.predictor import predict


def guess_singles_or_doubles_from_filename(filename: str) -> str:
    """ """
    basename = os.path.basename(filename)
    sord = basename.split('_')[-2][0]
    if sord == 'S':
        return 'singles'
    elif sord == 'D':
        return 'doubles'
    return 'unsure'


def main():
    csv_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {csv_folder=} ...')
    singles_or_doubles = args['singles_or_doubles']
    logger.info(f'Using {singles_or_doubles} ...')

    model_suite = ModelSuite(singles_or_doubles)
    logger.info(f'Using {args["model.name"]} ...')

    csvs = [os.path.join(csv_folder, fn) for fn in os.listdir(csv_folder)
            if fn.endswith('.csv')]
    # subset csvs to singles or doubles
    csv_sord = [csv for csv in csvs
                if guess_singles_or_doubles_from_filename(csv) in [singles_or_doubles, 'unsure']]
    logger.info(f'Found {len(csv_sord)} csvs')

    # load __cs_to_manual_json.yaml
    cs_to_manual_fn = os.path.join(csv_folder, '__cs_to_manual_json.yaml')
    with open(cs_to_manual_fn, 'r') as f:
        cs_to_manual = yaml.safe_load(f)
    logger.info(f'Found cs_to_manual with {len(cs_to_manual)} entries ...')

    out_dir = os.path.join(csv_folder, args['model.name'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    stats = defaultdict(int)
    for csv in tqdm(csv_sord):
        cs: ChartStruct = ChartStruct.from_file(csv)
        if cs.singles_or_doubles() != singles_or_doubles:
            continue

        out_fn = os.path.join(out_dir, os.path.basename(csv))
        if os.path.isfile(out_fn):
            continue

        try:
            if csv in cs_to_manual:
                logger.debug(f'updating with manual - {csv}')
                # if existing manual, load that json, and update cs with json
                manual_json = cs_to_manual[csv]
                cjs = ChartJsStruct.from_json(manual_json)
                cs.update_from_manual_json(cjs)
                stats['N updated from manual'] += 1
            else:
                logger.debug(f'predicting - {csv}')
                cs, fcs, pred_limbs = predict(cs, model_suite)
            
                # annotate
                arrow_coords = cs.get_arrow_coordinates()
                int_to_limb = {0: 'l', 1: 'r'}
                pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
                cs.add_limb_annotations(arrow_coords, pred_limb_strs, 'Limb annotation')
                stats['N predicted'] += 1
        except Exception as e:
            logger.error(str(e))
            logger.error(csv)
            import code; code.interact(local=dict(globals(), **locals()))

        # save to file
        cs.to_csv(out_fn)

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Predicts limbs on chart structs without existing limb annotations
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/r0729-ae0728-092124',
    )
    parser.add_argument(
        '--singles_or_doubles', 
        default = 'singles',
    )
    args.parse_args(
        parser, 
        '/home/maxwshen/piu-annotate/artifacts/models/092124/model-config.yaml'
    )
    main()