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
            arrow_coords = cs.get_arrow_coordinates()
            int_to_limb = {0: 'l', 1: 'r'}
            pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
            cs.add_limb_annotations(arrow_coords, pred_limb_strs, 'Limb annotation')
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
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/rayden-072924-arroweclipse-072824/Over_The_Horizon_-_Yamajet_S11_ARCADE.csv',
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/main/Betrayer_-act.2-_-_msgoon_D15_ARCADE.csv'
    )
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/main/',
    )
    args.parse_args(
        parser, 
        '/home/maxwshen/piu-annotate/artifacts/models/120524/model-config.yaml'
    )
    main()