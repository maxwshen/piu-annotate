import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys
import pandas as pd
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.difficulty import featurizers
from piu_annotate.difficulty.models import DifficultyStepchartModelPredictor
from cli.difficulty.train_difficulty_predictor import build_full_stepchart_dataset

def main():
    # run on folder
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    dmp = DifficultyStepchartModelPredictor()
    dmp.load_models()

    dataset = build_full_stepchart_dataset()
    file_to_x = {file: x for file, x in zip(dataset['files'], dataset['x'])}

    dd = defaultdict(list)
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        x = file_to_x[cs_file]
        x = x.reshape(1, -1)
        pred_level = dmp.predict(x, cs.singles_or_doubles())

        smd = cs.metadata['Segment metadata']
        dd['shortname'].append(cs.metadata['shortname'])
        dd['max segment level'].append(max([md['level'] for md in smd]))
        dd['segment levels'].append([md['level'] for md in smd])
        dd['chart level'].append(cs.get_chart_level())
        dd['pred level'].append(pred_level)
        dd['sord'].append(cs.singles_or_doubles())
        
    df = pd.DataFrame(dd)
    df.to_csv('/home/maxwshen/piu-annotate/artifacts/analysis/segment-difficulty.csv')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/',
    )
    args.parse_args(parser)
    main()