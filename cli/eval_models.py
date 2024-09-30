import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray
from collections import defaultdict
import pandas as pd

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict


def accuracy(fcs: featurizers.ChartStructFeaturizer, pred_limbs: NDArray):
    eval_d = fcs.evaluate(pred_limbs, verbose = False)
    return eval_d['accuracy-float']


def main():
    if not args['run_folder']:
        csv = args['chart_struct_csv']
        logger.info(f'Using {csv=}')
        cs: ChartStruct = ChartStruct.from_file(args['chart_struct_csv'])
        singles_or_doubles = cs.singles_or_doubles()
        model_suite = ModelSuite(singles_or_doubles)

        cs, fcs, pred_limbs = predict(cs, model_suite, verbose = True)

        # annotate
        arrow_coords = cs.get_arrow_coordinates()
        int_to_limb = {0: 'l', 1: 'r'}
        pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
        cs.add_limb_annotations(arrow_coords, pred_limb_strs, '__pred limb final')

        cs.df['Error'] = (
            cs.df['__pred limb final'] != cs.df['Limb annotation']
        ).astype(int)

        basename = os.path.basename(args['chart_struct_csv'])
        out_fn = f'temp/{basename}'
        cs.to_csv(out_fn)
        logger.info(f'Saved to {out_fn}')

    else:
        csv_folder = args['manual_chart_struct_folder']
        singles_or_doubles = args['singles_or_doubles']
        logger.info(f'Running {singles_or_doubles} ...')
        model_suite = ModelSuite(singles_or_doubles)

        # crawl all subdirs for csvs
        csvs = []
        dirpaths = set()
        for dirpath, _, files in os.walk(csv_folder):
            for file in files:
                if file.endswith('.csv') and 'exclude' not in dirpath:
                    csvs.append(os.path.join(dirpath, file))
                    dirpaths.add(dirpath)
        logger.info(f'Found {len(csvs)} csvs in {len(dirpaths)} directories ...')
        # csvs = [os.path.join(csv_folder, fn) for fn in os.listdir(csv_folder)
        #         if fn.endswith('.csv')]
        
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
        stats_df.to_csv(f'temp/stats-{singles_or_doubles}.csv')

        logger.info(stats_df['Accuracy'].describe())

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Evaluates models on chart structs with existing limb annotations
    """)
    parser.add_argument(
        '--chart_struct_csv', 
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/091924/Indestructible_-_Matduke_D22_ARCADE.csv'
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/092424/Feel_My_Happiness_-_3R2_D21_ARCADE.csv',
        default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/092124/Nyarlathotep_-_SHORT_CUT_-_-_Nato_D24_SHORTCUT.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/Rising_Star_-_M2U_S17_arcade.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/Conflict_-_Siromaru___Cranky_S11_arcade.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/Headless_Chicken_-_r300k_S21_arcade.csv'
    )
    parser.add_argument(
        '--manual_chart_struct_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/',
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
        '/home/maxwshen/piu-annotate/artifacts/models/092124/model-config.yaml'
    )
    main()