import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray
from collections import defaultdict
import pandas as pd

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict
from piu_annotate.reasoning.reasoners import PatternReasoner


def main():
    if not args['run_folder']:
        csv = args['chart_struct_csv']
        logger.info(f'Using {csv=}')

        cs: ChartStruct = ChartStruct.from_file(args['chart_struct_csv'])
        reasoner = PatternReasoner(cs, verbose = True)

        logger.info(f'Checking limb reuse pattern ...')
        stats = reasoner.check()
        stats = reasoner.check(breakpoint = True)
        for k, v in stats.items():
            logger.debug(f'{k}: {v}')

        logger.info(f'Checking limb proposals ...')
        stats = reasoner.check_proposals()
        stats = reasoner.check_proposals(breakpoint = True)
        for k, v in stats.items():
            logger.debug(f'{k}: {v}')

        logger.info('Done')
        import code; code.interact(local=dict(globals(), **locals()))

    else:
        csv_folder = args['manual_chart_struct_folder']

        # crawl all subdirs for csvs
        csvs = []
        dirpaths = set()
        for dirpath, _, files in os.walk(csv_folder):
            for file in files:
                if file.endswith('.csv') and 'exclude' not in dirpath and 'piucenter' not in dirpath:
                    csvs.append(os.path.join(dirpath, file))
                    dirpaths.add(dirpath)
        logger.info(f'Found {len(csvs)} csvs in {len(dirpaths)} directories ...')
        # csvs = [os.path.join(csv_folder, fn) for fn in os.listdir(csv_folder)
        #         if fn.endswith('.csv')]
        
        dd = defaultdict(list)
        for csv in tqdm(csvs):
            cs = ChartStruct.from_file(csv)
            reasoner = PatternReasoner(cs)
            stats = reasoner.check()

            if len(stats['Time of violations']):
                dd['csv'].append(csv)
                for k, v in stats.items():
                    dd[k].append(v)

                logger.warning(csv)
                logger.warning(stats['Time of violations'])

            # reasoner.check_proposals()

        stats_df = pd.DataFrame(dd)
        stats_df.to_csv('temp/check_pattern_reasoning_violations.csv')
        print(stats_df.describe())

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
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/092124/Nyarlathotep_-_SHORT_CUT_-_-_Nato_D24_SHORTCUT.csv',
        default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/092424/Amphitryon_-_Gentle_Stick_S18_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/Rising_Star_-_M2U_S17_arcade.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/Conflict_-_Siromaru___Cranky_S11_arcade.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/piucenter-manual-090624/Headless_Chicken_-_r300k_S21_arcade.csv'
    )
    parser.add_argument(
        '--manual_chart_struct_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/',
    )
    parser.add_argument(
        '--run_folder', 
        default = False,
    )
    args.parse_args(parser)
    main()