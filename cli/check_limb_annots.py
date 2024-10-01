"""
    Featurize
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import limbchecks

def main():
    folder = args['chart_struct_csv_folder']
    
    # crawl all subdirs for csvs
    csvs = [os.path.join(folder, fn) for fn in os.listdir(folder)
            if fn.endswith('.csv')]
    logger.info(f'Found {len(csvs)} csvs in {folder} ...')

    for csv in tqdm(csvs):
        logger.debug(csv)
        cs = ChartStruct.from_file(csv)
        found_idxs = limbchecks.check_unforced_doublestep(cs)

        if len(found_idxs) > 0:
            logger.warning(f'{csv}: Found {len(found_idxs)}')
            # import code; code.interact(local=dict(globals(), **locals()))

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Check limb annotations in ChartStruct CSVs
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-092124',
    )
    args.parse_args(parser)
    main()