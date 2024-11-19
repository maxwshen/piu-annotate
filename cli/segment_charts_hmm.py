import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import sys

from piu_annotate.formats.chart import ChartStruct
from piu_annotate import utils
from piu_annotate.segment.segment_hmm import segment_hmm


def segment_single_chart(csv: str):
    cs = ChartStruct.from_file(csv)
    num_sections = segment_hmm(cs)
    logger.info(f'{num_sections=}')
    return


def main():
    if 'csv' in args and args['csv'] is not None:
        # run single
        logger.info(f'Running single ...')
        segment_single_chart(args['csv'])
        return

    # run on folder
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    from collections import defaultdict
    dd = defaultdict(list)
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        try:
            cs = ChartStruct.from_file(inp_fn)
        except:
            logger.error(f'Failed to load {inp_fn}')
            sys.exit()

        sections = segment_hmm(cs)
        cs.metadata['Segments'] = [s.to_tuple() for s in sections]
        # cs.to_csv(inp_fn)

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Segments ChartStruct CSVs, updating metadata field
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()