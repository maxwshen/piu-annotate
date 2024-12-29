import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate import utils
from piu_annotate.segment.segment import segmentation
from piu_annotate.segment.segment_breaks import get_segment_metadata


def segment_single_chart(csv: str):
    cs = ChartStruct.from_file(csv)
    num_sections = segmentation(cs, debug = True)
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

    debug = args.setdefault('debug', False)
    if debug:
        folder = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/'
        chartstruct_files = [
            'Gargoyle_-_FULL_SONG_-_-_Sanxion7_D25_FULLSONG.csv',
            # 'The_End_of_the_World_ft._Skizzo_-_MonstDeath_D22_ARCADE.csv',
            # 'Altale_-_sakuzyo_D19_ARCADE.csv',
        ]
        chartstruct_files = [folder + f for f in chartstruct_files]

    rerun_all = args.setdefault('rerun_all', False)
    stats = defaultdict(int)

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        try:
            cs = ChartStruct.from_file(inp_fn)
        except:
            logger.error(f'Failed to load {inp_fn}')
            sys.exit()

        # if 'Segments' in cs.metadata:
        #     continue
        # print(cs_file)
        if not rerun_all and 'Segments' in cs.metadata:
            stats['skipped'] += 1
            continue

        sections = segmentation(cs, debug = debug)
        cs.metadata['Segments'] = [s.to_tuple() for s in sections]
        cs.metadata['Segment metadata'] = [get_segment_metadata(cs, s) for s in sections]
        stats['segmented'] += 1
        cs.to_csv(inp_fn)

    logger.debug(stats)
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
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Nyarlathotep_-_nato_S21_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/BOOOM!!_-_RiraN_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()