import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import sys

from piu_annotate.formats.chart import ChartStruct
from piu_annotate import utils
from piu_annotate.segment.skills import annotate_skills


def segment_single_chart(csv: str):
    logger.info(f'Running on {csv=}')
    cs = ChartStruct.from_file(csv)
    annotate_skills(cs)
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
            'Tales_of_Pumpnia_-_Applesoda_S17_ARCADE.csv',
            # 'Phantom_-Intermezzo-_-_Banya_Production_S7_ARCADE.csv',
            # 'Rock_the_house_-_Matduke_D22_INFOBAR_TITLE_ARCADE.csv',
            # 'The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog_-_Doin_D24_ARCADE.csv',
            # 'Mopemope_-_LeaF_D25_ARCADE.csv',
            # 'Vacuum_-_Doin_S19_INFOBAR_TITLE_ARCADE.csv',
            # 'Papasito_(feat.__KuTiNA)_-_FULL_SONG_-_-_Yakikaze_&_Cashew_S19_FULLSONG.csv',
            # 'Papasito_(feat.__KuTiNA)_-_Yakikaze_&_Cashew_D21_ARCADE.csv',
            # 'Sudden_Appearance_Image_-_Blacklolita_S21_ARCADE.csv',
        ]
        chartstruct_files = [folder + f for f in chartstruct_files]

    from collections import defaultdict
    dd = defaultdict(list)
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        try:
            cs = ChartStruct.from_file(inp_fn)
        except:
            logger.error(f'Failed to load {inp_fn}')
            sys.exit()

        annotate_skills(cs)
        # cs.to_csv(inp_fn)

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Annotates ChartStructs with skill columns, writing or updating
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/The_End_of_the_World_ft._Skizzo_-_MonstDeath_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/BOOOM!!_-_RiraN_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()