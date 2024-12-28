import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import sys

from piu_annotate.formats.chart import ChartStruct
from piu_annotate import utils
from piu_annotate.segment.skills import hands


ok_hands = [
    'Ugly_Dee_-_Banya_Production_D18_ARCADE',
    'Ugly_Dee_-_Banya_Production_D17_ARCADE',
    'Come_to_Me_-_Banya_S17_INFOBAR_TITLE_ARCADE',
    'ESCAPE_-_D_AAN_D26_ARCADE',
    'Chimera_-_YAHPP_S23_ARCADE',
    'Chimera_-_YAHPP_D26_ARCADE',
    'Come_to_Me_-_Banya_S13_ARCADE',
    'Naissance_2_-_BanYa_D16_ARCADE',
    'Achluoias_-_D_AAN_D26_ARCADE',
    'Jump_-_BanYa_S16_ARCADE',
    'Gun_Rock_-_Banya_Production_S20_ARCADE',
    'Uh-Heung_-_DKZ_S22_ARCADE',
    'Love_is_a_Danger_Zone_2_-_FULL_SONG_-_-_Yahpp_S20_FULLSONG',
    'Love_is_a_Danger_Zone_2_-_FULL_SONG_-_-_Yahpp_D21_FULLSONG',
    'Love_is_a_Danger_Zone_2_Try_To_B.P.M_-_BanYa_D23_INFOBAR_TITLE_REMIX',
    'Love_is_a_Danger_Zone_2_Try_To_B.P.M_-_BanYa_S21_REMIX',
    'Hi-Bi_-_BanYa_D21_ARCADE',
    'Fire_Noodle_Challenge_-_Memme_S23_REMIX',
    'Slam_-_Novasonic_S18_INFOBAR_TITLE_ARCADE',
    'Bee_-_BanYa_D23_ARCADE',
    'Bee_-_BanYa_S17_INFOBAR_TITLE_ARCADE',
    'Another_Truth_-_Novasonic_D19_ARCADE',
]

def main():


    # run on folder
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        hands(cs)
        has_hands = any(cs.df['__hands'])

        if has_hands:
            shortname = cs.metadata["shortname"]
            if shortname not in ok_hands:
                logger.debug(f'Found hands')
                logger.debug(f'{cs.metadata["shortname"]}')
                print(cs.df[cs.df['__hands']])
                import code; code.interact(local=dict(globals(), **locals()))

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Checks ChartStruct with limb annotations implying use of hands.
        Serves as a way to sanity check predictions and efficiently discover
        prediction mistakes, as intentional hands are rare.
        However, this requires a list of stepcharts that actually have hands.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()