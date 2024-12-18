import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct


def main():
    song_ssc_file = args['song_ssc_file']
    description_songtype = args['description_songtype']
    logger.info(f'Loading {song_ssc_file} - {description_songtype} ...')
    stepchart = StepchartSSC.from_song_ssc_file(
        song_ssc_file,
        description_songtype,
    )

    cs_df, holdticks, message = stepchart_ssc_to_chartstruct(
        stepchart,
        debug = True
    )
    logger.info(f'{message=}')

    cs_df['Line'] = [f'`{line}' for line in cs_df['Line']]
    cs_df['Line with active holds'] = [f'`{line}' for line in cs_df['Line with active holds']]

    cs = ChartStruct.from_stepchart_ssc(stepchart)

    logger.debug(f'Created ChartStruct -- save it, or continue to try conversion to json')
    import code; code.interact(local=dict(globals(), **locals()))
    cjs = ChartJsStruct.from_chartstruct(cs)

    # write
    if message != 'success':
        out_dir = '/home/maxwshen/piu-annotate/output/problematic-df'
        out_file = os.path.join(out_dir, stepchart.shortname() + '.csv')
        cs_df.to_csv(out_file)
        logger.info(f'Wrote problematic df to {out_file}')

    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Debug StepchartSSC -> ChartStruct -> ChartJson
        """
    )
    parser.add_argument(
        '--song_ssc_file', 
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/12 - PRIME/1430 - Scorpion King/1430 - Scorpion King.ssc'
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/13 - PRIME 2/1594 - Cross Time/1594 - Cross Time.ssc'
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/14 - XX/1689 - Over The Horizon/1689 - Over The Horizon.ssc'
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/14 - XX/1695 - Phalanx RS2018 edit/1695 - Phalanx RS2018 edit.ssc'
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/10 - FIESTA 2/(1) 13A2 - [Remix] Infinity RMX/13A2 - [Remix] Infinity RMX.ssc'
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/09 - FIESTA EX/1160 - Jonathan\'s Dream/1160 - Jonathan\'s Dream.ssc',
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/14 - XX/1698 - Life is PIANO/1698 - Life is PIANO.ssc'
        # default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/16 - PHOENIX/18120 - Amor Fati/Amor Fati.ssc'
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/14 - XX/1629 - Tales of Pumpnia/1629 - Tales of Pumpnia.ssc'
    )
    parser.add_argument(
        '--description_songtype',
        # default = 'D19_REMIX',
        # default = 'S8_ARCADE',
        # default = 'D21_ARCADE',
        # default = 'D23_ARCADE',
        default = 'S17_ARCADE',
    )
    args.parse_args(parser)
    main()