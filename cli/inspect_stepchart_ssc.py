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

    cs_df, message = stepchart_ssc_to_chartstruct(
        stepchart,
        debug = True
    )
    logger.info(f'{message=}')

    cs_df['Line'] = [f'`{line}' for line in cs_df['Line']]
    cs_df['Line with active holds'] = [f'`{line}' for line in cs_df['Line with active holds']]

    cs = ChartStruct.from_stepchart_ssc(stepchart)
    cjs = ChartJsStruct.from_chartstruct(cs)

    # write
    if message != 'success':
        out_dir = '/home/maxwshen/piu-annotate/output/problematic-df'
        out_file = os.path.join(out_dir, stepchart.shortname() + '.csv')
        cs_df.to_csv(out_file)
        logger.info(f'Wrote problematic df to {out_file}')
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
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/13 - PRIME 2/1594 - Cross Time/1594 - Cross Time.ssc'
    )
    parser.add_argument(
        '--description_songtype',
        default = 'D22_ARCADE',
    )
    args.parse_args(parser)
    main()