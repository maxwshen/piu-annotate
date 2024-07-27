"""
    Crawls PIU-Simfiles folder
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct


def main():
    stepchart_ssc_file = args['stepchart_ssc_file']

    stepchart = StepchartSSC.from_file(stepchart_ssc_file)

    cs_df, num_bad_lines = stepchart_ssc_to_chartstruct(
        stepchart,
        debug = True
    )
    logger.info(f'{num_bad_lines=}')

    cs_df['Line'] = [f'`{line}' for line in cs_df['Line']]
    cs_df['Line with active holds'] = [f'`{line}' for line in cs_df['Line with active holds']]

    out_dir = '/home/maxwshen/piu-annotate/output/problematic-df'
    basename = os.path.basename(stepchart_ssc_file).replace('.ssc', '')
    out_file = os.path.join(out_dir, basename + '.csv')
    cs_df.to_csv(out_file)
    logger.success(f'Wrote to {out_file}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
        """
    )
    parser.add_argument(
        '--stepchart_ssc_file', 
        default = '/home/maxwshen/piu-annotate/output/problematic-ssc/Bon_Bon_Chocolat_-_EVERGLOW_S16_ARCADE.ssc'
    )
    args.parse_args(parser)
    main()