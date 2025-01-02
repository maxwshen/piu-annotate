import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys
from collections import defaultdict
import pandas as pd

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.sscfile import StepchartSSC, SongSSC
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct


def notecount(cs: ChartStruct) -> int:
    """ Attempt to count notes in `cs` """
    # holdticks = cs.metadata['Hold ticks']

    source_ssc = cs.metadata['ssc_file']
    desc_songtype = cs.metadata['DESCRIPTION'] + '_' + cs.metadata['SONGTYPE']
    stepchart_ssc = StepchartSSC.from_song_ssc_file(source_ssc, desc_songtype)
    _, holdticks, msg = stepchart_ssc_to_chartstruct(stepchart_ssc)

    total_holdticks = sum([ht[2] for ht in holdticks])

    # num_lines_with_1 = sum(['1' in l and '3' not in l for l in cs.df['Line']])
    num_lines_with_1 = sum(['1' in l for l in cs.df['Line']])

    total = num_lines_with_1 + total_holdticks
    debug = args.setdefault('debug', False)
    if debug:
        logger.debug(holdticks[:8])
        logger.debug(len(holdticks))
    return total


def annotate_segment_similarity():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
            'Doppelganger_-_MonstDeath_D26_ARCADE.csv',
            'Nyan-turne_(feat._KuTiNA)_-_Cashew__Castellia_D21_ARCADE.csv',
            # 'GLORIA_-_Croire_D21_ARCADE.csv',
            # 'Final_Audition_2__-_SHORT_CUT_-_-_Banya_S17_SHORTCUT.csv',
        ]
        chartstruct_files = [os.path.join(cs_folder, f) for f in chartstruct_files]

    dd = defaultdict(list)
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        count = notecount(cs)
        if debug:
            logger.debug((cs.metadata['shortname'], count))
            import code; code.interact(local=dict(globals(), **locals()))

        dd['Shortname (piucenter)'].append(cs.metadata['shortname'])
        dd['Song'].append(cs.metadata['TITLE'])
        dd['Type'].append(cs.singles_or_doubles())
        dd['Level'].append(cs.metadata['METER'])
        dd['Note Count'].append(count)
        dd['Pack'].append(cs.metadata['pack'])

    out_fn = '/home/maxwshen/piu-annotate/artifacts/notecounts/notecounts.csv'
    df = pd.DataFrame(dd)
    df.to_csv(out_fn)
    logger.info(f'Wrote to {out_fn}')
    return


def main():
    annotate_segment_similarity()

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Count total "notes" in chart, summing lines and hold ticks.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/main/lgbm-120524/',
    )
    args.parse_args(parser)
    main()