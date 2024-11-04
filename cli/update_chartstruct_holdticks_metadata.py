import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate import utils
from piu_annotate.formats.sscfile import StepchartSSC, SongSSC


def main():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    for cs_file in tqdm(chartstruct_files):
        cs = ChartStruct.from_file(os.path.join(cs_folder, cs_file))

        source_ssc = cs.metadata['ssc_file']
        desc_songtype = cs.metadata['DESCRIPTION'] + '_' + cs.metadata['SONGTYPE']

        stepchart_ssc = StepchartSSC.from_song_ssc_file(source_ssc, desc_songtype)

        _, holdticks, msg = stepchart_ssc_to_chartstruct(stepchart_ssc)

        cs.metadata['Hold ticks'] = holdticks

        # logger.debug(f'{source_ssc}, {desc_songtype}')
        cs.to_csv(os.path.join(cs_folder, cs_file))

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Updates metadata for a folder of ChartStruct CSVs,
        by recomputing metadata from source .ssc file. 
        Used in particular to update HoldTick info.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    args.parse_args(parser)
    main()