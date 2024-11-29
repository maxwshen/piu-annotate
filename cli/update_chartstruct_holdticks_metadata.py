import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys
import yaml

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate import utils
from piu_annotate.formats.sscfile import StepchartSSC, SongSSC
from piu_annotate.formats.nps import annotate_enps


def main():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    # load __cs_to_manual_json.yaml
    # assumes that cs_folder is like /home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/
    # so that parent_dir is like /home/maxwshen/piu-annotate/artifacts/chartstructs/092424/,
    # which holds all chartstruct CSVs, and contains __cs_to_manual_json.yaml
    parent_dir = Path(cs_folder).parent
    cs_to_manual_fn = os.path.join(parent_dir, '__cs_to_manual_json.yaml')
    logger.info(f'Attempting to load {cs_to_manual_fn=} ...')
    with open(cs_to_manual_fn, 'r') as f:
        cs_to_manual = yaml.safe_load(f)
    logger.info(f'Found cs_to_manual with {len(cs_to_manual)} entries')

    rerun_all = args.setdefault('rerun_all', False)

    if args.setdefault('debug', False):
        chartstruct_files = [
            'Ultimatum_-_Cosmograph_S21_ARCADE.csv'
        ]

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        try:
            cs = ChartStruct.from_file(inp_fn)
        except:
            logger.error(f'Failed to load {inp_fn}')
            sys.exit()

        # update hold tick counts
        if args.setdefault('holdticks', False) or rerun_all:
            source_ssc = cs.metadata['ssc_file']
            desc_songtype = cs.metadata['DESCRIPTION'] + '_' + cs.metadata['SONGTYPE']
            stepchart_ssc = StepchartSSC.from_song_ssc_file(source_ssc, desc_songtype)
            _, holdticks, msg = stepchart_ssc_to_chartstruct(stepchart_ssc)
            cs.metadata['Hold ticks'] = holdticks

        # annotate effective nps
        if args.setdefault('enps_annotations', False) or rerun_all:
            enps_annots = annotate_enps(cs)
            cs.metadata['eNPS annotations'] = enps_annots
            import code; code.interact(local=dict(globals(), **locals()))

        # fix/edit lines that occur after LASTSECONDHINT
        # this impacts dement d24, mental rider d22
        if args.setdefault('clip_long_chart_errors', False) or rerun_all:
            if 'LASTSECONDHINT' in cs.metadata:
                lastsecondhint = float(cs.metadata['LASTSECONDHINT'])
                if max(cs.df['Time']) > 400:
                    logger.debug(f'{cs_file=}')
                    cs.df['Time'] = cs.df['Time'].clip(upper = lastsecondhint)

        # annotate which limb annotations are manual
        # builds the path to CSV that is used as input for limb prediction
        if args.setdefault('add_manual_limb_annot_flag', False) or rerun_all:
            limb_annot_inp_csv = os.path.join(parent_dir, cs_file)
            cs.metadata['Manual limb annotation'] = bool(limb_annot_inp_csv in cs_to_manual)

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
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-112624/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    args.parse_args(parser)
    main()