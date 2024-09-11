"""
    Updates ChartStruct using a manually annotated chart json
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate import utils
from piu_annotate import hints


def main():
    cs_folder = args['chart_struct_csv_folder']
    csvs = [os.path.join(cs_folder, fn) for fn in os.listdir(cs_folder)
            if fn.endswith('.csv')]

    hint = hints.AlternateSoloArrows()

    for csv in csvs:
        print(csv)
        cs = ChartStruct.from_file(csv)
        sections = hints.apply_hint(cs, hint, min_lines = 3)

        for start_idx, end_idx in sections:
            ok, msg = hint.validate(cs.df, start_idx, end_idx)
            if not ok:
                logger.warning(f'Hint did not validate on {csv=}, {start_idx=}, {end_idx=}, {msg=}')
                import code; code.interact(local=dict(globals(), **locals()))

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/',
    )
    args.parse_args(parser)
    main()