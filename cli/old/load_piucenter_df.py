import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct

OUT_DIR = '/home/maxwshen/piu-annotate/output/chart-json-piucenter-annot-070824/'


def pcdf_to_json(inp_file: str, out_file: str, verbose: bool = False):
    if verbose:
        logger.info(f'Loaded {inp_file} ...')
    pc_df = PiuCenterDataFrame(inp_file)

    cs = ChartStruct.from_piucenterdataframe(pc_df)
    try:
        cs.validate()
    except:
        logger.error(f'Failed to validate {inp_file=}')
        return

    cjss = ChartJsStruct.from_chartstruct(cs)

    cjss.to_json(out_file)
    if verbose:
        logger.info(f'Wrote to {out_file}.')
    return


def main():
    if args['file']:
        pcdf_to_json(args['file'])
    else:
        # run all
        logger.info(f'Running all ...')

        inp_dir = '/home/maxwshen/piu-annotate/data/d_annotate/'
        fns = [fn for fn in os.listdir(inp_dir) if '_features.csv' not in fn]
        for fn in tqdm(fns):
            out_file = os.path.join(
                OUT_DIR, 
                f'{os.path.basename(fn).replace('.csv', '')}.json'
            )
            if os.path.isfile(out_file):
                continue

            print(fn)
            pcdf_to_json(os.path.join(inp_dir, fn), out_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', 
        # default = '/home/maxwshen/piu-annotate/jupyter/Conflict - Siromaru + Cranky D24 arcade.csv'
    )
    args.parse_args(parser)
    main()