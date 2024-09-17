"""
    Convert piucenter dijkstra files to chartstruct
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct

OUT_DIR = '/home/maxwshen/piu-annotate/output/chartstructs-piucenter-dijkstra-090124/'
Path(OUT_DIR).mkdir(parents = True, exist_ok = True)


def pcdf_to_chartstruct(inp_file: str, out_file: str, verbose: bool = False):
    if verbose:
        logger.info(f'Loading {inp_file} ...')
    pc_df = PiuCenterDataFrame(inp_file)

    cs = ChartStruct.from_piucenterdataframe(pc_df)
    try:
        cs.validate()
    except:
        logger.error(f'Failed to validate {inp_file=}')
        return
    
    if verbose:
        logger.info(f'Writing to {out_file}.')
    cs.df.to_csv(out_file)
    return


def main():
    if args['file']:
        pcdf_to_chartstruct(
            args['file'],
            os.path.join(OUT_DIR, os.path.basename(args['file'])),
            verbose = True,
        )
    else:
        # run all
        logger.info(f'Running all ...')

        inp_dir = '/home/maxwshen/piu-annotate/data/d_annotate/'
        fns = [fn for fn in os.listdir(inp_dir) if '_features.csv' not in fn]
        for fn in tqdm(fns):
            out_file = os.path.join(OUT_DIR, fn)
            if os.path.isfile(out_file):
                continue
            # print(fn)
            pcdf_to_chartstruct(os.path.join(inp_dir, fn), out_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', 
        # default = '/home/maxwshen/piu-annotate/data/d_annotate/video_out_c_-_Vospi_S22_arcade.csv'
    )
    args.parse_args(parser)
    main()