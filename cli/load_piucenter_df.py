import argparse
import os
from hackerargs import args
from loguru import logger

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct

def main():
    file = args['file']
    logger.info(f'Loaded {file} ...')

    pc_df = PiuCenterDataFrame(file)
    cs = ChartStruct.from_piucenterdataframe(pc_df)

    cjss = ChartJsStruct.from_chartstruct(cs)

    out_file = os.path.join('/home/maxwshen/piu-annotate/output/chart-json', 'conflict-d24.json')
    cjss.to_json(out_file)
    logger.info(f'Wrote to {out_file}.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', 
        default = '/home/maxwshen/piu-annotate/jupyter/Conflict - Siromaru + Cranky D24 arcade.csv'
    )
    args.parse_args(parser)
    main()