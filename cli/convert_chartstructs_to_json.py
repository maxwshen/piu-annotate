import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct


# output folder should be /chart-json/, for compatibility with make_search_json.py
def main():
    folder = args['chartstruct_csv_folder']
    csvs = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(csvs)} csvs in {folder} ...')

    out_dir = os.path.join(folder, 'chart-json/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info(f'Writing to {out_dir=}')

    for csv in tqdm(csvs):
        out_fn = os.path.join(out_dir, os.path.basename(csv).replace('.csv', '.json'))
        if os.path.isfile(out_fn):
            continue

        cs = ChartStruct.from_file(csv)
        cjs = ChartJsStruct.from_chartstruct(cs)
        cjs.to_json(out_fn)

    logger.success(f'Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Converts a folder of ChartStructs to ChartJson
    """)
    parser.add_argument(
        '--chartstruct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/rayden-072924-arroweclipse-072824/lgbm-091624',
    )
    args.parse_args(parser)
    main()