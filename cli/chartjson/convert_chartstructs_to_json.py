import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.utils import make_basename_url_safe


# output folder should be /chart-json/, for compatibility with make_search_json.py
def main():
    folder = args['chart_struct_csv_folder']
    csvs = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(csvs)} csvs in {folder} ...')

    out_dir = os.path.join(folder, 'chart-json/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info(f'Writing to {out_dir=}')

    rerun_all = args.setdefault('rerun_all', False)
    update_metadata_only = args.setdefault('update_metadata_only', False)
    assert not rerun_all or not update_metadata_only, 'only one can be true'
    logger.info(f'Using {rerun_all=}, {update_metadata_only=}')

    for csv in tqdm(csvs):
        basename = make_basename_url_safe(os.path.basename(csv).replace('.csv', '.json'))
        out_fn = os.path.join(out_dir, basename)

        if update_metadata_only:
            if not os.path.isfile(out_fn):
                logger.error(f'Attempted to update metadata only, but {out_fn} does not exist. Do a fresh run first')
            cjs = ChartJsStruct.from_json(out_fn)
            cs = ChartStruct.from_file(csv)
            cjs.update_metadata(cs)
            cjs.to_json(out_fn)

        else:
            # fresh run: can rerun all, or skip finished files
            if os.path.isfile(out_fn) and not rerun_all:
                continue

            cs = ChartStruct.from_file(csv)
            try:
                cjs = ChartJsStruct.from_chartstruct(cs)
                cjs.to_json(out_fn)
            except Exception as e:
                logger.error(str(e))
                logger.error(csv)

    logger.success(f'Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Converts a folder of ChartStructs to ChartJson
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/r0729-ae0728-092124/lgbm-092124',
    )
    args.parse_args(parser)
    main()