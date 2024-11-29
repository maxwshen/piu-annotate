"""
    Make search json data structure, for search bar: list all chart json files
    available in a folder
"""
import argparse
import os
import json
from hackerargs import args
from loguru import logger
from tqdm import tqdm

from piu_annotate.formats.jsplot import ChartJsStruct


def main():
    """ Writes special json files for metadata to chart-json folder.
        Uses special reserved prefix "__"
    """
    chart_json_folder = os.path.join(args['chart_json_folder'])
    logger.info(f'Looking for json files in {chart_json_folder} ...')
    fns = [fn.replace('.json', '')
           for fn in os.listdir(chart_json_folder)
           if fn.endswith('.json') and not fn.startswith('__')]
    logger.info(f'Found {len(fns)} chart jsons')

    out_fn = os.path.join(chart_json_folder, '__search-struct.json')
    with open(out_fn, 'w') as f:
        json.dump(fns, f)
    logger.info(f'Wrote to {out_fn}')

    # make list of manually annotated charts
    manual_annotated = []
    logger.info(f'Finding ChartJSONs with manual limb annotations ...')
    for fn in tqdm(os.listdir(chart_json_folder)):
        if fn.startswith('__'):
            continue
        cjs = ChartJsStruct.from_json(os.path.join(chart_json_folder, fn))
        if cjs.metadata['Manual limb annotation'] is True:
            manual_annotated.append(fn.replace('.json', ''))
    logger.info(f'Found {len(manual_annotated)} ChartJSONs with manual limb annotations')
    
    out_fn = os.path.join(chart_json_folder, '__manual-limb-annotated.json')
    with open(out_fn, 'w') as f:
        json.dump(manual_annotated, f)
    logger.info(f'Wrote to {out_fn}')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Finds all json files in chart-json folder.
            Writes to chart-json folder / search-struct.json
        """
    )
    parser.add_argument(
        '--chart_json_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/r0729-ae0728-092124/lgbm-092124/chart-json/'
    )
    args.parse_args(parser)
    main()