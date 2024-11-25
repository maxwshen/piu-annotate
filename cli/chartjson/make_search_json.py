"""
    Make search json data structure, for search bar: list all chart json files
    available in a folder
"""
import argparse
import os
import json
from hackerargs import args
from loguru import logger


def main():
    chart_json_folder = os.path.join(args['chart_json_folder'])
    logger.info(f'Looking for json files in {chart_json_folder} ...')
    fns = [fn.replace('.json', '')
           for fn in os.listdir(chart_json_folder)
           if fn.endswith('.json') and fn != 'search-struct.json']
    logger.info(f'Found {len(fns)} chart jsons')

    out_fn = os.path.join(chart_json_folder, 'search-struct.json')
    with open(out_fn, 'w') as f:
        json.dump(fns, f)
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