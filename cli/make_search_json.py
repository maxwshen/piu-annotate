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
    collection_folder = os.path.join(args['collection_folder'])
    logger.info(f'Looking for json files in {collection_folder} ...')
    fns = [fn.replace('.json', '')
           for fn in os.listdir(os.path.join(collection_folder, 'chart-json/'))
           if fn.endswith('.json')]
    logger.info(f'Found {len(fns)} chart jsons')

    out_fn = os.path.join(collection_folder, 'search-struct.json')
    with open(out_fn, 'w') as f:
        json.dump(fns, f)
    logger.info(f'Wrote to {out_fn}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Finds all json files in {collection_folder}/chart-json/.
            Writes to {collection_folder}/search-struct.json
        """
    )
    parser.add_argument(
        '--collection_folder', 
        default = '/home/maxwshen/piu-annotate/output/piucenter-annot-070824/'
    )
    args.parse_args(parser)
    main()