import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import difflib
import yaml

from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.formats.chart import ChartStruct


def main():
    cs_folder = args['chartstruct_csv_folder']
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]

    json_folder = args['manual_json_folder']
    # json_files = [fn for fn in os.listdir(json_folder) if fn.endswith('.json')]
    # crawl all subdirs for jsons
    json_files = []
    dirpaths = set()
    for dirpath, _, files in os.walk(json_folder):
        for file in files:
            if file.endswith('.json') and 'exclude' not in dirpath:
                json_files.append(os.path.join(dirpath, file))
                dirpaths.add(dirpath)
    logger.info(f'Found {len(json_files)} jsons in {len(dirpaths)} directories ...')

    json_to_candidates = dict()
    json_to_csfn = dict()
    logger.info('Enumerating over manual json files ...')
    for full_json_fn in tqdm(json_files):
        json_basename = os.path.basename(full_json_fn)
        fms = difflib.get_close_matches(json_basename, chartstruct_files)
        cjs = ChartJsStruct.from_json(full_json_fn)

        json_to_candidates[json_basename] = fms

        for match_csv in fms:
            full_cs_fn = os.path.join(cs_folder, match_csv)
            cs = ChartStruct.from_file(full_cs_fn)
            exact_match = False
            try:
                exact_match = cs.matches_chart_json(cjs, with_limb_annot = False)
            except Exception as e:
                logger.error(str(e))
                logger.error(full_cs_fn)
                exact_match = False

            if exact_match:
                json_to_csfn[full_json_fn] = full_cs_fn

    logger.info(f'Found {len(json_to_csfn)} matches out of {len(json_files)}')

    # save dict
    csfn_to_json = {v: k for k, v in json_to_csfn.items()}
    out_fn = os.path.join(cs_folder, '__cs_to_manual_json.yaml')
    with open(out_fn, 'w') as f:
        yaml.dump(csfn_to_json, f)
    logger.success(f'Wrote to {out_fn}')

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Try to match ChartStruct with manually annotated jsons,
            fuzzy matching filenames, then computing compatibility in arrow times
            and positions.
            
            Writes a data structure of matched ChartStruct csvs to their manual json
            to a private YAML file in `chartstruct_csv_folder`.
            
            Use to run predictions on ChartStructs without manually annotated jsons,
            and use manual annotations otherwise.
        """
    )
    parser.add_argument(
        '--chartstruct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/r0729-ae0728-092124/'
    )
    parser.add_argument(
        '--manual_json_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/manual-jsons/'
    )
    args.parse_args(parser)
    main()