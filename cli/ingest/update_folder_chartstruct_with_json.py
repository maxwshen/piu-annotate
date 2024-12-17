import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import difflib

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate import utils


def main():
    chart_json_folder = args['chart_json_folder']
    output_folder = args['output_folder']
    utils.make_dir(output_folder)
    logger.info(f'Saving to {output_folder}')

    cs_folder = args['chart_struct_csv_folder']
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]

    chart_jsons = [fn for fn in os.listdir(chart_json_folder) if fn.endswith('.json')]
    print(len(chart_jsons))
    logger.info(f'Found {len(chart_jsons)} in {chart_json_folder=}')

    rerun = args.setdefault('rerun', False)

    failures = []
    success = 0
    for chart_json_basename in tqdm(chart_jsons):
        chart_json_file = os.path.join(chart_json_folder, chart_json_basename)
        chart_csv_file = chart_json_basename.replace('.json', '.csv')
        out_fn = os.path.join(output_folder, chart_csv_file)
        if os.path.isfile(out_fn) and not rerun:
            if os.path.getmtime(out_fn) > os.path.getmtime(chart_json_file):
                continue

        logger.info(chart_json_basename)
        chart_json = ChartJsStruct.from_json(chart_json_file)

        chart_struct_csv = os.path.join(cs_folder, chart_csv_file)
        if not os.path.isfile(chart_struct_csv):
            matches = difflib.get_close_matches(chart_json_basename, chartstruct_files)
            if len(matches) == 0:
                logger.warning(f'Did not find any matching CSVs ...')
                continue
            chart_struct_csv = matches[0]
            logger.warning(f'CSV not found; using {chart_struct_csv} instead ...')
            chart_struct_csv = os.path.join(cs_folder, chart_struct_csv)

        cs = ChartStruct.from_file(chart_struct_csv)
        cs.update_from_manual_json(chart_json)
        if not cs.matches_chart_json(chart_json):
            logger.error(f'Failed: {chart_json_basename}')
            failures.append(chart_json_basename)
            continue
        
        cs.to_csv(out_fn)
        success += 1

    if failures:
        for basename in failures:
            logger.warning(f'Failed: {basename}')

    logger.info(f'Found {success} successes')
    logger.info(f'Found {len(failures)} failures')

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Uses a folder of manual jsons to update ChartStructs
    """)
    parser.add_argument(
        '--chart_json_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/manual-jsons/manual-annot-101824/',
    )
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424',
    )
    parser.add_argument(
        '--output_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/manual-chartstructs/101824/',
    )
    args.parse_args(parser)
    main()