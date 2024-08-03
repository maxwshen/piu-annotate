"""
    Crawls PIU-Simfiles folder
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import pandas as pd
from collections import defaultdict, Counter


from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.crawl import crawl_stepcharts

OUT_DIR = '/home/maxwshen/piu-annotate/output/rayden-072624/chart-json/'


def stepchart_to_json(stepchart: StepchartSSC) -> bool:
    """ Attempts to convert StepChartSSC -> ChartStruct, then write to json.
        Returns success status.
    """
    chart_struct: ChartStruct = ChartStruct.from_stepchart_ssc(stepchart)
    try:
        chart_struct.validate()
    except:
        return False
    chartjs: ChartJsStruct = ChartJsStruct.from_chartstruct(chart_struct)
    chartjs.to_json(os.path.join(OUT_DIR, stepchart.shortname() + '.json'))
    return True


def main():
    simfiles_folder = args['simfiles_folder']

    skip_packs = ['INFINITY']
    logger.info(f'Skipping packs: {skip_packs}')
    stepcharts = crawl_stepcharts(simfiles_folder, skip_packs = skip_packs)

    standard_stepcharts = [sc for sc in stepcharts if not sc.is_nonstandard()]
    logger.success(f'Found {len(standard_stepcharts)} standard stepcharts')

    import multiprocessing as mp
    inputs = [[stepchart] for stepchart in standard_stepcharts]
    with mp.Pool(num_processes := 6) as pool:
        results = pool.starmap(
            stepchart_to_json,
            tqdm(inputs, total = len(inputs))
        )

    success_counts = Counter(results)
    logger.info(f'{success_counts=}')

    failed_stepcharts = [stepchart for stepchart, res in zip(standard_stepcharts, results)
                         if res is False]
    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Crawls PIU-Simfiles folder
        """
    )
    parser.add_argument(
        '--simfiles_folder', 
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-072124/'
    )
    args.parse_args(parser)
    main()