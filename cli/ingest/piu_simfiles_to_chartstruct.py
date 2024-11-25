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
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.crawl import crawl_stepcharts
from piu_annotate.utils import make_dir


def ssc_to_cs(stepchart: StepchartSSC, out_folder: str) -> bool:
    """ Attempts to convert StepChartSSC -> ChartStruct, then write to json.
        Returns success status.
    """
    chart_struct: ChartStruct = ChartStruct.from_stepchart_ssc(stepchart)
    try:
        chart_struct.validate()
    except:
        return False
    chart_struct.to_csv(os.path.join(out_folder, stepchart.shortname() + '.csv'))
    return True


def main():
    simfiles_folder = args['simfiles_folder']
    out_folder = args['output_folder']
    make_dir(out_folder)

    skip_packs = ['INFINITY']
    logger.info(f'Skipping packs: {skip_packs}')
    stepcharts = crawl_stepcharts(simfiles_folder, skip_packs = skip_packs)

    standard_stepcharts = [sc for sc in stepcharts if not sc.is_nonstandard()]
    logger.success(f'Found {len(standard_stepcharts)} standard stepcharts')

    import multiprocessing as mp
    inputs = [[stepchart] for stepchart in standard_stepcharts]
    with mp.Pool(num_processes := 6) as pool:
        results = pool.starmap(
            ssc_to_cs,
            tqdm(inputs, total = len(inputs))
        )

    success_counts = Counter(results)
    logger.info(f'{success_counts=}')

    failed_stepcharts = [stepchart for stepchart, res in zip(standard_stepcharts, results)
                         if res is False]
    logger.info(f'{failed_stepcharts=}')
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
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/'
    )
    parser.add_argument(
        '--output_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/rayden-072924/'
    )
    args.parse_args(parser)
    main()