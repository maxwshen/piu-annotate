"""
    Crawls PIU-Simfiles folder
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from piu_annotate.crawl import crawl_sscs
from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct


def get_num_bad_lines(stepchart: StepchartSSC) -> int:
    cs_df, num_bad_lines = stepchart_ssc_to_chartstruct(stepchart)
    return num_bad_lines


def main():
    simfiles_folder = args['simfiles_folder']

    skip_packs = ['INFINITY']
    logger.info(f'Skipping packs: {skip_packs}')

    song_sscs = crawl_sscs(simfiles_folder, skip_packs = skip_packs)

    # get stepcharts
    stepcharts: list[StepchartSSC] = []
    for song in song_sscs:
        stepcharts += song.stepcharts
    logger.info(f'Found {len(stepcharts)} stepcharts')

    standard_stepcharts = [sc for sc in stepcharts if not sc.is_nonstandard()]
    logger.success(f'Found {len(standard_stepcharts)} standard stepcharts')

    # make stepchart df
    stepchart_dd = []
    logger.info(f'Writing description df of stepcharts ...')
    for sc in tqdm(stepcharts):
        d = pd.Series(sc.describe())
        stepchart_dd.append(d)
    stepchart_df = pd.DataFrame(stepchart_dd)
    stepchart_df.to_csv('__stepcharts.csv')
    logger.info(f'Wrote to __stepcharts.csv')

    import multiprocessing as mp
    logger.info(f'Computing num. bad lines in standard stepcharts ...')
    inputs = [[stepchart] for stepchart in standard_stepcharts]
    with mp.Pool(num_processes := 6) as pool:
        mp_num_bad_lines = pool.starmap(
            get_num_bad_lines,
            tqdm(inputs, total = len(inputs))
        )
    from collections import Counter
    logger.info(f'Computing num. bad lines count in standard stepcharts: ')
    logger.info(Counter(mp_num_bad_lines))

    # make standard stepchart df
    stepchart_dd = []
    logger.info(f'Writing description df of standard stepcharts ...')
    for sc in tqdm(standard_stepcharts):
        d = pd.Series(sc.describe())
        stepchart_dd.append(d)
    stepchart_df = pd.DataFrame(stepchart_dd)
    stepchart_df['Num. bad lines'] = mp_num_bad_lines
    stepchart_df.to_csv('__standard_stepcharts.csv')
    logger.info(f'Wrote to __standard_stepcharts.csv')

    problem_dir = '/home/maxwshen/piu-annotate/output/problematic-ssc/'
    n_files_written = 0
    for stepchart, num_bad_lines in zip(standard_stepcharts, mp_num_bad_lines):
        if num_bad_lines > 0:
            n_files_written += 1
            stepchart.to_file(os.path.join(problem_dir, stepchart.shortname() + '.ssc'))
    logger.info(f'Wrote {n_files_written} .ssc to {problem_dir}')

    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Crawls PIU-Simfiles folder for .ssc files -> StepChartSSC objects ->
            ChartStruct objects.
        """
    )
    parser.add_argument(
        '--simfiles_folder', 
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/'
    )
    args.parse_args(parser)
    main()