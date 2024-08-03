"""
    Check coverage of charts list
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
import pandas as pd
from collections import defaultdict, Counter

from piu_annotate.formats.arroweclipse import ArrowEclipseStepchartListJson, ArrowEclipseChartInfo
from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.crawl import crawl_stepcharts


def matches(ae_ci: ArrowEclipseChartInfo, stepcharts: list[StepchartSSC]) -> bool:
    return any(ae_ci.matches_stepchart_ssc(sc) for sc in stepcharts)


def get_num_matches(
    ae_ci: ArrowEclipseChartInfo, 
    song_name_to_stepcharts: dict[str, StepchartSSC]
) -> int | str:
    song_name = ae_ci.data['song']['name'].lower()
    if song_name not in song_name_to_stepcharts:
        return f'failed to match song name'
        # return f'failed to match song name {song_name}'
    stepcharts = song_name_to_stepcharts[song_name]
    return len([sc for sc in stepcharts if ae_ci.matches_stepchart_ssc(sc)])


def main():
    ae_json = args['charts_list_json']
    simfiles_folder = args['simfiles_folder']

    charts_lj = ArrowEclipseStepchartListJson(ae_json)
    logger.info(f'Loaded {len(charts_lj)} stepcharts from {ae_json}')

    skip_packs = ['INFINITY']
    logger.info(f'Skipping packs: {skip_packs}')
    stepcharts = crawl_stepcharts(simfiles_folder, skip_packs = skip_packs)

    # filter UCS and hidden
    stepcharts = [sc for sc in stepcharts if not sc.is_ucs() and not sc.is_hidden()]

    song_name_to_stepcharts = defaultdict(list)
    for sc in stepcharts:
        song_name_to_stepcharts[sc['TITLE'].lower()].append(sc)

    # try to match ArrowEclipse chart list to stepchart sscs
    n_matches = defaultdict(int)
    for ae_ci in tqdm(charts_lj.charts):
        match_message = get_num_matches(ae_ci, song_name_to_stepcharts)
        n_matches[match_message] += 1
    for k, v in n_matches.items():
        print(f'{k}: {v}')

    # get stepchart sscs that match
    matched_sscs = []
    for ae_ci in tqdm(charts_lj.charts):
        song_name = ae_ci.data['song']['name'].lower()
        if song_name in song_name_to_stepcharts:
            stepcharts = song_name_to_stepcharts[song_name]
            matches = [sc for sc in stepcharts if ae_ci.matches_stepchart_ssc(sc)]
            if len(matches) == 1:
                matched_sscs += matches
    matched_sscs: list[StepchartSSC] = list(set(matched_sscs))
    logger.info(f'Found {len(matched_sscs)} StepchartSSCs matching charts list')

    standard_stepcharts = [sc for sc in matched_sscs if not sc.is_nonstandard()]
    logger.success(f'Found {len(standard_stepcharts)} standard stepcharts')

    # get reasons for nonstandard charts
    nonstandard_reasons = [sc.is_nonstandard_reason() for sc in matched_sscs if sc.is_nonstandard()]
    for k, v in Counter(nonstandard_reasons).items():
        print(f'{k}: {v}')

    # make stepchart df
    stepchart_dd = []
    logger.info(f'Writing description df of stepcharts ...')
    for sc in tqdm(matched_sscs):
        d = pd.Series(sc.describe())
        stepchart_dd.append(d)
    stepchart_df = pd.DataFrame(stepchart_dd)
    stepchart_df.to_csv('__072924-phoenix-stepcharts.csv')
    logger.info(f'Wrote to __072924-phoenix-stepcharts.csv')

    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Check how many charts in `charts_list_json` are covered by
            .ssc in `simfiles_folder`.
        """
    )
    parser.add_argument(
        '--charts_list_json', 
        default = '/home/maxwshen/piu-annotate/data/accessible-stepcharts/072824_phoenix_stepcharts_arroweclipse.json'
    )
    parser.add_argument(
        '--simfiles_folder', 
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-072924/'
    )
    args.parse_args(parser)
    main()