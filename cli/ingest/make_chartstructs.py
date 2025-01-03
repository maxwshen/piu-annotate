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
import difflib
import yaml

from piu_annotate.formats.arroweclipse import ArrowEclipseStepchartListJson, ArrowEclipseChartInfo
from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.crawl import crawl_stepcharts
from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
from piu_annotate.utils import make_dir, make_basename_url_safe


def allowed_multimatch(ae_ci: ArrowEclipseChartInfo) -> bool:
    multimatches = {
        'Baroque Virus - FULL SONG -': {
            'level': 23,
            'type': 'Double',
        },
        'Gargoyle - FULL SONG -': {
            'level': 21,
            'type': 'Single',
        }
    }
    def match_dict(query: dict, target: dict):
        return all(query[k] == target[k] for k in query)

    if ae_ci.data['song']['name'] not in multimatches:
        return False
    
    for song_name, mm in multimatches.items():
        if match_dict(mm, ae_ci.data):
            return True
    return False


def fuzzy_match_song_name(query: str, targets: list[str]) -> str | None:
    close_matches = difflib.get_close_matches(query, targets)
    if len(close_matches) > 0:
        return close_matches[0]
    else:
        return None


def match_aeci_to_ssc(
    ae_ci: ArrowEclipseChartInfo, 
    song_name_to_stepcharts: dict[str, StepchartSSC]
) -> tuple[list[StepchartSSC], str]:
    """ Attempts to match an ArrowEclipseChartInfo (with level) to StepChartSSC.

        Supports manually-approved multiple matches, such as:
        - Baroque Virus Full Song D23
        - Gargoyle Full Song S21
        In this case, we add 'v1' / 'v2' to the TITLE.
        
        Output
        ------
        matched_stepcharts: list[StepchartSSC]
        message: str
    """
    song_name = ae_ci.data['song']['name'].lower()
    target_songs = list(song_name_to_stepcharts.keys())
    if song_name in target_songs:
        matched_song_name = song_name
    else:
        fuzzy_matched_song = fuzzy_match_song_name(song_name, target_songs)
        if fuzzy_matched_song is None:
            return [], f'failed to fuzzy match song name'
        matched_song_name = fuzzy_matched_song

    stepcharts = song_name_to_stepcharts[matched_song_name]
    matched_ssc = [sc for sc in stepcharts if ae_ci.matches_stepchart_ssc_partial(sc)]

    if len(matched_ssc) > 1:
        if not allowed_multimatch(ae_ci):
            return [], f'Unexpectedly matched {len(matched_ssc)} stepcharts in .ssc'
        else:
            for i, mssc in enumerate(matched_ssc):
                mssc['TITLE'] += f' v{i+1}'

    return matched_ssc, 'ok'


def try_ssc_to_chartstruct(stepchart: StepchartSSC, out_folder: str) -> str:
    basename = make_basename_url_safe(stepchart.shortname() + '.csv')
    out_file = os.path.join(out_folder, basename)

    if os.path.isfile(out_file):
        return 'skipped because exists'
    # print(basename)

    cs_df, holdticks, cs_message = stepchart_ssc_to_chartstruct(stepchart)
    if cs_message == 'success':
        cs = ChartStruct.from_stepchart_ssc(stepchart)
        cs.to_csv(out_file)
    return cs_message


def main():
    ae_json = args['charts_list_json']
    simfiles_folder = args['simfiles_folder']

    """
        Load phoenix-accessible stepcharts
    """
    charts_ae = ArrowEclipseStepchartListJson(ae_json).charts

    # Filter co-op and too low level stepcharts
    def filt_ae(ae_ci: ArrowEclipseChartInfo) -> bool:
        return any([
            ae_ci.is_coop(),
            # (ae_ci.is_singles() and ae_ci.level() < 6),
            # (ae_ci.is_doubles() and ae_ci.level() < 10),
        ])
    charts_ae = [c for c in charts_ae if not filt_ae(c)]
    logger.info(f'Loaded {len(charts_ae)} filtered stepcharts from {ae_json}')

    """
        Load .ssc
    """
    skip_packs = ['INFINITY', 'MOBILE EDITION']
    logger.info(f'Skipping packs: {skip_packs}')
    stepcharts = crawl_stepcharts(simfiles_folder, skip_packs = skip_packs)

    # filter stepchartssc
    def filt(sc: StepchartSSC) -> bool:
        return any([
            sc.is_ucs(),
            sc.is_hidden(),
            sc.is_quest(),
            sc.is_infinity(),
            sc.is_train(),
            sc.is_pro(),
            sc.is_performance(),
            sc.is_jump_edition(),
        ])
    stepcharts = [sc for sc in stepcharts if not filt(sc)]

    song_name_to_stepcharts = defaultdict(list)
    for sc in stepcharts:
        song_name_to_stepcharts[sc['TITLE'].lower()].append(sc)

    """
        1. try to match ArrowEclipse chart list to stepchart sscs
    """
    matched_sscs = []
    n_matches = defaultdict(int)
    msg_to_charts = defaultdict(list)
    for ae_ci in tqdm(charts_ae):
        matches, message = match_aeci_to_ssc(ae_ci, song_name_to_stepcharts)
        n_matches[message] += 1
        if message == 'ok':
            matched_sscs += matches
        else:
            msg_to_charts[message].append(ae_ci.shortname())
    for k, v in n_matches.items():
        logger.info(f'{k}: {v}')
    matched_sscs: list[StepchartSSC] = list(set(matched_sscs))
    logger.info(f'Found {len(matched_sscs)} StepchartSSCs matching charts list')

    # save log
    log_file = '/home/maxwshen/piu-annotate/output/logs/match-ae-to-ssc.yaml'
    with open(log_file, 'w') as f:
        yaml.dump(dict(msg_to_charts), f)
    logger.info(f'Saved log to {log_file}')

    logger.debug('Results from matching AE list to .ssc files:')
    for k, v in msg_to_charts.items():
        logger.debug(f'{k}: {v}')

    """
        2. Matched .ssc: perform checks if meets standard
    """
    logger.info(f'Checking ssc ... ')
    standard_stepcharts = [sc for sc in tqdm(matched_sscs) if not sc.is_nonstandard()]
    logger.success(f'Found {len(standard_stepcharts)} standard stepcharts')

    # get reasons for nonstandard charts
    nonstandard_reasons = [sc.is_nonstandard_reason() for sc in matched_sscs
                           if sc.is_nonstandard()]
    for k, v in Counter(nonstandard_reasons).items():
        logger.info(f'{k}: {v}')

    """
        3. Convert standard stepchartssc into chartstruct
    """
    # convert standard .ssc into chartstruct
    out_chartstruct_folder = args['output_chartstruct_folder']
    make_dir(out_chartstruct_folder)
    import multiprocessing as mp
    logger.info(f'Converting standard .ssc to chartstruct ...')
    inputs = [[stepchart, out_chartstruct_folder] for stepchart in standard_stepcharts]

    with mp.Pool(num_processes := 6) as pool:
        mp_cs_msg = pool.starmap(
            try_ssc_to_chartstruct,
            tqdm(inputs, total = len(inputs))
        )
    for k, v in Counter(mp_cs_msg).items():
        logger.info(f'{k}: {v}')

    # mp_cs_msg = []
    # for input in tqdm(inputs):
    #     try_ssc_to_chartstruct(*input)

    # build message to stepchartssc mapping
    cs_dd = defaultdict(list)
    for msg, stepchart in zip(mp_cs_msg, standard_stepcharts):
        if msg != 'success':
            cs_dd[msg].append(stepchart)

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Use `charts_list_json` (officially accessible
            stepcharts) to select .ssc files in `simfiles_folder`,
            then convert .ssc to chartstructs.
        """
    )
    parser.add_argument(
        '--charts_list_json', 
        default = '/home/maxwshen/piu-annotate/artifacts/accessible-stepcharts/122824-arroweclipse.json'
    )
    parser.add_argument(
        '--simfiles_folder', 
        default = '/home/maxwshen/PIU-Simfiles-rayden-61-122824/'
    )
    parser.add_argument(
        '--output_chartstruct_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/'
    )
    args.parse_args(parser)
    main()