import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import itertools
import functools
import sys
from collections import Counter
import pandas as pd
import json
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.utils import make_basename_url_safe
from piu_annotate import utils
from piu_annotate.formats.nps import calc_bpm

from cli.page_content.skills_page import skill_cols, renamed_skill_cols, make_skills_dataframe
from cli.page_content.tierlists import get_notetype_and_bpm_info

chart_skill_df = None

@functools.lru_cache
def get_subset_df(
    skill_col: str, 
    sord: str,
    lower_level: int,
    upper_level: int,
) -> float:
    crit = (chart_skill_df['sord'] == sord) & (chart_skill_df['chart level'] >= lower_level) \
        & (chart_skill_df['chart level'] <= upper_level)
    return np.array(chart_skill_df[crit][skill_col])


def get_top_chart_skills(
    chart_skill_df: pd.DataFrame, 
    cs: ChartStruct,
) -> list[str]:
    """ Gets the most distinguishing skills for stepchart `cs`, compared to
        all stepchart skill statistics.
    """
    name = make_basename_url_safe(cs.metadata['shortname'])
    sord = cs.singles_or_doubles()
    level = cs.get_chart_level()
    skill_to_pcts = {}
    for skill in list(renamed_skill_cols.values()):
        data = get_subset_df(skill, sord, min(level - 1, 24), level)
        val = chart_skill_df.loc[chart_skill_df['shortname'] == name, skill].iloc[0]
        percentile = sum(val > data) / len(data)
        skill_to_pcts[skill] = percentile

    # aggregate and reduce some skills
    to_group = {
        'twists': ['twist_close', 'twist_over90', 'twist_90', 'twist_far']
    }
    for k, v in to_group.items():
        skill_to_pcts[k] = max(skill_to_pcts[c] for c in v)
        for c in v:
            del skill_to_pcts[c]

    # find highest percentile skills
    sorted_skills = sorted(skill_to_pcts, key = skill_to_pcts.get, reverse = True)
    return sorted_skills[:3]


def get_chart_badges() -> dict[str, dict[str, any]]:
    """ Returns dict with keys "shortname": dict[col_name, any]
        for keys like:
        - pack
        - sord
        - level
        - skill badge summary
        - eNPS
        - run length (time under tension)
    """
    global chart_skill_df
    chart_skill_df = make_skills_dataframe()

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    if args.setdefault('debug', False):
        chartstruct_files = [
            'Good_Night_-_Dreamcatcher_D22_ARCADE.csv',
            'Clematis_Rapsodia_-_Jehezukiel_D23_ARCADE.csv',
            'After_LIKE_-_IVE_D20_ARCADE.csv',
        ]

    all_chart_dicts = []
    for cs_file in tqdm(chartstruct_files):
        chart_dict = dict()
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        shortname = make_basename_url_safe(cs.metadata['shortname'])
        sord = cs.singles_or_doubles()
        level = cs.get_chart_level()

        # metadata
        chart_dict['name'] = shortname
        chart_dict['sord'] = sord
        chart_dict['level'] = level
        chart_dict['pack'] = cs.metadata.get('pack', '')
        # chart_dict['songtype'] = cs.metadata.get('SONGTYPE', '')
        # chart_dict['songcategory'] = cs.metadata.get('SONGCATEGORY', '')
        # chart_dict['displaybpm'] = cs.metadata.get('DISPLAYBPM', '')

        # add badges for skills that stepchart is enriched in
        top_chart_skills = get_top_chart_skills(chart_skill_df, cs)
        chart_dict['skills'] = top_chart_skills

        # add badges for skill warnings in segments
        # for section, section_dict in cs.metadata['Segment metadata'].items():
        #     badges += section_dict['rare skills']

        # add other badges
        nps = np.percentile(cs.metadata['eNPS timeline data'], 95)

        # calc bpm
        display_bpm = cs.metadata.get('DISPLAYBPM', None)
        if display_bpm == '':
            display_bpm = None

        if display_bpm is not None and ':' not in display_bpm and float(display_bpm) > 0:
            # use display bpm if available
            bpm, notetype = calc_bpm(1 / nps, float(display_bpm))
            notetype_bpm_info = f'{notetype} @ {round(bpm)} bpm'
        else:
            notetype_bpm_info = get_notetype_and_bpm_info(nps)

        chart_dict['NPS'] = np.round(nps, decimals = 1)
        chart_dict['BPM info'] = notetype_bpm_info

        # time under tension
        range_len = lambda r: r[1] - r[0] + 1
        roi = cs.metadata['eNPS ranges of interest']
        run_length = 0
        if len(roi) > 0:
            run_length = max(range_len(r) for r in roi)
        chart_dict['Sustain time'] = run_length

        # nps
        all_chart_dicts.append(chart_dict)

        # also update metadata
        # todo - refactor this to occur elsewhere
        cs.metadata['chart_skill_summary'] = top_chart_skills
        cs.metadata['nps_summary'] = np.round(nps, decimals = 1)
        cs.metadata['notetype_bpm_summary'] = notetype_bpm_info

        cs.to_csv(os.path.join(cs_folder, cs_file))

    return all_chart_dicts


def main():
    cs_folder = args['chart_struct_csv_folder']

    chart_badges = get_chart_badges()

    output_file = os.path.join(cs_folder, 'page-content', 'chart-table.json')
    utils.make_dir(output_file)
    with open(output_file, 'w') as f:
        json.dump(chart_badges, f)
    logger.info(f'Wrote to {output_file}')
    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Create chart summary information, for table
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
    )
    args.parse_args(parser)
    main()