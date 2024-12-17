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

from cli.page_content.skills_page import skill_cols, renamed_skill_cols, make_skills_dataframe

chart_skill_df = None

@functools.lru_cache
def get_percentile(
    skill_col: str, 
    sord: str,
    lower_level: int,
    upper_level: int,
    pct: int
) -> float:
    crit = (chart_skill_df['sord'] == sord) & (chart_skill_df['chart level'] >= lower_level) \
        & (chart_skill_df['chart level'] <= upper_level)
    return np.percentile(np.array(chart_skill_df[crit][skill_col]), pct)


def get_top_chart_skills(
    chart_skill_df: pd.DataFrame, 
    name: str, 
    sord: str,
    level: int,
    pct: int = 80
) -> list[str]:
    badges = []
    for skill in list(renamed_skill_cols.values()):
        percentile = get_percentile(skill, sord, min(level - 1, 24), level, pct)
        val = chart_skill_df.loc[chart_skill_df['shortname'] == name, skill].iloc[0]
        if val > percentile:
            badges.append(skill)

    return badges


def get_chart_badges() -> dict[str, list[str]]:
    """ Returns dict with keys "shortname": list of badges
    """
    global chart_skill_df
    chart_skill_df = make_skills_dataframe()

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    if args.setdefault('debug', False):
        chartstruct_files = [
            # 'Clematis_Rapsodia_-_Jehezukiel_D23_ARCADE.csv',
            'After_LIKE_-_IVE_D20_ARCADE.csv',
        ]

    chart_badges = dict()
    for cs_file in tqdm(chartstruct_files):
        badges = []
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        shortname = cs.metadata['shortname']
        sord = cs.singles_or_doubles()
        level = cs.get_chart_level()

        # add badges for skills that stepchart is enriched in
        badges += get_top_chart_skills(chart_skill_df, shortname, sord, level)

        # add badges for skill warnings in segments
        for section, section_dict in cs.metadata['Segment metadata'].items():
            badges += section_dict['rare skills']

        # add other badges

        enps_data = cs.metadata['eNPS timeline data']
        # sustained vs bursty

        # time under tension

        # nps

    return chart_badges


def main():
    cs_folder = args['chart_struct_csv_folder']

    chart_badges = get_chart_badges()

    output_file = os.path.join(cs_folder, 'page-content', 'chart-badges.json')
    utils.make_dir(output_file)
    with open(output_file, 'w') as f:
        json.dump(chart_badges, f)
    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Create content for chart badges
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
    )
    args.parse_args(parser)
    main()