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


class SkillComparer:
    def __init__(self, chart_skill_df: pd.DataFrame):
        """ Stores `chart_skill_df`, and slices of it """
        self.df = chart_skill_df
        self.shortnames = list(self.df['shortname'])
        self.store = dict()    

    def get_values(self, skill_col: str, sord: str, lower_level: int, upper_level: int):
        """ Subsets chart_skill_df to charts between `lower_level` and `upper_level`,
            with `sord`, and returns skill_col values.
            Can be used to compute the percentile of a query skill mean frequency
            compared to other stepcharts.
        """
        key = (skill_col, sord, lower_level, upper_level)
        if key in self.store:
            return self.store[key]
        
        values = self.df.query(
            "sord == @sord and `chart level`.between(@lower_level, @upper_level)"
        )[skill_col].to_numpy()
        self.store[key] = values
        return values
    
    def get_value_for_chart(self, name: str, skill: str):
        return self.df.at[self.shortnames.index(name), skill]


def get_top_chart_skills(
    chart_skill_df: pd.DataFrame, 
    skill_comparer: SkillComparer,
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
        data = skill_comparer.get_values(skill, sord, min(level - 1, 24), level)
        val = skill_comparer.get_value_for_chart(name, skill)
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
    chart_skill_df = make_skills_dataframe()
    skill_comparer = SkillComparer(chart_skill_df)

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
            'Clematis_Rapsodia_-_Jehezukiel_D23_ARCADE.csv',
            'PaPa_Gonzales_-_BanYa_S22_ARCADE.csv',
            'Can-can_Orpheus_in_The_Party_Mix_-_SHORT_CUT_-_-_Sr._Lan_Belmont_D25_SHORTCUT.csv',
            # 'Good_Night_-_Dreamcatcher_D22_ARCADE.csv',
            # 'After_LIKE_-_IVE_D20_ARCADE.csv',
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
        top_chart_skills = get_top_chart_skills(chart_skill_df, skill_comparer, cs)
        chart_dict['skills'] = top_chart_skills
        if debug:
            logger.debug('Debugging top chart skills ...')
            import code; code.interact(local=dict(globals(), **locals()))

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
            if bpm > float(display_bpm) * 1.05:
                notetype_bpm_info = f'>{notetype} @ {round(float(display_bpm))} bpm'
            elif bpm < float(display_bpm) * 0.95:
                notetype_bpm_info = f'<{notetype} @ {round(float(display_bpm))} bpm'
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
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/main/lgbm-120524/',
    )
    args.parse_args(parser)
    main()