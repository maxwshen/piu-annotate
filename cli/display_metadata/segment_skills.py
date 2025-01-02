import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.segment import Section
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.utils import make_basename_url_safe
from cli.page_content.skills_page import make_skills_dataframe, skill_cols, renamed_skill_cols
from cli.page_content.chart_table import SkillComparer


def annotate_segment_similarity():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    chart_skill_df = make_skills_dataframe()
    skill_comparer = SkillComparer(chart_skill_df)

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
            'Ugly_Dee_-_Banya_Production_D18_ARCADE.csv',
            # 'Can-can_Orpheus_in_The_Party_Mix_-_SHORT_CUT_-_-_Sr._Lan_Belmont_D25_SHORTCUT.csv',
            # 'GLORIA_-_Croire_D21_ARCADE.csv',
            # 'PaPa_Gonzales_-_BanYa_S22_ARCADE.csv',
            # 'Final_Audition_2__-_SHORT_CUT_-_-_Banya_S17_SHORTCUT.csv',
        ]
        chartstruct_files = [os.path.join(cs_folder, f) for f in chartstruct_files]

    eligible_skill_cols = None

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        shortname = cs.metadata['shortname']
        chart_level = cs.get_chart_level()
        sord = cs.singles_or_doubles()

        # should not need to be rerun after skills_page.py
        # annotate_skills(cs)

        lower_level = min(chart_level - 1, 24)
        upper_level = chart_level

        assert 'Segment metadata' in cs.metadata, 'Expected segment metadata dicts to already be created'
        meta_dicts = cs.metadata['Segment metadata']
        # one dict per section

        for section_idx, section in enumerate(sections):
            # featurize the section
            start, end = section.start, section.end
            dfs = cs.df.iloc[start:end]

            # annotate skills already called to create skills_df
            if eligible_skill_cols is None:
                # find skills in df - ignores chart-wide skills like bursty/sustained
                eligible_skill_cols = [col for col in skill_cols if col in dfs.columns]
                
                # disallow `run without twists`
                disallowed = ['__run without twists']
                for col in disallowed:
                    eligible_skill_cols.remove(col)

            skill_fq_dfs = dfs[eligible_skill_cols].mean(axis = 0)

            # get pct
            # compare to skills_df
            skill_to_pcts = dict()
            for skill in eligible_skill_cols:
                renamed_skill = renamed_skill_cols[skill]

                ref_vals = skill_comparer.get_values(
                    renamed_skill, sord, lower_level, upper_level
                )
                query_val = skill_fq_dfs[skill]
                if query_val > 0:
                    percentile = sum(query_val > ref_vals) / len(ref_vals)
                    skill_to_pcts[renamed_skill] = percentile

            # find highest percentile skills
            sorted_skills = sorted(skill_to_pcts, key = skill_to_pcts.get, reverse = True)
            skill_badges = sorted_skills[:3]

            meta_dicts[section_idx]['Skill badges'] = skill_badges

        cs.metadata['Segment metadata'] = meta_dicts
        if debug:
            import code; code.interact(local=dict(globals(), **locals()))
        cs.to_csv(inp_fn)

    return


def main():
    annotate_segment_similarity()

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Update ChartStruct metadata Segment metadata with skill badges.

        Requires as input:
            skill dataframe, from skills_page.py
            segments, from segment_charts.py
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/main/lgbm-120524/',
    )
    args.parse_args(parser)
    main()