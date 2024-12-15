import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import itertools
import sys
from collections import Counter
import pandas as pd
import json
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.utils import make_basename_url_safe
from piu_annotate import utils


# sync to skills.py
# order here determines order on /skill/ navigation page
skill_cols_to_description = {
    '__jump': 'Jumps require you to use both feet to hit two or more notes at the same time. This can take a lot of energy, so it is valuable to learn how to conserve energy on jumps. At fast NPS, jumps can alternatively be executed as a run, at double time rhythm.',
    '__drill': 'Drills are a series of notes (here, defined as 5 or more notes) that alternate between just two panels. Compared to runs at the same level, drills can demand higher speeds and length because in principle you can conserve much more energy on drills. Most players find blue-panel drills executed with heels much easier than red-panel drills that require toes, to the extent that some players even position their feet to perform red drills with their heels.',
    '__run': 'Runs, or streams, are a series of notes (here, defined as 7 or more notes) at constant rhythm. Unlike drills which alternate between just two panels, runs involve a variety of panels, which in doubles can require you to move around the pads. In most arcade-length charts, the crux is an ending run, between 5 seconds to 25 seconds long. 8th note runs at 100 BPM to 200 BPM correspond to ~3.3 notes per second (NPS) to ~6.7 NPS and are found at S6~S14. 16th note runs at 100 BPM to 200+ BPM, or ~6.7 NPS to ~13.3+ NPS start to be introduced at S16 and beyond.',
    '__anchor run': 'Anchor runs are an intermediate between drills and runs, where one foot hits the same panel repeatedly (the "anchor"), while the other foot hits a variety of patterns. As anchor runs are less complex than full runs, they are a good training ground and stepping stone for improving at runs.',
    '__run without twists': 'Runs without twists are the most straightforward type of run. For a given difficulty level, these are longer in duration than full runs, which make them a useful stepping stone for improving at runs and improving time under tension.',
    '__twist 90': 'A twist at 90 degrees is the simplest type of twist, and generally involves hitting red and blue panels. Compared to the base stance where your body faces the screen, during a 90 degree twist, you step directly towards the screen, or directly away from the screen, so that your body or hips become perpendicular to the screen.',
    '__twist over90': 'A twist at >90 degrees involves hitting panels at greater than 90 degree angle. Compared to the base stance where your body faces the screen, during a >90 degree twist, you step with one foot fully crossing the other foot, so that your body or hips twist beyond being perpendicular to the screen.',
    '__twist close': 'A close twist is a twist involving panels close to each other. Generally speaking, close twists involve a red/blue panel and the center panel. These twists are much more common than "far" twists.',
    '__twist far': 'A far twist is a twist involving panels that are not close to each other. In singles, a far twist involves the diagonal-opposite red-blue panels. These twists are relatively rare, generally harder to read, and require a higher level of commitment to execute, as they place the body into a more demanding position.',
    '__side3 singles': 'Side3 singles describes chart sections that primarily use the side 3 arrows in singles. Twists in these sections are sometimes called corner twists.',
    '__mid6 doubles': 'Mid6 doubles, also known as half-doubles, describes chart sections that primarily use the middle 6 panels in doubles. Strong familiarity with mid6 double patterns are necessary for advancing in doubles beyond a certain level. Half-double patterns can sometimes be the hardest part of learning doubles after learning singles.',
    '__mid4 doubles': 'Mid4 doubles describes chart sections that primarily use the middle 4 panels in doubles. These patterns are typically a challenging barrier for singles players advancing in doubles.',
    '__doublestep': 'Doublestepping is the act of stepping with the same foot twice in a row, contrasting with the typical flow of alternating feet. Doublestepping is primarily forced in stepcharts by requiring one foot remain on a hold, while the other foot hits multiple arrows in a row.',
    '__jack': 'Jacks are when you execute a pattern of multiple arrows in a row on the same panel, by using one foot to hit all the arrows. To properly execute jacks, it is important to completely lift your foot off the panel each time, which can be a common mistake at higher speeds. Jacks are related to footswitches, where repeated arrows are instead hit with alternating feet.',
    '__footswitch': 'Footswitches are when you execute a pattern of multiple arrows in a row on the same panel, by alternating feet. These are an alternative to jacks, where the correct choice depends on the chart context. Typically, footswitches are the correct answer with irregular rhythm, or when they avoid a difficult twist forced with jacks.',
    '__bracket': 'A bracket is hitting two arrows with one foot at the same time. Brackets require the two arrows to be physically next to each other on the pad.',
    '__staggered bracket': 'A staggered bracket are two arrows close in time, but not at the same time, which are nevertheless executed as a bracket with one foot.',
    '__bracket run': 'These identify brackets that occur in runs.',
    '__bracket drill': 'These identify brackets that occur in drills.',
    '__bracket jump': 'Bracket jumps are jumps that involve brackets.',
    '__bracket twist': 'Bracket twists are lines that executed by bracketing and twisting at the same time.',
    # '__split',
    '__hands': 'Rare patterns can require the use of feet and hands together.',
}
skill_cols = list(skill_cols_to_description.keys())
renamed_skill_cols = {skill_col: skill_col.replace('__', '').replace(' ', '_')
                      for skill_col in skill_cols}


def make_skills_dataframe():
    """ Create dataframe: each row is a stepchart, columns are skill frequencies
    """
    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/skills/stepchart-skills.csv'
    rerun_all = args.setdefault('rerun', False)
    if not rerun_all:
        if os.path.exists(dataset_fn):
            df = pd.read_csv(dataset_fn)
            return df

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    dd = defaultdict(list)

    logger.info('Annotating skills ...')
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        shortname = cs.metadata['shortname']
        sord = cs.singles_or_doubles()

        annotate_skills(cs)

        for skill_col in skill_cols:
            dd[renamed_skill_cols[skill_col]].append(cs.df[skill_col].mean())

        dd['shortname'].append(make_basename_url_safe(shortname))
        dd['sord'].append(sord)
        dd['chart level'].append(cs.get_chart_level())
    
    df = pd.DataFrame(dd)

    df.to_csv(dataset_fn, index = False)
    logger.info(f'Saved dataset to {dataset_fn}')
    return df


def get_chart_dict(df: pd.DataFrame, skill: str) -> dict[str, list[str]]:
    """ Gets list of stepcharts for skill, subset by level and sord.
        Returns dict with keys "S17": list of stepchart shortnames
    """
    sord_to_chartlevels = {
        'singles': (7, 26),
        'doubles': (10, 28),
    }
    top_n = 20

    df_sorted = df.sort_values(by = skill, ascending = False).copy()

    charts_dict = dict()
    for sord in ['singles', 'doubles']:
        lower_level, upper_level = sord_to_chartlevels[sord]
        for level in range(lower_level, upper_level + 1):
            crit = (df_sorted['sord'] == sord) & (df_sorted['chart level'] == level)
            dfs = df_sorted[crit]

            charts = list(dfs[dfs[skill] > 0]['shortname'])[:top_n]

            sordlevel = f'{sord[0].upper()}{level}'
            if len(charts) > 0:
                charts_dict[sordlevel] = charts

    return charts_dict


def main():
    cs_folder = args['chart_struct_csv_folder']

    df = make_skills_dataframe()

    skill_dict = dict()
    for skill in list(renamed_skill_cols.values()):
        skill_dict[skill] = get_chart_dict(df, skill)

    skill_descriptions = {renamed_skill_cols[k]: v for k, v in
                          skill_cols_to_description.items()}

    # reformat dd into tierlists jsons
    output_file = os.path.join(cs_folder, 'page-content', 'stepchart-skills.json')
    utils.make_dir(output_file)

    with open(output_file, 'w') as f:
        json.dump((skill_dict, skill_descriptions), f)

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Create content for skills page
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
    )
    args.parse_args(parser)
    main()