import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys
import pandas as pd
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.difficulty import featurizers


def find_longest_true_run(values: list[bool]) -> tuple[int, int]:
    """
    Find the start and end indices of the longest consecutive run of True values.
    
    Args:
        values (list[bool]): List of boolean values
        
    Returns:
        tuple[int, int]: (start_index, end_index) of the longest run.
                        If no True values exist, returns (-1, -1).
                        The end_index is inclusive.
    """
    if len(values) == 0:
        return (-1, -1)
    
    max_length = 0
    max_start = -1
    max_end = -1
    
    current_start = 0
    current_length = 0
    
    for i, value in enumerate(values):
        if value:
            # Extend current run
            if current_length == 0:
                current_start = i
            current_length += 1
            
            # Update max if current run is longer
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
                max_end = i
        else:
            # Reset current run
            current_length = 0
    
    return (max_start, max_end)


def main():
    # run on folder
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    dd = defaultdict(list)
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        annotate_skills(cs)

        runs = np.array(cs.df['__run'])
        not_staggered_bracket = np.logical_not(np.array(cs.df['__staggered bracket']))
        bool_list = np.logical_and(runs, not_staggered_bracket)

        start, end = find_longest_true_run(bool_list)
        times = list(cs.df['Time'])
        run_time_length = times[end] - times[start]
        nps = (end - start + 1) / run_time_length if run_time_length > 0 else 0

        dd['shortname'].append(cs.metadata['shortname'])
        dd['nps'].append(nps)
        dd['time length'].append(run_time_length)
        dd['level'].append(int(cs.metadata['METER']))
        dd['sord'].append(cs.singles_or_doubles())

    df = pd.DataFrame(dd)
    df.to_csv('/home/maxwshen/piu-annotate/artifacts/analysis/runs.csv')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/092424/lgbm-110424/',
    )
    args.parse_args(parser)
    main()