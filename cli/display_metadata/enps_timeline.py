import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import sys
from collections import Counter
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import nps
from piu_annotate.segment.segment import Section
from piu_annotate.utils import make_basename_url_safe


def get_enps_list(cs: ChartStruct) -> list[float]:
    """ Returns a list of eNPS values; idx i = second i.
    """
    max_time = round(max(cs.df['Time']) + 1)

    edp_times = nps.calc_effective_downpress_times(cs)
    edp_times = np.array(edp_times)
    
    time_since_edp = edp_times[1:] - edp_times[:-1]
    time_since_edp = np.insert(time_since_edp, 0, 3)
    # make shape match edp_times

    ts = list(edp_times)
    tdp = list(time_since_edp)

    second_to_num_downpresses = dict(Counter(np.floor(edp_times).astype(int)))
    
    second_to_enps = dict()
    for t in range(0, max_time - 1):
        num_popped = 0
        tdps = []
        while ts and ts[0] < t + 1:
            ts.pop(0)
            tdps.append(tdp.pop(0))
            num_popped += 1

        # filter very small time since
        filt_tdps = [t for t in tdps if t > 0.01]
        if filt_tdps:
            mean_enps_from_time_since = 1 / np.mean(filt_tdps)
            n_dps = second_to_num_downpresses.get(t, 0)

            # it is possible that num actual downpresses differs substantially from
            # mean enps calculated from time_since, for instance if only two arrows occur
            # but at 16th note speed.
            # if there is a substantial difference, defer to actual num downpresses
            if n_dps <= mean_enps_from_time_since * 0.7:
                second_to_enps[t] = np.round(n_dps, 2)
            else:
                second_to_enps[t] = np.round(mean_enps_from_time_since, 2)

    # convert to list
    enps_vector = np.zeros(max_time)
    to_np = lambda x: np.array(list(x))
    enps_vector[to_np(second_to_enps.keys())] = to_np(second_to_enps.values())

    return list(enps_vector)


def find_runs(enps_list: list[float]):
    """ Finds 'runs', or long-ish sections with sustained high eNPS,
        to highlight in enps timeline plot.
        These runs can differ from segments, and are useful to label explicitly.
    """
    # enps thresholds for defining bins
    bin_thresholds = [0, 1.5, 4, 8, 13]

    # required number of seconds in stepchart to accept bin as max eNPS bin
    MIN_SECONDS_IN_BIN = 5

    # min seconds length to accept a run
    MIN_RUN_LENGTH = 3

    # max seconds allowed at 1 bin lower than max bin
    MAX_BREAK = 1

    # find max bin
    enps_array = np.array(enps_list)
    bin_counts = {i: sum(enps_array > threshold) for i, threshold in enumerate(bin_thresholds)}
    max_bin = max([i for i, count in bin_counts.items() if count >= MIN_SECONDS_IN_BIN])
    enps_threshold = bin_thresholds[max_bin]
    lower_threshold = bin_thresholds[max_bin - 1] if max_bin > 0 else 0

    # find runs
    # holds (start, end exclusive) ranges
    runs = []
    curr_run_start = None
    run_end = None
    for i, enps in enumerate(enps_array):
        if not curr_run_start and enps > enps_threshold:
            curr_run_start = i
            continue

        if curr_run_start:
            if enps > enps_threshold:
                continue
            else:
                if enps > lower_threshold:
                    if enps_array[i - 1] > enps_threshold:
                        continue

                # end run
                if lower_threshold > enps_array[i - 1] > enps_threshold:
                    run_end = i - 1
                else:
                    run_end = i

                # append if run is long enough
                run_length = run_end - curr_run_start
                if run_length >= MIN_RUN_LENGTH:
                    runs.append((curr_run_start, run_end))
                curr_run_start = None
                run_end = None

    if curr_run_start:
        runs.append((curr_run_start, len(enps_array)))

    debug = args.setdefault('debug', False)
    if debug:
        print(max_bin, enps_threshold)
        print(runs)

    return runs


def annotate_enps_timeline():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
            'Your_Mind_-_Roy_Mikelate_D23_ARCADE.csv',
            'Ultimatum_-_Cosmograph_S21_ARCADE.csv',
            'STEP_-_SID-SOUND_D20_ARCADE.csv',
            'CO5M1C_R4ILR0AD_-_kanone_D22_ARCADE.csv',
            'GLORIA_-_Croire_D21_ARCADE.csv',
            'Wedding_Crashers_-_SHK_D23_ARCADE.csv',
            # 'Final_Audition_2__-_SHORT_CUT_-_-_Banya_S17_SHORTCUT.csv',
        ]
        chartstruct_files = [os.path.join(cs_folder, f) for f in chartstruct_files]

    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)

        # get list of enps
        enps_list = get_enps_list(cs)

        # find runs
        high_enps_ranges = find_runs(enps_list)

        cs.metadata['eNPS timeline data'] = enps_list
        cs.metadata['eNPS ranges of interest'] = high_enps_ranges

        if debug:
            import code; code.interact(local=dict(globals(), **locals()))

        if not debug:
            cs.to_csv(inp_fn)

    return


def main():
    annotate_enps_timeline()

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Update ChartStruct metadata with eNPS timeline data.
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524',
    )
    parser.add_argument(
        '--csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/Conflict_-_Siromaru_+_Cranky_D25_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/Nyarlathotep_-_nato_S21_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/BOOOM!!_-_RiraN_D22_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-112624/My_Dreams_-_Banya_Production_D22_ARCADE.csv',
    )
    args.parse_args(parser)
    main()