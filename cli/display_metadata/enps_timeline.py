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

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import nps
from piu_annotate.segment.segment import Section
from piu_annotate.utils import make_basename_url_safe



def annotate_enps_timeline():
    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    debug = args.setdefault('debug', False)
    if debug:
        chartstruct_files = [
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

        edp_times = nps.calc_effective_downpress_times(cs)
        edp_times = np.array(edp_times)
        
        time_since_edp = edp_times[1:] - edp_times[:-1]
        time_since_edp = np.insert(time_since_edp, 0, 3)
        # make shape match edp_times

        second_to_num_downpresses = dict(Counter(np.floor(edp_times).astype(int)))

        from collections import defaultdict
        second_to_enps = dict()

        ts = list(edp_times)
        tdp = list(time_since_edp)
        
        max_time = round(max(cs.df['Time']) + 1)
        for t in range(0, max_time - 1):
            num_popped = 0
            tdps = []
            while ts and ts[0] < t + 1:
                ts.pop(0)
                tdps.append(tdp.pop(0))
                num_popped += 1

            # Filter very small time since
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

        # # round to nearest second, count
        # second_to_enps = dict(Counter(np.round(edp_times).astype(int)))

        # enps_vector = np.zeros(round(max(cs.df['Time']) + 1))

        # convert to list
        enps_vector = np.zeros(max_time)
        to_np = lambda x: np.array(list(x))
        enps_vector[to_np(second_to_enps.keys())] = to_np(second_to_enps.values())

        cs.metadata['eNPS timeline data'] = list(enps_vector)

        if debug:
            enps_vals = np.array(list(second_to_enps.values()))
            print(cs.metadata['shortname'], sum(enps_vals > 8))
            import code; code.interact(local=dict(globals(), **locals()))

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