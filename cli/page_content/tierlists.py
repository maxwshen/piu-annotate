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
import json
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.utils import make_basename_url_safe
from cli.difficulty.train_segment_difficulty import build_segment_feature_store
from piu_annotate.difficulty.models import DifficultyStepchartModelPredictor
from piu_annotate import utils
from piu_annotate.formats.nps import calc_bpm


def make_data_struct():
    dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/tierlist-struct.pkl'
    rerun_all = args.setdefault('rerun', False)
    if not rerun_all:
        if os.path.exists(dataset_fn):
            with open(dataset_fn, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded feature store from {dataset_fn}')
            return dataset

    cs_folder = args['chart_struct_csv_folder']
    logger.info(f'Using {cs_folder=}')
    chartstruct_files = [fn for fn in os.listdir(cs_folder) if fn.endswith('.csv')]
    logger.info(f'Found {len(chartstruct_files)} ChartStruct CSVs ...')

    ft_store = build_segment_feature_store('stepchart')
    segment_ft_store = build_segment_feature_store('segment')

    # Load models
    dmp = DifficultyStepchartModelPredictor()
    dmp.load_models()

    dd = defaultdict(list)

    ft_names = ft_store['feature names']

    logger.info('Predicting stepchart difficulties ...')
    for cs_file in tqdm(chartstruct_files):
        inp_fn = os.path.join(cs_folder, cs_file)
        cs = ChartStruct.from_file(inp_fn)
        shortname = cs.metadata['shortname']
        sord = cs.singles_or_doubles()

        x = ft_store[f'{shortname}-fullstepchart']
        x = x.reshape(1, -1)

        # pred_stepchart_level = dmp.predict(x, sord)[0]
        # dd['pred-level'].append(pred_stepchart_level)
        max_segment_level = max([d['level'] for d in cs.metadata['Segment metadata']])
        dd['pred-level'].append(max_segment_level)
        dd['Shortname'].append(shortname)
        dd['sord-level'].append(f'{sord[0].upper()}{cs.get_chart_level()}')

        dd['enps-95th-pct'].append(
            np.percentile(cs.metadata['eNPS timeline data'], 95)
        )

        edp2 = x[0][ft_names.index('edp-2')]
        dd['edp-2'].append(edp2)

    with open(dataset_fn, 'wb') as f:
        pickle.dump(dd, f)
    logger.info(f'Saved dataset to {dataset_fn}')
    return dd


def get_notetype_and_bpm_info(nps: float):
    # get allowed note type
    if nps < 0.8333:
        allowed_notetype = [1]
    elif nps < 1.666:
        allowed_notetype = [2]
    elif nps < 3.333:
        allowed_notetype = [4]
    elif nps < 6.666:
        allowed_notetype = [8]
    else:
        allowed_notetype = [16]

    bpm, notetype = calc_bpm(1 / nps, None, allowed_notetypes = allowed_notetype)
    return f'{notetype} @ {round(bpm)} bpm'


def main():
    cs_folder = args['chart_struct_csv_folder']

    dd = make_data_struct()

    # reformat dd into tierlists jsons
    output_file = os.path.join(cs_folder, 'page-content', 'tierlists.json')
    utils.make_dir(output_file)

    tierlist_struct = dict()
    for sordlevel in sorted(set(dd['sord-level'])):
        idxs = [i for i, sdl in enumerate(dd['sord-level']) if sdl == sordlevel]
        # edps = np.array(dd['edp-2'])[idxs]
        edps = np.array(dd['enps-95th-pct'])[idxs]
        charts = np.array(dd['Shortname'])[idxs]
        pred_levels = np.array(dd['pred-level'])[idxs]
        num_charts = len(idxs)

        # find number of groups, based on num. charts in level
        n_groups = min(int(np.floor(num_charts / 5) + 1), 5)
        interval = int(100 / n_groups)
        while True:
            edp_percentiles = [
                np.round(np.percentile(edps, n), 2)
                for n in range(interval, 100, interval)
            ]
            if len(set(edp_percentiles)) != len(edp_percentiles):
                n_groups -= 1
                interval = int(100 / n_groups)
            else:
                break

        edp_percentiles.insert(0, min(edps))
        edp_percentiles.append(max(edps) + 0.1)
        def floor_to_decimal_pts(a: float) -> float:
            n_decimals = 2
            return ((a*10**n_decimals)//1)/(10**n_decimals)
        edp_percentiles = [floor_to_decimal_pts(x) for x in edp_percentiles]

        # place charts into groups
        groups = dict()
        pairs = list(itertools.pairwise(edp_percentiles))
        for lower, upper in pairs[::-1]:
            selector = (edps >= lower) & (edps < upper)
            charts_in_group = charts[selector]
            charts_in_group = [utils.make_basename_url_safe(c) for c in charts_in_group]
            levels_in_group = pred_levels[selector]
            
            chart_to_level = {c: l for c, l in zip(charts_in_group, levels_in_group)}
            sorted_charts = sorted(chart_to_level, key = chart_to_level.get, reverse = True)
            sorted_levels = [chart_to_level[c] for c in sorted_charts]

            # get note type and BPM info
            lower_notetype_bpm = get_notetype_and_bpm_info(lower)
            upper_notetype_bpm = get_notetype_and_bpm_info(upper)

            if upper != edp_percentiles[-1]:
                name = f'{lower}-{upper} NPS\n{lower_notetype_bpm} - {upper_notetype_bpm}'
            else:
                name = f'{lower}+ NPS\n>{lower_notetype_bpm}'
            
            if len(sorted_charts) > 0:
                groups[name] = (sorted_charts, sorted_levels)

        tierlist_struct[sordlevel] = groups
        print(sordlevel, groups.keys())
        # for k, v in groups.items():
            # print(k, v)

    with open(output_file, 'w') as f:
        json.dump(tierlist_struct, f)
    import code; code.interact(local=dict(globals(), **locals()))

    logger.success('done')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Create content for tier lists page
    """)
    parser.add_argument(
        '--chart_struct_csv_folder', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/120524/lgbm-120524/',
    )
    args.parse_args(parser)
    main()