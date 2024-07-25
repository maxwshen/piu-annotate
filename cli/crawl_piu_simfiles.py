"""
    Crawls PIU-Simfiles folder
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm

from piu_annotate.formats.sscfile import SongSSC, StepchartSSC

SKIP_PACKS = ['INFINITY']


def crawl_sscs(base_simfiles_folder: str) -> list[SongSSC]:
    """ Crawls `base_simfiles_folder` assuming structure:
        base_simfiles_folder / <pack_folder> / < song folder > / song.ssc.
        Returns list of SongSSC objects.
    """
    ssc_files = []
    packs = []

    for dirpath, _, files in os.walk(base_simfiles_folder):
        subdir = dirpath.replace(base_simfiles_folder, '')
        pack = subdir.split(os.sep)[0].split(' - ')[-1]
        level = subdir.count(os.sep)

        if pack in SKIP_PACKS:
            continue

        if level == 1:
            for file in files:
                if file.endswith('.ssc'):
                    ssc_files.append(os.path.join(dirpath, file))
                    packs.append(pack)
    logger.info(f'Found {len(ssc_files)} .ssc files in {base_simfiles_folder}')
    logger.info(f'Found packs: {sorted(list(set(packs)))}')

    valid = []
    invalid = []
    song_sscs = []
    valid_song_sscs = []
    for ssc_file, pack in tqdm(zip(ssc_files, packs), total = len(ssc_files)):
        song_ssc = SongSSC(ssc_file, pack)
        song_sscs.append(song_ssc)

        if song_ssc.validate():
            valid.append(ssc_file)
            valid_song_sscs.append(song_ssc)
        else:
            invalid.append(ssc_file)

    logger.success(f'Found {len(valid)} valid song ssc files')
    if len(invalid):
        logger.error(f'Found {len(invalid)} invalid song ssc files')
    else:
        logger.info(f'Found {len(invalid)} invalid song ssc files')
    return valid_song_sscs


def main():
    simfiles_folder = args['simfiles_folder']

    logger.info(f'Skipping packs: {SKIP_PACKS}')

    song_sscs = crawl_sscs(simfiles_folder)

    # get stepcharts
    stepcharts: list[StepchartSSC] = []
    for song in song_sscs:
        stepcharts += song.stepcharts
    logger.info(f'Found {len(stepcharts)} stepcharts')

    standard_stepcharts = [sc for sc in stepcharts if not sc.is_nonstandard()]
    logger.success(f'Found {len(standard_stepcharts)} standard stepcharts')

    from piu_annotate.formats.ssc_to_chartstruct import stepchart_ssc_to_chartstruct
    from collections import defaultdict
    logger.info(f'Checking stepcharts ...')
    nbl = defaultdict(int)
    for stepchart in tqdm(standard_stepcharts):
        cs_df, num_bad_lines = stepchart_ssc_to_chartstruct(stepchart)
        nbl[num_bad_lines] += 1
    
    print(nbl)

    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Crawls PIU-Simfiles folder
        """
    )
    parser.add_argument(
        '--simfiles_folder', 
        default = '/home/maxwshen/PIU-Simfiles/'
    )
    args.parse_args(parser)
    main()