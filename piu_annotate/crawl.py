import os
from loguru import logger
from tqdm import tqdm


from piu_annotate.formats.sscfile import SongSSC, StepchartSSC


def crawl_sscs(
    base_simfiles_folder: str,
    skip_packs: list[str],
) -> list[SongSSC]:
    """ Crawls `base_simfiles_folder` assuming structure:
        base_simfiles_folder / <pack_folder> / < song folder > / song.ssc.
        Skips pack_folder in `skip_packs`.
        Returns list of SongSSC objects.
    """
    ssc_files = []
    packs = []

    for dirpath, _, files in os.walk(base_simfiles_folder):
        subdir = dirpath.replace(base_simfiles_folder, '')
        pack = subdir.split(os.sep)[0].split(' - ')[-1]
        level = subdir.count(os.sep)

        if pack in skip_packs:
            continue

        if level == 1:
            for file in files:
                if file.lower().endswith('.ssc'):
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


def crawl_stepcharts(
    base_simfiles_folder: str,
    skip_packs: list[str],
) -> list[StepchartSSC]:
    song_sscs = crawl_sscs(base_simfiles_folder, skip_packs = skip_packs)
    # get stepcharts
    stepcharts: list[StepchartSSC] = []
    for song in song_sscs:
        stepcharts += song.stepcharts
    logger.info(f'Found {len(stepcharts)} stepcharts')
    return stepcharts
