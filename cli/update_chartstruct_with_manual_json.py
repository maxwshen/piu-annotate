import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct


def main():
    chart_json = ChartJsStruct.from_json(args['chart_json'])
    cs = ChartStruct.from_file(args['chart_struct_csv'])

    print(cs.matches_chart_json(chart_json))
    print(cs.matches_chart_json(chart_json, with_limb_annot = False))

    cs.update_from_manual_json(chart_json)

    if cs.matches_chart_json(chart_json):
        logger.success(f'Successfully updated chartstruct with json!')
    else:
        logger.error(f'Failed')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Updates ChartStruct using a manually annotated chart json
    """)
    parser.add_argument(
        '--chart_json', 
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-jsons/piucenter-070824-v1/Clematis_Rapsodia_-_Jehezukiel_S22_arcade.json',
        default = '/home/maxwshen/piu-annotate/artifacts/manual-jsons/piucenter-070824-v1/Silhouette_Effect_-_Nato_S7_arcade.json',
        # default = '/home/maxwshen/piu-annotate/artifacts/manual-jsons/piucenter-070824-v1/ASDF_-_Doin_S17_arcade.json',
    )
    parser.add_argument(
        '--chart_struct_csv', 
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-dijkstra-090124/Clematis_Rapsodia_-_Jehezukiel_S22_arcade.csv',
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/rayden-072924-arroweclipse-072824/Silhouette_Effect_-_Nato_S7_ARCADE.csv',
        # default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-dijkstra-090124/ASDF_-_Doin_S17_arcade.csv',
    )
    args.parse_args(parser)
    main()