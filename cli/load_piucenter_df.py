import argparse
from hackerargs import args

from piu_annotate.formats.piucenterdf import PiuCenterDataFrame
from piu_annotate.formats.chart import ChartStruct


def main():
    pc_df = PiuCenterDataFrame(args['file'])
    cs = ChartStruct.from_piucenterdataframe(pc_df)

    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', 
        default = '/home/maxwshen/piu-annotate/jupyter/Conflict - Siromaru + Cranky D24 arcade.csv'
    )
    args.parse_args(parser)
    main()