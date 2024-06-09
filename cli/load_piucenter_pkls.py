import argparse
from hackerargs import args

from piu_annotate.formats.piucenter import PiuCenterStruct

def main():
    struct = PiuCenterStruct(args['file'])
    import code; code.interact(local=dict(globals(), **locals()))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', 
        default = '/home/maxwshen/piu-annotate/jupyter/Hyperion - M2U D21 shortcut.pkl'
    )
    args.parse_args(parser)
    main()