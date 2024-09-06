import os
from pathlib import Path


def make_dir(filename: str) -> None:
    """ Creates directories for filename """
    Path(os.path.dirname(filename)).mkdir(parents = True, exist_ok = True)
    return
