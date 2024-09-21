import os
from pathlib import Path


def make_dir(filename: str) -> None:
    """ Creates directories for filename """
    Path(os.path.dirname(filename)).mkdir(parents = True, exist_ok = True)
    return


def make_basename_url_safe(text: str) -> str:
    """ Makes a basename safe for URL. Removes / too. """
    ok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:?#[]@!$&()*+,;=')
    return ''.join([c for c in text if c in ok])