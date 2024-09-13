"""
    Model
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
from numpy.typing import NDArray
from operator import itemgetter
import itertools

from sklearn.ensemble import GradientBoostingClassifier


class ModelSuite:
    def __init__(self):
        """ Stores suite of ML models for limb prediction """
        self.model_arrows_to_limb = self.load('model.arrows_to_limb')
        self.model_arrowlimbs_to_limb = self.load('model.arrowlimbs_to_limb')
        self.model_arrows_to_matchnext = self.load('model.arrows_to_matchnext')
        self.model_arrows_to_matchprev = self.load('model.arrows_to_matchprev')

    def load(self, arg_key: str) -> GradientBoostingClassifier:
        if arg_key not in args:
            logger.error(f'Failed to find {arg_key} in args')
            exit(1)
        with open(args[arg_key], 'rb') as f:
            model = pickle.load(f)
        return model