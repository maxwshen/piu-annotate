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
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import Booster


supported_models = ['lightgbm']


class ModelWrapper:
    def __init__(self):
        pass

    def predict(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 1 binary array of 0 or 1 """
        raise NotImplementedError    

    def predict_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of p(0) and p(1) """
        raise NotImplementedError    
    
    def predict_log_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of logp(0) and logp(1) """
        raise NotImplementedError    


class ModelSuite:
    def __init__(self):
        """ Stores suite of ML models for limb prediction """
        model_type = args['model']
        assert model_type in supported_models, f'{model_type=} not in {supported_models=}'
        self.model_type = model_type

        self.model_arrows_to_limb = self.load('model.arrows_to_limb')
        self.model_arrowlimbs_to_limb = self.load('model.arrowlimbs_to_limb')
        self.model_arrows_to_matchnext = self.load('model.arrows_to_matchnext')
        self.model_arrows_to_matchprev = self.load('model.arrows_to_matchprev')

    def load(self, arg_key: str) -> ModelWrapper:
        if arg_key not in args:
            logger.error(f'Failed to find {arg_key} in args')
            exit(1)
        
        model_file = args[arg_key]
        if self.model_type == 'lightgbm':
            model = LGBModel.load(model_file)
        return model


class LGBModel(ModelWrapper):
    def __init__(self, bst: Booster):
        self.bst = bst

    @staticmethod
    def load(file: str):
        return LGBModel(lgb.Booster(model_file = file))

    @staticmethod
    def train(points: NDArray, labels: NDArray):
        train_x, test_x, train_y, test_y = train_test_split(points, labels)

        train_data = lgb.Dataset(train_x, label = train_y)
        test_data = lgb.Dataset(test_x, label = test_y)
        params = {'objective': 'binary', 'metric': 'binary_logloss'}
        bst = lgb.train(params, train_data, valid_sets = [test_data])
        return LGBModel(bst)

    def save(self, file: str) -> None:
        self.bst.save_model(file)

    def predict(self, points: NDArray) -> NDArray:
        """ For N points, returns N-length binary array of 0 or 1 """
        return self.bst.predict(points).round().astype(int)

    def predict_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of p(0) and p(1) """
        p = self.bst.predict(points)
        return np.stack([1 - p, p]).T
    
    def predict_log_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of logp(0) and logp(1) """
        return np.log(self.predict_prob(points))
    
