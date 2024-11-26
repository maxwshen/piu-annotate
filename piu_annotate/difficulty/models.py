"""
    Models for predicting difficulty
"""
import os
import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
from hackerargs import args
from loguru import logger

from sklearn.ensemble import HistGradientBoostingRegressor

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.difficulty import featurizers
from piu_annotate.segment.segment import Section


class DifficultyModelPredictor:
    def __init__(self):
        """ Class for predicting difficulty of chart segments.
            Loads multiple models trained on potentially different subsets of
            features, and can combine models to form final prediction.
        """
        self.model_path = args.setdefault(
            'difficulty_models_path',
            '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts'
        )

    def load_models(self) -> None:
        logger.info(f'Loading difficulty models from {self.model_path}')

        models: dict[str, HistGradientBoostingRegressor] = dict()
        for sd in ['singles', 'doubles']:
            for feature_subset in ['all', 'bracket', 'edp']:
                name = f'{sd}-{feature_subset}'
                logger.info(f'Loaded model: {name}')
                with open(os.path.join(self.model_path, name + '.pkl'), 'rb') as f:
                    model: HistGradientBoostingRegressor = pickle.load(f)
                models[name] = model
        self.models = models
        return

    def predict_segment_difficulties(self, cs: ChartStruct):
        """ Predict difficulties of chart segments.
            Featurizes each segment separately, which amounts to calculating
            the highest frequency of skill events in varying-length time windows
            in segment.
        """
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        fter = featurizers.DifficultyFeaturizer(cs)
        ft_names = fter.get_feature_names()
        xs = fter.featurize_sections(sections)
        sord = cs.singles_or_doubles()

        # prediction using all features
        y_all = self.predict(xs, sord)

        # prediction only using bracket frequencies
        y_bracket = self.predict_skill_subset('bracket', xs, sord, ft_names)
        # y_edp = self.predict_skill_subset('edp', xs, sord, ft_names)

        # adjust base prediction upward
        # predicted level tends to be low, as we featurize based on cruxes
        pred = y_all * (1/.95)

        # if bracket pred. model predicts higher difficulty than base prediction,
        # adjust prediction upwards.
        # Empirically, this helps because base prediction tends to place too little
        # importance on brackets.
        idxs = (y_bracket > pred)
        w = 0.66
        pred[idxs] = (1 - w) * pred[idxs] + w * y_bracket[idxs]

        level = cs.get_chart_level()
        pred[(pred <= level + 1)] += 0.5

        debug = args.setdefault('debug', False)
        if debug:
            print(cs.metadata['shortname'])
            # print(y_all)
            # print(y_bracket)
            # print(y_edp)
            # print(y_all * (1/.95) + 1)
            print(pred)
            import code; code.interact(local=dict(globals(), **locals()))

        return pred

    def predict(self, xs: npt.NDArray, sord: str):
        model = self.models[f'{sord}-all']
        return model.predict(xs)

    def predict_skill_subset(
        self, 
        feature_subset_name: str,
        xs: npt.NDArray,
        sord: str,
        ft_names: list[str]
    ) -> npt.NDArray:
        model = self.models[f'{sord}-{feature_subset_name}']
        ft_idxs = [i for i, nm in enumerate(ft_names) if feature_subset_name in nm]
        inp = xs[:, ft_idxs]
        pred = model.predict(inp)

        # change prediction to 0 for segments without any brackets
        missing_segments = np.where(np.all(inp == 0, axis = 1))
        pred[missing_segments] = 0
        return pred