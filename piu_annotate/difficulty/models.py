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
from collections import defaultdict

import lightgbm as lgb
from lightgbm import Booster

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.difficulty import featurizers
from piu_annotate.segment.segment import Section


class DifficultySegmentModelPredictor:
    def __init__(self):
        self.model_path = args.setdefault(
            'segment_difficulty_models_path',
            '/home/maxwshen/piu-annotate/artifacts/difficulty/segments/'
        )

        stepchart_dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/datasets/temp.pkl'
        with open(stepchart_dataset_fn, 'rb') as f:
            dataset = pickle.load(f)
        self.stepchart_dataset = dataset
        logger.info(f'Loaded dataset from {stepchart_dataset_fn}')

    def load_models(self) -> None:
        logger.info(f'Loading segment difficulty models from {self.model_path}')

        models: dict[str, Booster] = dict()
        for sd in ['singles', 'doubles']:
            name = f'{sd}'
            logger.info(f'Loaded model: {name}')
            model_fn = os.path.join(self.model_path, f'lgbm-{name}.txt')
            model = lgb.Booster(model_file = model_fn)
            models[name] = model
        self.models = models
        return

    def predict_segments(
        self, 
        cs: ChartStruct, 
        xs: npt.NDArray,
        ft_names: list[str],
    ) -> list[dict]:
        """ Predicts segments featurized into `xs` from ChartStruct `cs`.
        """
        sord = cs.singles_or_doubles()
        chart_level = cs.get_chart_level()
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]

        # prediction using all features
        model = self.models[sord]
        y_all = model.predict(xs)
        pred = y_all.copy()

        # clip
        pred = np.clip(pred, 0.7, 28.3)

        debug = args.setdefault('debug', False)

        # rare skill
        rare_skill_cands = {
            'twistclose-5': 96, 
            'jump-2': 96, 
            'jack-5': 96,
            'bracket-5': 96,
            'bracket run-5': 96,
            'bracket drill-5': 96,
            'bracket jump-5': 96,
            'bracket twist-5': 96,
        }
        # only use doublestep as rare skill for manually annotated stepcharts,
        # because doublestep is a common error for predicted limb annotations,
        # especially on chart sections with holds and taps
        if cs.metadata['Manual limb annotation']:
            rare_skill_cands['doublestep-7'] = 99

        stepchart_data = self.stepchart_dataset['x']
        stepchart_levels = self.stepchart_dataset['y']
        # maps segment idx to list of rare skills
        rare_skill_dd = defaultdict(list)
        for rare_skill_name, percentile_threshold in rare_skill_cands.items():
            ft_idx = ft_names.index(rare_skill_name)
            threshold = np.percentile(
                stepchart_data[stepchart_levels <= chart_level, ft_idx], 
                percentile_threshold
            )
            rare_skill_idxs = xs[:, ft_idx] > threshold
            if rare_skill_idxs.any():
                for i in np.where(rare_skill_idxs)[0]:
                    rare_skill_dd[i].append(rare_skill_name)
                if debug:
                    print(rare_skill_name, rare_skill_idxs)

        if debug:
            print(cs.metadata['shortname'])
            print(y_all)
            print(pred)
            import code; code.interact(local=dict(globals(), **locals()))

        segment_dicts = []
        for i in range(len(sections)):
            d = {
                'level': np.round(pred[i], 2),
                'rare skills': rare_skill_dd[i],
            }
            segment_dicts.append(d)
        return segment_dicts

    def predict_segment_difficulties(self, cs: ChartStruct) -> list[dict]:
        """ Predict difficulties of chart segments from `cs`, by
            first featurizing segments in cs.

            Featurizes each segment separately, which amounts to calculating
            the highest frequency of skill events in varying-length time windows
            in segment.

            Returns a list of dicts, one dict per segment.
        """
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        fter = featurizers.DifficultySegmentFeaturizer(cs)
        ft_names = fter.get_feature_names()
        xs = fter.featurize_sections(sections)
        sord = cs.singles_or_doubles()

        segment_dicts = self.predict_segments(xs, sord, ft_names)
        return segment_dicts


class DifficultyStepchartModelPredictor:
    def __init__(self):
        """ Holds models trained to predict difficulty on stepcharts.
            Class exposes methods for running predictions on new stepcharts,
            or on segments.

            Loads multiple models trained on potentially different subsets of
            features, and can combine models to form final prediction.
        """
        self.model_path = args.setdefault(
            'stepchart_difficulty_models_path',
            '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts'
        )

        dataset_fn = '/home/maxwshen/piu-annotate/artifacts/difficulty/full-stepcharts/datasets/temp.pkl'
        with open(dataset_fn, 'rb') as f:
            dataset = pickle.load(f)
        self.dataset = dataset
        logger.info(f'Loaded dataset from {dataset_fn}')

    def load_models(self) -> None:
        logger.info(f'Loading stepchart difficulty models from {self.model_path}')

        models: dict[str, Booster] = dict()
        for sd in ['singles', 'doubles']:
            for feature_subset in ['all', 'bracket', 'edp']:
                name = f'{sd}-{feature_subset}'
                logger.info(f'Loaded model: {name}')
                model_fn = os.path.join(self.model_path, f'lgbm-{name}.txt')
                model = lgb.Booster(model_file = model_fn)
                models[name] = model
        self.models = models
        return

    def predict_stepchart(self, cs: ChartStruct):
        fter = featurizers.DifficultyStepchartFeaturizer(cs)
        x = fter.featurize_full_stepchart()
        x = x.reshape(1, -1)
        return self.predict(x, cs.singles_or_doubles())

    def predict_segments(
        self, 
        cs: ChartStruct, 
        xs: npt.NDArray,
        ft_names: list[str],
    ) -> list[dict]:
        """ Predicts segments featurized into `xs` from ChartStruct `cs`.
        """
        sord = cs.singles_or_doubles()
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]

        # prediction using all features
        y_all = self.predict(xs, sord)

        # prediction only using bracket frequencies
        # y_bracket = self.predict_skill_subset('bracket', xs, sord, ft_names)
        # y_edp = self.predict_skill_subset('edp', xs, sord, ft_names)

        # adjust base prediction upward
        # predicted level tends to be low, as we featurize based on cruxes
        pred = y_all * (1/.958)

        chart_level = cs.get_chart_level()

        # adjust underrated charts towards chart level
        max_segment_level = max(pred)
        if max_segment_level < chart_level:
            # pick up segments with difficulty close to max
            adjustment = (chart_level - max_segment_level) / 2
            pred[(pred >= max_segment_level - 3)] += adjustment

        before_rare_skill = pred.copy()

        # clip
        pred = np.clip(pred, 0.7, 28.3)

        debug = args.setdefault('debug', False)

        # rare skill
        rare_skill_cands = {
            'twistclose-5': 96, 
            'jump-2': 96, 
            'jack-5': 96,
            'edp-5': 90, 
            'bracket-5': 96,
            'bracket run-7': 98,
            # 'bracket drill-5': 96,
            # 'bracket jump-5': 96,
            'bracket twist-7': 98,
        }
        # only use doublestep as rare skill for manually annotated stepcharts,
        # because doublestep is a common error for predicted limb annotations,
        # especially on chart sections with holds and taps
        if cs.metadata['Manual limb annotation']:
            rare_skill_cands['doublestep-7'] = 99

        train_data = self.dataset['x']
        train_levels = self.dataset['y']
        # maps segment idx to list of rare skills
        rare_skill_dd = defaultdict(list)
        for rare_skill_name, percentile_threshold in rare_skill_cands.items():
            ft_idx = ft_names.index(rare_skill_name)
            threshold = np.percentile(
                train_data[train_levels <= chart_level, ft_idx], 
                percentile_threshold
            )
            rare_skill_idxs = xs[:, ft_idx] > threshold
            if rare_skill_idxs.any():
                if debug:
                    print(rare_skill_name, rare_skill_idxs)
            
                for i in np.where(rare_skill_idxs)[0]:
                    # set difficulty floor based on official stepchart level
                    if pred[i] < chart_level + 0.35:
                        pred[i] = chart_level + 0.35
                    else:
                        # for multiple rare skills, or if segment is already predicted
                        # to be hard, lift difficulty beyond
                        # if pred[i] == chart_level + 0.35:
                            # pred[i] += 0.5
                        pass

                    rare_skill_dd[i].append(rare_skill_name)

        if debug:
            print(cs.metadata['shortname'])
            print(y_all)
            print(before_rare_skill)
            # print(y_bracket)
            # print(y_edp)
            # print(y_all * (1/.95) + 1)
            print(pred)
            import code; code.interact(local=dict(globals(), **locals()))

        segment_dicts = []
        for i in range(len(sections)):
            d = {
                'level': np.round(pred[i], 2),
                'rare skills': rare_skill_dd[i],
            }
            segment_dicts.append(d)

        return segment_dicts

    def predict_segment_difficulties(self, cs: ChartStruct) -> list[dict]:
        """ Predict difficulties of chart segments from `cs`, by
            first featurizing segments in cs.

            Featurizes each segment separately, which amounts to calculating
            the highest frequency of skill events in varying-length time windows
            in segment.

            Returns a list of dicts, one dict per segment.
        """
        sections = [Section.from_tuple(tpl) for tpl in cs.metadata['Segments']]
        fter = featurizers.DifficultyStepchartFeaturizer(cs)
        ft_names = fter.get_feature_names()
        xs = fter.featurize_sections(sections)
        sord = cs.singles_or_doubles()

        segment_dicts = self.predict_segments(xs, sord, ft_names)
        return segment_dicts

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