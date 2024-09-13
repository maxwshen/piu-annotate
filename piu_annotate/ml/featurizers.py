"""
    Featurize
"""
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from hackerargs import args

from piu_annotate.formats.chart import ChartStruct, PredictionCoordinate
from piu_annotate.formats import notelines
from piu_annotate.ml.datapoints import LimbLabel, ArrowDataPoint, ArrowDataPointWithLimbContext


class ChartStructFeaturizer:
    def __init__(self, cs: ChartStruct):
        """ Featurizer for ChartStruct, generating:
            - list of ArrowDataPoint
            - list of ArrowDataPointWithLimbContext
            - list of LimbLabels
            and creating prediction inputs for each arrow with context,
            as NDArrays
        """
        self.cs = cs

        cs.annotate_time_since_downpress()
        cs.annotate_line_repeats_previous()
        cs.annotate_line_repeats_next()
        self.pred_coords: list[PredictionCoordinate] = self.cs.get_prediction_coordinates()

        self.singles_or_doubles = cs.singles_or_doubles()
        self.points_nolimb = self.get_arrows_nolimb()

    """
        Build
    """    
    def get_arrows_nolimb(self) -> list[ArrowDataPoint]:
        all_points_nolimb = []
        for pred_coord in self.pred_coords:
            row = self.cs.df.iloc[pred_coord.row_idx]
            line = row['Line']
            point = ArrowDataPoint(
                arrow_pos = pred_coord.arrow_pos,
                time_since_prev_downpress = row['__time since prev downpress'],
                n_arrows_in_same_line = line.count('1') + line.count('2'),
                line_repeats_previous = row['__line repeats previous'],
                line_repeats_next = row['__line repeats next'],
                singles_or_doubles = self.singles_or_doubles,
            )
            all_points_nolimb.append(point)
        return all_points_nolimb

    def get_arrows_withlimb(
        self, 
        limb_array: NDArray,
    ) -> list[ArrowDataPointWithLimbContext]:
        all_points_withlimb = []
        assert limb_array is not None
        assert len(limb_array) == len(self.pred_coords)
        for pred_coord, limb_val in zip(self.pred_coords, limb_array):
            row = self.cs.df.iloc[pred_coord.row_idx]
            line = row['Line']
            point = ArrowDataPointWithLimbContext(
                arrow_pos = pred_coord.arrow_pos,
                time_since_prev_downpress = row['__time since prev downpress'],
                n_arrows_in_same_line = line.count('1') + line.count('2'),
                line_repeats_previous = row['__line repeats previous'],
                line_repeats_next = row['__line repeats next'],
                limb_annot = limb_val,
                singles_or_doubles = self.singles_or_doubles,
            )
            all_points_withlimb.append(point)
        return all_points_withlimb

    def get_labels_from_limb_col(self, limb_col: str) -> NDArray:
        all_labels = []
        for pred_coord in self.pred_coords:
            row = self.cs.df.iloc[pred_coord.row_idx]
            label = LimbLabel.from_limb_annot(
                row[limb_col][pred_coord.limb_idx],
            )
            all_labels.append(label)
        return np.stack([label.to_array() for label in all_labels])

    def get_label_matches_next(self, limb_col: str) -> NDArray:
        labels = self.get_labels_from_limb_col(limb_col)
        return np.concatenate([labels[:-1] == labels[1:], [False]]).astype(int)

    def get_label_matches_prev(self, limb_col: str) -> NDArray:
        labels = self.get_labels_from_limb_col(limb_col)
        return np.concatenate([[False], labels[1:] == labels[:-1]]).astype(int)

    """
        Featurize
    """
    def featurize_arrows_with_context(self) -> NDArray:
        """ For N arrows with D feature dims, constructs prediction input
            for each arrow including context arrows on both sides.
            
            If not using limb_context, returns shape N x [(2*context_len + 1)*D]
            If using limb_context, includes limb features (observed for training,
            predicted at test time) in context arrows.
        """
        pt_nolimb_array = [point.to_array() for point in self.points_nolimb]

        # Concatenate arrows and pad
        context_len = args.setdefault('ft.context_length', 20)
        empty_pt_nolimb = np.zeros(len(pt_nolimb_array[0]))

        padded_nolimb = [empty_pt_nolimb]*context_len + pt_nolimb_array + \
                        [empty_pt_nolimb]*context_len

        # Form prediction inputs: each arrow, surrounded by `context_len` arrows
        # on both sides.
        ft_inps = []
        for i in range(context_len, len(padded_nolimb) - context_len):
            window = np.concatenate(padded_nolimb[i - context_len : i + context_len + 1])
            ft_inps.append(window)
        return np.array(ft_inps)

    def featurize_arrowlimbs_with_context(self, limb_probs: NDArray) -> NDArray:
        """ For N arrows with D feature dims, constructs prediction input
            for each arrow including context arrows on both sides.
            
            If not using limb_context, returns shape N x [(2*context_len + 1)*D]
            If using limb_context, includes limb features (observed for training,
            predicted at test time) in context arrows.
        """
        pt_nolimb_array = [point.to_array() for point in self.points_nolimb]

        points_withlimb = self.get_arrows_withlimb(limb_probs)
        pt_withlimb_array = [point.to_array() for point in points_withlimb]

        # Concatenate arrows and pad
        context_len = args.setdefault('ft.context_length', 20)
        empty_pt_withlimb = np.zeros(len(pt_withlimb_array[0]))
        empty_pt_nolimb = np.zeros(len(pt_nolimb_array[0]))

        padded_nolimb = [empty_pt_nolimb]*context_len + pt_nolimb_array + \
                        [empty_pt_nolimb]*context_len
        padded_withlimb = [empty_pt_withlimb]*context_len + pt_withlimb_array + \
                        [empty_pt_withlimb]*context_len

        # Form prediction inputs: each arrow, surrounded by `context_len` arrows
        # on both sides.
        ft_inps = []
        for i in range(context_len, len(padded_nolimb) - context_len):
            window = np.concatenate([
                np.concatenate(padded_withlimb[i - context_len : i]),
                padded_nolimb[i],
                np.concatenate(padded_withlimb[i + 1 : i + context_len])
            ])
            ft_inps.append(window)
        return np.array(ft_inps)

    """
        Evaluation
    """
    def evaluate(self, pred_limbs: NDArray) -> dict[str, any]:
        """ Evaluate vs 'Limb annotation' column """
        labels = self.get_labels_from_limb_col('Limb annotation')
        eval_dict = {
            'accuracy': np.sum(labels == pred_limbs) / len(labels), 
            'error_idxs': np.where(labels != pred_limbs),
        }
        for k, v in eval_dict.items():
            logger.debug(f'{k}={v}')
        return eval_dict
