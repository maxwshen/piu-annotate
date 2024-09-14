"""
    Featurize
"""
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from hackerargs import args
import functools

from piu_annotate.formats.chart import ChartStruct, PredictionCoordinate
from piu_annotate.formats import notelines
from piu_annotate.ml.datapoints import LimbLabel, ArrowDataPoint


class ChartStructFeaturizer:
    def __init__(self, cs: ChartStruct):
        """ Featurizer for ChartStruct, generating:
            - list of ArrowDataPoint
            - list of LimbLabels
            and creating prediction inputs for each arrow with context,
            as NDArrays
        """
        self.cs = cs
        self.context_len = args.setdefault('ft.context_length', 20)

        cs.annotate_time_since_downpress()
        cs.annotate_line_repeats_previous()
        cs.annotate_line_repeats_next()
        self.pred_coords: list[PredictionCoordinate] = self.cs.get_prediction_coordinates()

        self.singles_or_doubles = cs.singles_or_doubles()
        self.points_nolimb = self.get_arrows_nolimb()
        self.pt_array = [pt.to_array() for pt in self.points_nolimb]

    """
        Build
    """    
    def get_arrows_nolimb(self) -> list[ArrowDataPoint]:
        all_points_nolimb = []
        for pred_coord in self.pred_coords:
            row = self.cs.df.iloc[pred_coord.row_idx]
            line = row['Line with active holds'].replace('`', '')
            arrow_pos = pred_coord.arrow_pos
            point = ArrowDataPoint(
                arrow_pos = arrow_pos,
                is_hold = bool(line[arrow_pos] == '2'),
                active_hold_idxs = [i for i, s in enumerate(line) if s in list('34')],
                time_since_prev_downpress = row['__time since prev downpress'],
                n_arrows_in_same_line = line.count('1') + line.count('2'),
                line_repeats_previous = row['__line repeats previous'],
                line_repeats_next = row['__line repeats next'],
                singles_or_doubles = self.singles_or_doubles,
            )
            all_points_nolimb.append(point)
        return all_points_nolimb

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
    @functools.cache
    def get_padded_array(self) -> NDArray:
        pt_array = self.pt_array
        context_len = self.context_len
        empty_pt = np.zeros(len(pt_array[0]))
        return np.array([empty_pt]*context_len + pt_array + [empty_pt]*context_len)

    @functools.cache
    def featurize_arrows_with_context(self) -> NDArray:
        """ For N arrows with D feature dims, constructs prediction input
            for each arrow including context arrows on both sides.
            
            If not using limb_context, returns shape N x [(2*context_len + 1)*D]
        """
        padded_pts = self.get_padded_array()
        context_len = self.context_len

        c2_plus_1 = 2 * context_len + 1
        view = np.lib.stride_tricks.sliding_window_view(
            padded_pts, 
            (c2_plus_1), 
            axis = 0
        )
        (N, D, c2_plus_1) = view.shape
        view = np.reshape(view, (N, D*c2_plus_1))
        return view

    def featurize_arrowlimbs_with_context(self, limb_probs: NDArray) -> NDArray:
        """ Include `limb_probs` as features.
            At training, limb_probs are binary.
            At test time, limb_probs can be floats or binary.
            
            For speed, we precompute featurized arrows into np.array,
            and concatenate this to limb_probs subsection in sliding windows.
        """
        context_len = self.context_len
        c2_plus_1 = 2 * context_len + 1

        arrow_view = self.featurize_arrows_with_context()

        padded_limbs = np.concatenate([
            [-1]*context_len,
            limb_probs,
            [-1]*context_len
        ])
        limb_view = np.lib.stride_tricks.sliding_window_view(
            padded_limbs, 
            (c2_plus_1), 
            axis = 0
        )
        # remove limb annot for arrow to predict on, but keep context
        limb_view = np.concatenate([
            limb_view[:, :context_len],
            limb_view[:, -context_len:]
        ], axis = 1)
        array_view = np.concatenate([arrow_view, limb_view], axis = 1)
        return array_view

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
