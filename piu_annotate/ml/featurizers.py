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

        self.cs.annotate_time_since_downpress()
        self.cs.annotate_line_repeats_previous()
        self.cs.annotate_line_repeats_next()
        self.pred_coords: list[PredictionCoordinate] = self.cs.get_prediction_coordinates()

        self.singles_or_doubles = cs.singles_or_doubles()
        self.points_nolimb = self.get_arrows_nolimb()
        self.pt_array = [pt.to_array() for pt in self.points_nolimb]

        self.pc_idx_to_prev = self.cs.get_previous_used_pred_coord_for_arrow()
        # shift by +1, and replace None with 0
        shifted = [x if x is not None else -1 for x in self.pc_idx_to_prev.values()]
        self.prev_pc_idx_shifted = np.array(shifted) + 1

    """
        Build
    """    
    def get_arrows_nolimb(self) -> list[ArrowDataPoint]:
        all_points_nolimb = []
        pc_to_time_last_arrow_use = self.cs.get_time_since_last_same_arrow_use()
        for idx, pred_coord in enumerate(self.pred_coords):
            row = self.cs.df.iloc[pred_coord.row_idx]
            line = row['Line with active holds'].replace('`', '')

            same_line_as_next_datapoint = False
            if idx + 1 < len(self.pred_coords):
                if self.pred_coords[idx + 1].row_idx == pred_coord.row_idx:
                    same_line_as_next_datapoint = True

            arrow_pos = pred_coord.arrow_pos
            point = ArrowDataPoint(
                arrow_pos = arrow_pos,
                is_hold = bool(line[arrow_pos] == '2'),
                active_hold_idxs = [i for i, s in enumerate(line) if s in list('34')],
                same_line_as_next_datapoint = same_line_as_next_datapoint,
                time_since_last_same_arrow_use = pc_to_time_last_arrow_use[pred_coord],
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

        # add feature for limb used for nearby arrows
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

        # add feature for prev limb used for nearby arrows
        shifted_limb_probs = np.concatenate([[-1], limb_probs])
        # shift enables using value -1 for no previous
        prev_limb_probs = shifted_limb_probs[self.prev_pc_idx_shifted]
        prev_limb_feature_context_len = 8
        padded_prev_limb_probs = np.concatenate([
            [-1]*prev_limb_feature_context_len,
            prev_limb_probs,
            [-1]*prev_limb_feature_context_len,
        ])
        prev_limb_view = np.lib.stride_tricks.sliding_window_view(
            padded_prev_limb_probs, 
            (2*prev_limb_feature_context_len + 1), 
            axis = 0
        )
        features = np.concatenate([arrow_view, limb_view, prev_limb_view], axis = 1)
        return features

    """
        Evaluation
    """
    def evaluate(self, pred_limbs: NDArray, verbose: bool = False) -> dict[str, any]:
        """ Evaluate vs 'Limb annotation' column """
        labels = self.get_labels_from_limb_col('Limb annotation')
        accuracy = np.sum(labels == pred_limbs) / len(labels)
        eval_dict = {
            'accuracy-float': accuracy, 
            'accuracy': f'{accuracy:.2%}', 
            'error_idxs': np.where(labels != pred_limbs),
        }
        if verbose:
            for k, v in eval_dict.items():
                logger.debug(f'{k}={v}')
        return eval_dict
