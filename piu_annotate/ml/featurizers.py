"""
    Featurize
"""
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from hackerargs import args
import functools
import os

from piu_annotate.formats.chart import ChartStruct, ArrowCoordinate
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
        self.context_with_hold_release = args.setdefault('ft.context_with_hold_release', False)
        self.prev_limb_feature_context_len = args.setdefault('ft.prev_limb_context_len', 8)

        self.singles_or_doubles = cs.singles_or_doubles()

        self.cs.annotate_time_since_downpress()
        self.cs.annotate_line_repeats_previous()
        self.cs.annotate_line_repeats_next()
        self.arrow_coords = self.cs.get_arrow_coordinates()
        self.pred_coords = self.cs.get_prediction_coordinates()

        self.row_idx_to_prevs = self.cs.get_previous_used_pred_coord()

        # Featurize all arrows: 1, 2, 3
        self.arrowdatapoints_with_3 = self.get_arrowdatapoints(with_hold_release = True)
        self.arrowdatapoints_without_3 = self.get_arrowdatapoints(with_hold_release = False)
        if self.context_with_hold_release:
            self.arrowdatapoints_ft = self.arrowdatapoints_with_3
        else:
            self.arrowdatapoints_ft = self.arrowdatapoints_without_3

        self.pt_array = [pt.to_array_categorical() for pt in self.arrowdatapoints_ft]
        self.pt_feature_names = self.arrowdatapoints_ft[0].get_feature_names_categorical()

        self.chart_metadata_features = self.get_chart_metadata_features()

        self.pc_idx_to_prev = self.cs.get_previous_used_pred_coord_for_arrow()
        # shift by +1, and replace None with 0
        shifted = [x if x is not None else -1 for x in self.pc_idx_to_prev.values()]
        self.prev_pc_idx_shifted = np.array(shifted) + 1

    """
        Build
    """    
    def get_arrowdatapoints(self, with_hold_release: bool) -> list[ArrowDataPoint]:
        """ Featurize chart into ArrowDataPoints

            Options
            -------
            context_with_hold_release: If True, then include all 1/2/3 arrows
                Otherwise, only use prediction coordinates (1/2)
        """
        all_arrowdatapoints = []
        ac_to_time_last_arrow_use = self.cs.get_time_since_last_same_arrow_use()

        if with_hold_release:
            coords = self.arrow_coords
        else:
            coords = self.pred_coords

        for idx, arrow_coord in enumerate(coords):
            row = self.cs.df.iloc[arrow_coord.row_idx]
            line = row['Line with active holds'].replace('`', '')
            line_is_bracketable = notelines.line_is_bracketable(line)

            prior_line_only_releases_hold_on_this_arrow = False
            row_idx = arrow_coord.row_idx
            if row_idx > 0:
                prev_line = self.cs.df.iloc[row_idx - 1]['Line'].replace('`', '')
                if prev_line[arrow_coord.arrow_pos] == '3':
                    if prev_line.count('0') in [4, 9]:
                        prior_line_only_releases_hold_on_this_arrow = True

            next_line_only_releases_hold_on_this_arrow = False
            if row_idx + 1 < len(self.cs.df):
                prev_line = self.cs.df.iloc[row_idx + 1]['Line'].replace('`', '')
                if prev_line[arrow_coord.arrow_pos] == '3':
                    if prev_line.count('0') in [4, 9]:
                        next_line_only_releases_hold_on_this_arrow = True

            arrow_pos = arrow_coord.arrow_pos
            point = ArrowDataPoint(
                arrow_pos = arrow_pos,
                arrow_symbol = line[arrow_pos],
                line_with_active_holds = line,
                active_hold_idxs = [i for i, s in enumerate(line) if s in list('34')],
                prior_line_only_releases_hold_on_this_arrow = prior_line_only_releases_hold_on_this_arrow,
                time_since_last_same_arrow_use = ac_to_time_last_arrow_use[arrow_coord],
                time_since_prev_downpress = row['__time since prev downpress'],
                num_downpress_in_line = line.count('1') + line.count('2'),
                line_is_bracketable = line_is_bracketable,
                line_repeats_previous_downpress_line = row['__line repeats previous downpress line'],
                line_repeats_next_downpress_line = row['__line repeats next downpress line'],
                singles_or_doubles = self.singles_or_doubles,
                prev_pc_idxs = self.row_idx_to_prevs[arrow_coord.row_idx],
                next_line_only_releases_hold_on_this_arrow = next_line_only_releases_hold_on_this_arrow,
            )
            all_arrowdatapoints.append(point)
        return all_arrowdatapoints

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

    def get_chart_metadata_features(self) -> NDArray:
        """ Builds NDArray of features for a chart, which are constant
            for all arrowdatapoints in the same chart.
        """
        level = self.cs.get_chart_level()
        return np.array([level])

    """
        Featurize
    """
    def get_padded_array(self) -> NDArray:
        pt_array = self.pt_array
        context_len = self.context_len
        empty_pt = np.ones(len(pt_array[0])) * -1
        empty_pt.fill(np.nan)
        return np.array([empty_pt]*context_len + pt_array + [empty_pt]*context_len)

    @functools.lru_cache
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
        view = np.reshape(view, (N, D*c2_plus_1), order = 'F')

        # append chart-level features
        cmf = np.repeat(self.chart_metadata_features.reshape(-1, 1), N, axis = 0)
        # shaped into (N, d)
        all_x = np.concatenate((view, cmf), axis = 1)

        if len(all_x) > len(self.pred_coords):
            # Using context_with_hold_release; subset to PredictionCoordinates
            pred_coord_idxs = [idx for idx, ac in enumerate(self.arrow_coords)
                            if ac in self.pred_coords]
            all_pred_coord_fts = all_x[pred_coord_idxs]
        else:
            all_pred_coord_fts = all_x
        return all_pred_coord_fts

    def get_arrow_context_feature_names(self) -> list[str]:
        """ Must be aligned with featurize_arrows_with_context """
        fnames = self.pt_feature_names
        all_feature_names = []
        for context_pos in range(-self.context_len, self.context_len + 1):
            all_feature_names += [f'{fn}-{context_pos}' for fn in fnames]
        all_feature_names += ['chart_level']
        assert len(all_feature_names) == self.featurize_arrows_with_context().shape[-1]
        return all_feature_names

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
        padded_prev_limb_probs = np.concatenate([
            [-1] * self.prev_limb_feature_context_len,
            prev_limb_probs,
            [-1] * self.prev_limb_feature_context_len,
        ])
        prev_limb_view = np.lib.stride_tricks.sliding_window_view(
            padded_prev_limb_probs, 
            (2 * self.prev_limb_feature_context_len + 1), 
            axis = 0
        )
        features = np.concatenate([arrow_view, limb_view, prev_limb_view], axis = 1)
        return features

    def get_arrowlimb_context_feature_names(self) -> list[str]:
        """ Must be aligned with featurize_arrowlimbs_with_context """
        fnames = self.get_arrow_context_feature_names()

        # add limb feature for nearby arrows
        fnames += [f'limb_nearby_arrow_{idx}' for idx in range(self.context_len * 2)]

        # add limb feature for previous limb used for nearby arrows
        fnames += [f'prev_limb_nearby_arrow_{idx}'
                   for idx in range(2 * self.prev_limb_feature_context_len + 1)]

        n_pred_coords = self.featurize_arrows_with_context().shape[0]
        __fake_limb_probs = np.ones(n_pred_coords)
        assert len(fnames) == self.featurize_arrowlimbs_with_context(__fake_limb_probs).shape[-1]
        return fnames

    def downpress_idx_to_time(self, dp_idx: int) -> float:
        row_idx = self.pred_coords[dp_idx].row_idx
        return float(self.cs.df.iloc[row_idx]['Time'])

    """
        Evaluation
    """
    def evaluate(self, pred_limbs: NDArray, verbose: bool = False) -> dict[str, any]:
        """ Evaluate vs 'Limb annotation' column """
        labels = self.get_labels_from_limb_col('Limb annotation')
        accuracy = np.sum(labels == pred_limbs) / len(labels)
        error_idxs = np.where(labels != pred_limbs)[0]
        error_times = [self.downpress_idx_to_time(i) for i in error_idxs]
        eval_dict = {
            'accuracy-float': accuracy, 
            'accuracy': f'{accuracy:.2%}', 
            'error_idxs': error_idxs,
            'error_times': [f'{t:.2f}' for t in error_times],
        }
        if verbose:
            for k, v in eval_dict.items():
                logger.debug(f'{k}={v}')
        return eval_dict
