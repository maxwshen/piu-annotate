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
from piu_annotate.ml.datapoints import ArrowDataPointConstructor, LimbLabel, ArrowDataPoint, ArrowDataPointWithLimbContext


class FeaturizedChartStruct:
    def __init__(self, cs: ChartStruct):
        """ Stores featurized ChartStruct as:
            - list of ArrowDataPoint
            - list of ArrowDataPointWithLimbContext
            - list of LimbLabels
        """
        self.cs = cs
        cs.annotate_time_since_downpress()
        cs.annotate_line_repeats_previous()
        cs.annotate_line_repeats_next()
        self.pred_coords: list[PredictionCoordinate] = self.cs.get_prediction_coordinates()

        self.constructor = ArrowDataPointConstructor(cs.singles_or_doubles())

        points_nolimb, points_withlimb, labels = self.build()
        self.points_nolimb: list[ArrowDataPoint] = points_nolimb
        self.points_withlimb: list[ArrowDataPointWithLimbContext] = points_withlimb
        self.labels: list[LimbLabel] = labels

    def build(
        self, 
        limb_context_col: str = 'Limb annotation'
    ) -> tuple[list[ArrowDataPoint], list[ArrowDataPointWithLimbContext], list[LimbLabel]]:
        """
        """
        all_points_nolimb = []
        all_points_withlimb = []
        all_labels = []
        for pred_coord in self.pred_coords:
            row = self.cs.df.iloc[pred_coord.row_idx]
            arrow_pos = pred_coord.arrow_pos

            point = self.constructor.build(row, arrow_pos)
            point_wlimb = self.constructor.build(
                row, 
                arrow_pos, 
                limb_context = True,
                limb_context_col = limb_context_col,
            )
            label = LimbLabel.from_limb_annot(
                row['Limb annotation'][pred_coord.limb_idx],
            )

            all_points_nolimb.append(point)
            all_points_withlimb.append(point_wlimb)
            all_labels.append(label)
        return all_points_nolimb, all_points_withlimb, all_labels

    """
        Update
    """
    def update_with_pred_limb_probs(
        self, 
        limb_probs: NDArray,
        mask: NDArray,
    ) -> None:
        """ Update ArrowDataPointWithLimbContext with predicted limb probabilities
            at indices where mask is True.
        """
        assert len(limb_probs) == len(self.points_withlimb) == len(mask)
        for idx in range(len(limb_probs)):
            if mask[idx]:
                self.points_withlimb[idx].limb_annot = limb_probs[idx]
        return

    """
        Get np arrays
    """
    def labels_to_array(self) -> NDArray:
        return np.stack([label.to_array() for label in self.labels])

    def get_prediction_input_with_context(
        self, 
        use_limb_features: bool = False,
    ) -> NDArray:
        """ For N arrows with D feature dims, constructs prediction input
            for each arrow including context arrows on both sides.
            
            If not using limb_context, returns shape N x [(2*context_len + 1)*D]
            If using limb_context, includes limb features (observed for training,
            predicted at test time) in context arrows.
        """
        pt_nolimb_array = [point.to_array() for point in self.points_nolimb]
        pt_withlimb_array = [point.to_array() for point in self.points_withlimb]

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
            if not use_limb_features:
                window = np.concatenate(padded_nolimb[i - context_len : i + context_len + 1])
            elif use_limb_features:
                window = np.concatenate([
                    np.concatenate(padded_withlimb[i - context_len : i]),
                    padded_nolimb[i],
                    np.concatenate(padded_withlimb[i + 1 : i + context_len])
                ])
            ft_inps.append(window)
        return np.array(ft_inps)

