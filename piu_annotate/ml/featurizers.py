"""
    Featurize
"""
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from hackerargs import args

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines
from piu_annotate.ml.datapoints import ArrowDataPointConstructor, LimbLabel


def featurize_chart_struct(
    cs: ChartStruct,
    limb_context_col: str = 'Limb annotation',
    include_limb_context: bool = False,
) -> tuple[NDArray, NDArray]:
    """ Featurize all arrows in chart struct,
        treating each arrow individually (multiple arrows per line).
        Arrows ordered by time are taken with `args: ft.context_length`
        on both sides.

        Arguments
        ---------
        cs: ChartStruct
        limb_context_col
            Column name to use for featurizing arrows with limb annotation
        include_limb_context
            Whether or not to use limb annotation for featurization

        Returns
        -------
        featurized prediction inputs: N x M
            If single featurized arrow has D dimensions,
            M = (2*context_len + 1)*D
        labels: N
    """
    cs.annotate_time_since_downpress()
    cs.annotate_line_repeats_previous()
    cs.annotate_line_repeats_next()
    df = cs.df

    constructor = ArrowDataPointConstructor(cs.singles_or_doubles())

    # Build points and labels
    all_points_nolimb = []
    all_points_withlimb = []
    all_labels = []
    pred_coords = cs.get_prediction_coordinates()
    for pred_coord in pred_coords:
        row = cs.df.iloc[pred_coord.row_idx]
        arrow_pos = pred_coord.arrow_pos

        point = constructor.build(row, arrow_pos)
        point_wlimb = constructor.build(
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

    pt_nolimb_array = [point.to_array() for point in all_points_nolimb]
    pt_withlimb_array = [point.to_array() for point in all_points_withlimb]
    np_labels = np.stack([label.to_array() for label in all_labels])

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
        if not include_limb_context:
            window = np.concatenate(padded_nolimb[i - context_len : i + context_len + 1])
        elif include_limb_context:
            window = np.concatenate([
                np.concatenate(padded_withlimb[i - context_len : i]),
                padded_nolimb[i],
                np.concatenate(padded_withlimb[i + 1 : i + context_len])
            ])
        ft_inps.append(window)

    return np.array(ft_inps), np_labels

