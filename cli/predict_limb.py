"""
    Featurize
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

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers


def apply_index(array, idxs):
    return np.array([a[i] for a, i in zip(array, idxs)])


def group_list_consecutive(data: list[int]) -> list[list[int]]:
    """ Groups a flat list into a list of lists with all-consecutive numbers """
    ranges = []
    for k, g in itertools.groupby(enumerate(data), lambda x:x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1] + 1))
    return ranges


def get_ranges(iterable, value):
    """Get ranges of indices where the value matches."""
    ranges = []
    for key, group in itertools.groupby(enumerate(iterable), key=lambda x: x[1] == value):
        if key:  # If the value matches
            group = list(group)
            ranges.append((group[0][0], group[-1][0] + 1))
    return ranges


def score(
    cs: ChartStruct,
    limb_col: str,
    model_withlimb: GradientBoostingClassifier,
    model_label_matches_next: GradientBoostingClassifier,
    model_label_matches_prev: GradientBoostingClassifier,
) -> float:
    """ Scores limb annotations in `limb_col` in `cs`, as sum log probability under
        limb prediction model using limb,
        and model that predicts if label matches next label
    """
    fcs = featurizers.ChartStructFeaturizer(cs)

    points = fcs.featurize_arrows_with_context()
    curr_labels = fcs.get_labels_from_limb_col(limb_col)

    points_withlimb = fcs.featurize_arrowlimbs_with_context(curr_labels)
    log_probs_withlimb = model_withlimb.predict_log_proba(points_withlimb)
    log_prob_labels_withlimb = sum(apply_index(log_probs_withlimb, curr_labels))

    log_prob_matches = model_label_matches_next.predict_log_proba(points)
    matches_next = np.concatenate([curr_labels[:-1] == curr_labels[1:], [False]]).astype(int)
    log_prob_labels_matches = sum(apply_index(log_prob_matches, matches_next))

    log_prob_matches_prev = model_label_matches_prev.predict_log_proba(points)
    matches_prev = np.concatenate([[False], curr_labels[1:] == curr_labels[:-1], ]).astype(int)
    log_prob_labels_matches_prev = sum(apply_index(log_prob_matches_prev, matches_prev))

    return log_prob_labels_withlimb + np.mean([log_prob_labels_matches, log_prob_labels_matches_prev])


def score_predictions(
    cs: ChartStruct,
    predicted_limbs: NDArray,
    model_withlimb: GradientBoostingClassifier,
    model_label_matches_next: GradientBoostingClassifier,
    model_label_matches_prev: GradientBoostingClassifier,
) -> float:
    int_to_limb = {0: 'l', 1: 'r'}
    pred_limb_strs = [int_to_limb[i] for i in predicted_limbs]
    
    cs.add_limb_annotations(
        cs.get_prediction_coordinates(), 
        pred_limb_strs, 
        '__pred_limb_for_scoring'
    )
    return score(cs, '__pred_limb_for_scoring', model_withlimb, model_label_matches_next, model_label_matches_prev)


def propose_label_flips(
    cs: ChartStruct,
    limb_col: str,
    model_withlimb: GradientBoostingClassifier,
    model_label_matches_next: GradientBoostingClassifier,
    model_label_matches_prev: GradientBoostingClassifier,
) -> NDArray:
    """ Propose label flips that improve score """
    fcs = featurizers.ChartStructFeaturizer(cs)

    points = fcs.featurize_arrows_with_context()
    curr_labels = fcs.get_labels_from_limb_col(limb_col)

    points_withlimb = fcs.featurize_arrowlimbs_with_context(curr_labels)
    log_probs_withlimb = model_withlimb.predict_log_proba(points_withlimb)

    log_prob_matches_next = model_label_matches_next.predict_log_proba(points)
    matches_next = np.concatenate([curr_labels[:-1] == curr_labels[1:], [False]]).astype(int)

    log_prob_matches_prev = model_label_matches_prev.predict_log_proba(points)
    matches_prev = np.concatenate([[False], curr_labels[1:] == curr_labels[:-1], ]).astype(int)

    def calc_improve_parallel():
        curr_score = apply_index(log_probs_withlimb, curr_labels) + np.mean([
            apply_index(log_prob_matches_next, matches_next),
            apply_index(log_prob_matches_prev, matches_prev)
        ], axis = 0)
        flip_score = apply_index(log_probs_withlimb, 1 - curr_labels) + np.mean([
            apply_index(log_prob_matches_next, 1 - matches_next),
            apply_index(log_prob_matches_prev, 1 - matches_prev)
        ], axis = 0)
        return flip_score - curr_score

    improves = calc_improve_parallel()
    # improves has a lot of consecutive pairs
    cand_idxs = list(np.where(improves > 0)[0])

    # find groups of consecutive pairs, reduce by taking single best from each group
    logger.debug(f'{cand_idxs}')
    groups = group_list_consecutive(cand_idxs)
    reduced_idxs = []
    for start, end in groups:
        best_idx = start + np.argmax(improves[start:end])
        reduced_idxs.append(best_idx)
    logger.info(f'Found {len(reduced_idxs)} labels to flip')

    new_labels = curr_labels.copy()
    new_labels[reduced_idxs] = 1 - new_labels[reduced_idxs]
    return new_labels


def propose_jacks(
    cs: ChartStruct,
    pred_limbs: NDArray,
    model_withlimb: GradientBoostingClassifier,
    model_label_matches_next: GradientBoostingClassifier,
    model_label_matches_prev: GradientBoostingClassifier,
    only_consider_nonuniform_jacks: bool = True,
) -> NDArray:
    """ Use limb matching model to predict jacks, and score best limb for jacks.
    """
    fcs = featurizers.ChartStructFeaturizer(cs)
    points = fcs.featurize_arrows_with_context()
    pred_label_matches = model_label_matches_next.predict(points)

    ranges = get_ranges(pred_label_matches, 1)

    orig_pred_limbs = pred_limbs.copy()
    new_pred_limbs = pred_limbs.copy()

    for start, end in ranges:
        exp_match_end = end + 1
        pred_subset = pred_limbs[start : exp_match_end]
        if len(set(pred_subset)) == 1:
            if only_consider_nonuniform_jacks:
                continue
        logger.debug(f'{pred_subset}, {start}, {exp_match_end}')

        all_left = orig_pred_limbs.copy()
        all_left[start : exp_match_end] = 0
        left_score = score_predictions(cs, all_left, model_withlimb, model_label_matches_next, model_label_matches_prev)

        all_right = orig_pred_limbs.copy()
        all_right[start : exp_match_end] = 1
        right_score = score_predictions(cs, all_right, model_withlimb, model_label_matches_next, model_label_matches_prev)

        if left_score > right_score:
            new_pred_limbs[start : exp_match_end] = 0
        else:
            new_pred_limbs[start : exp_match_end] = 1

    return new_pred_limbs


def main():
    cs = ChartStruct.from_file(args['chart_struct_csv'])
    pred_coords = cs.get_prediction_coordinates()

    # load models
    with open(args['model_pkl'], 'rb') as f:
        model_nolimb: GradientBoostingClassifier = pickle.load(f)

    with open(args['model_label_matches_next_pkl'], 'rb') as f:
        model_label_matches_next: GradientBoostingClassifier = pickle.load(f)

    with open(args['model_label_matches_prev_pkl'], 'rb') as f:
        model_label_matches_prev: GradientBoostingClassifier = pickle.load(f)

    with open(args['model_limbcontext_pkl'], 'rb') as f:
        model_withlimb: GradientBoostingClassifier = pickle.load(f)

    # score true labels
    print(score(cs, 'Limb annotation', model_withlimb, model_label_matches_next, model_label_matches_prev))

    """
        First prediction pass: Without limb context
    """
    # featurize
    fcs = featurizers.ChartStructFeaturizer(cs)
    points = fcs.featurize_arrows_with_context()
    logger.info(f'Found dataset shape {points.shape}')

    pred_limbs = model_nolimb.predict(points)
    int_to_limb = {0: 'l', 1: 'r'}
    pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
    pred_limb_probs = model_nolimb.predict_proba(points)[:, -1]

    # compare
    fcs.evaluate(pred_limbs)

    # try label matches next
    labels_matches = fcs.get_label_matches_next('Limb annotation')
    pred_label_matches = model_label_matches_next.predict(points)
    accuracy = np.sum(labels_matches == pred_label_matches) / len(labels_matches)
    logger.info(f'{accuracy=}')

    # try label matches prev
    labels_matches = fcs.get_label_matches_prev('Limb annotation')
    pred_label_matches = model_label_matches_prev.predict(points)
    accuracy = np.sum(labels_matches == pred_label_matches) / len(labels_matches)
    logger.info(f'{accuracy=}')

    # score
    print(score_predictions(cs, pred_limbs, model_withlimb, model_label_matches_next, model_label_matches_prev))

    """
        Second prediction pass: Use predicted limb context
    """
    num_iters = 5
    for __i in range(num_iters):
        weight = (__i + 1) / (num_iters + 1)
        adj_limb_probs = weight * pred_limb_probs + (1 - weight) * 0.5 * np.ones(pred_limb_probs.shape)

        points_withlimb = fcs.featurize_arrowlimbs_with_context(adj_limb_probs)

        pred_limbs = model_withlimb.predict(points_withlimb)
        pred_limb_probs = model_withlimb.predict_proba(points_withlimb)[:, -1]

        # compare
        fcs.evaluate(pred_limbs)

        # score
        print(score_predictions(cs, pred_limbs, model_withlimb, model_label_matches_next, model_label_matches_prev))

    pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
    cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

    pred_limbs = propose_label_flips(cs, '__pred limb final', model_withlimb, model_label_matches_next, model_label_matches_prev)

    fcs.evaluate(pred_limbs)
    print(score_predictions(cs, pred_limbs, model_withlimb, model_label_matches_next, model_label_matches_prev))
    pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
    cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

    # try jacks
    pred_limbs = propose_jacks(cs, pred_limbs, model_withlimb, model_label_matches_next, model_label_matches_prev)

    fcs.evaluate(pred_limbs)
    print(score_predictions(cs, pred_limbs, model_withlimb, model_label_matches_next, model_label_matches_prev))
    pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
    cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

    cs.df.to_csv('temp/conflict-s11.csv')
    import code; code.interact(local=dict(globals(), **locals()))
    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chart_struct_csv', 
        default = '/home/maxwshen/piu-annotate/artifacts/chartstructs/piucenter-manual-090624/Conflict_-_Siromaru___Cranky_S11_arcade.csv',
    )
    parser.add_argument(
        '--model_pkl', 
        # default = '/home/maxwshen/piu-annotate/artifacts/models/091024/gbc-singles.pkl',
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/line-repeat/temp-singles.pkl'
    )
    parser.add_argument(
        '--model_label_matches_next_pkl', 
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/label-matches-next/temp-singles.pkl'
    )
    parser.add_argument(
        '--model_label_matches_prev_pkl', 
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/label-matches-prev/temp-singles.pkl'
    )
    parser.add_argument(
        '--model_limbcontext_pkl', 
        # default = '/home/maxwshen/piu-annotate/artifacts/models/091024/gbc-singles-withlimbcontext.pkl',
        default = '/home/maxwshen/piu-annotate/artifacts/models/091024/line-repeat/temp-withlimb-singles.pkl'
    )
    args.parse_args(parser)
    main()