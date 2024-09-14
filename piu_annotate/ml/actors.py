"""
    Actor
"""
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray
import itertools
import numpy as np
from operator import itemgetter
import functools

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.models import ModelSuite


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


def get_matches_next(array: NDArray) -> NDArray:
    return np.concatenate([array[:-1] == array[1:], [False]]).astype(int)


def get_matches_prev(array: NDArray) -> NDArray:
    return np.concatenate([[False], array[1:] == array[:-1]]).astype(int)


class Actor:
    def __init__(self, cs: ChartStruct, model_suite: ModelSuite):
        """ Actor uses a suite of ML models to optimize predicted limb
            annotations for a given ChartStruct.
        """
        self.cs = cs
        self.models = model_suite
        self.pred_coords = self.cs.get_prediction_coordinates()
        self.fcs = featurizers.ChartStructFeaturizer(self.cs)
    
    def score(self, pred_limbs: NDArray) -> float:
        log_probs_withlimb = self.predict_arrowlimbs(pred_limbs, logp = True)
        log_prob_labels_withlimb = sum(apply_index(log_probs_withlimb, pred_limbs))

        log_prob_matches = self.predict_matchnext(logp = True)
        matches_next = get_matches_next(pred_limbs)
        log_prob_labels_matches = sum(apply_index(log_prob_matches, matches_next))

        log_prob_matches_prev = self.predict_matchprev(logp = True)
        matches_prev = get_matches_prev(pred_limbs)
        log_prob_labels_matches_prev = sum(apply_index(log_prob_matches_prev, matches_prev))

        return sum([
            log_prob_labels_withlimb,
            np.mean([log_prob_labels_matches, log_prob_labels_matches_prev])
        ])
    
    def label_flip_improvement(self, pred_limbs: NDArray) -> NDArray:
        """ Returns score improvement vector """
        log_probs_withlimb = self.predict_arrowlimbs(pred_limbs, logp = True)
        log_prob_matches_next = self.predict_matchnext(logp = True)
        log_prob_matches_prev = self.predict_matchprev(logp = True)
        matches_next = get_matches_next(pred_limbs)
        matches_prev = get_matches_prev(pred_limbs)

        curr_score = apply_index(log_probs_withlimb, pred_limbs) + np.mean([
            apply_index(log_prob_matches_next, matches_next),
            apply_index(log_prob_matches_prev, matches_prev)
        ], axis = 0)
        flip_score = apply_index(log_probs_withlimb, 1 - pred_limbs) + np.mean([
            apply_index(log_prob_matches_next, 1 - matches_next),
            apply_index(log_prob_matches_prev, 1 - matches_prev)
        ], axis = 0)
        return flip_score - curr_score

    """
        Limb prediction handling
    """
    def iterative_refine(self, n_iter: int = 5) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        pred_limb_p = self.models.model_arrows_to_limb.predict_proba(points)[:, -1]

        for __i in range(n_iter):
            weight = (__i + 1) / (n_iter + 1)
            adj_limb_probs = weight * pred_limb_p + (1 - weight) * 0.5 * np.ones(pred_limb_p.shape)

            points_wlimb = self.fcs.featurize_arrowlimbs_with_context(adj_limb_probs)
            pred_limb_p = self.models.model_arrowlimbs_to_limb.predict_proba(points_wlimb)[:, -1]
        return self.predict_arrowlimbs(pred_limb_p)

    def flip_labels_by_score(self, pred_limbs: NDArray) -> NDArray:
        """ Flips individual limbs by improvement score.
            Only flips one label in contiguous groups of candidate improvement idxs.
        """
        improves = self.label_flip_improvement(pred_limbs)
        cand_idxs = list(np.where(improves > 0)[0])

        logger.debug(f'{cand_idxs}')
        groups = group_list_consecutive(cand_idxs)
        reduced_idxs = []
        for start, end in groups:
            best_idx = start + np.argmax(improves[start:end])
            reduced_idxs.append(best_idx)
        logger.debug(f'Found {len(reduced_idxs)} labels to flip')

        new_labels = pred_limbs.copy()
        new_labels[reduced_idxs] = 1 - new_labels[reduced_idxs]
        return new_labels

    def flip_jack_sections(
        self, 
        pred_limbs: NDArray,
        only_consider_nonuniform_jacks: bool = True,
    ) -> NDArray:
        """ Use parity prediction to find jack sections, and put best limb
        """
        logger.info(f'Flipping jack sections ...')
        pred_matches_next = self.predict_matchnext()
        ranges = get_ranges(pred_matches_next, 1)

        orig_pred_limbs = pred_limbs.copy()
        new_pred_limbs = pred_limbs.copy()
        for start, end in tqdm(ranges):
            exp_match_end = end + 1
            pred_subset = pred_limbs[start : exp_match_end]
            if len(set(pred_subset)) == 1:
                if only_consider_nonuniform_jacks:
                    continue
            # logger.debug(f'{pred_subset}, {start}, {exp_match_end}')

            all_left = orig_pred_limbs.copy()
            all_left[start : exp_match_end] = 0
            left_score = self.score(all_left)

            all_right = orig_pred_limbs.copy()
            all_right[start : exp_match_end] = 1
            right_score = self.score(all_right)

            if left_score > right_score:
                new_pred_limbs[start : exp_match_end] = 0
            else:
                new_pred_limbs[start : exp_match_end] = 1

        return new_pred_limbs

    def beam_search(
        self, 
        pred_limbs: NDArray, 
        width: int, 
        n_iter: int
    ) -> list[NDArray]:
        """ """
        def get_top_flips(pred_limbs: NDArray) -> NDArray:
            imp = self.label_flip_improvement(pred_limbs)
            return np.where(imp > sorted(imp)[-width])
        
        def flip(pred_limbs: NDArray, idx: int) -> NDArray:
            new = pred_limbs.copy()
            new[idx] = 1 - new[idx]
            return new

        def beam(pred_limbs: list[NDArray]) -> list[NDArray]:
            top_flip_idxs = [get_top_flips(pl) for pl in pred_limbs]
            return [
                flip(pl, idx) for pl, tfi in zip(pred_limbs, top_flip_idxs)
                for idx in tfi
            ]

        inp = [pred_limbs]
        all_pred_limbs = inp
        for i in range(n_iter):
            inp = beam(inp)
            all_pred_limbs += inp

        scores = [self.score(pl) for pl in all_pred_limbs]
        best = max(scores)
        return all_pred_limbs[scores.index(best)]

    """
        Model predictions
    """
    @functools.cache
    def predict_arrow(self, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        if logp:
            return self.models.model_arrows_to_limb.predict_log_proba(points)
        else:
            return self.models.model_arrows_to_limb.predict(points)

    def predict_arrowlimbs(self, limb_array: NDArray, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrowlimbs_with_context(limb_array)
        if logp:
            return self.models.model_arrowlimbs_to_limb.predict_log_proba(points)
        else:
            return self.models.model_arrowlimbs_to_limb.predict(points)

    @functools.cache
    def predict_matchnext(self, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        if logp:
            return self.models.model_arrows_to_matchnext.predict_log_proba(points)
        else:
            return self.models.model_arrows_to_matchnext.predict(points)

    @functools.cache
    def predict_matchprev(self, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        if logp:
            return self.models.model_arrows_to_matchprev.predict_log_proba(points)
        else:
            return self.models.model_arrows_to_matchprev.predict(points)
