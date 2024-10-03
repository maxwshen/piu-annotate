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
import math
import functools
from collections import defaultdict

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.datapoints import ArrowDataPoint
from piu_annotate.formats import notelines
from piu_annotate.ml import run_reasoning


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


class Tactician:
    def __init__(self, cs: ChartStruct, model_suite: ModelSuite, verbose: bool = False):
        """ Tactician uses a suite of ML models and a set of tactics to
            optimize predicted limb annotations for a given ChartStruct.
        """
        self.cs = cs
        self.models = model_suite
        self.verbose = verbose
        self.pred_coords = self.cs.get_prediction_coordinates()
        self.fcs = featurizers.ChartStructFeaturizer(self.cs)

        self.row_idx_to_pcs: dict[int, int] = defaultdict(list)
        for pc_idx, pc in enumerate(self.pred_coords):
            self.row_idx_to_pcs[pc.row_idx].append(pc_idx)
    
    def score(self, pred_limbs: NDArray, debug: bool = False) -> float:
        log_probs_withlimb = self.predict_arrowlimbs(pred_limbs, logp = True)
        log_prob_labels_withlimb = sum(apply_index(log_probs_withlimb, pred_limbs))

        log_prob_matches = self.predict_matchnext(logp = True)
        matches_next = get_matches_next(pred_limbs)
        log_prob_labels_matches = sum(apply_index(log_prob_matches, matches_next))

        log_prob_matches_prev = self.predict_matchprev(logp = True)
        matches_prev = get_matches_prev(pred_limbs)
        log_prob_labels_matches_prev = sum(apply_index(log_prob_matches_prev, matches_prev))

        score_components = [
            log_prob_labels_withlimb,
            np.mean([log_prob_labels_matches, log_prob_labels_matches_prev])
        ]
        if debug:
            logger.debug(score_components)
        return sum(score_components)
    
    def score_limbs_given_limbs(self, pred_limbs: NDArray) -> float:
        log_probs_withlimb = self.predict_arrowlimbs(pred_limbs, logp = True)
        log_prob_labels_withlimb = sum(apply_index(log_probs_withlimb, pred_limbs))
        return log_prob_labels_withlimb

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
    def initial_predict(self) -> NDArray:
        pred_limbs = self.predict_arrow()
        return self.predict_arrowlimbs(pred_limbs)

    def flip_labels_by_score(self, pred_limbs: NDArray) -> NDArray:
        """ Flips individual limbs by improvement score.
            Only flips one label in contiguous groups of candidate improvement idxs.
        """
        improves = self.label_flip_improvement(pred_limbs)
        cand_idxs = list(np.where(improves > 0)[0])

        # logger.debug(f'{cand_idxs}')
        groups = group_list_consecutive(cand_idxs)
        reduced_idxs = []
        for start, end in groups:
            best_idx = start + np.argmax(improves[start:end])
            reduced_idxs.append(best_idx)
        # if len(reduced_idxs) > 0:
            # logger.debug(f'Found {len(reduced_idxs)} labels to flip')

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
        pred_matches_next = self.predict_matchnext()
        ranges = get_ranges(pred_matches_next, 1)

        orig_pred_limbs = pred_limbs.copy()
        new_pred_limbs = pred_limbs.copy()
        for start, end in ranges:
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

    def fix_double_doublestep(self, pred_limbs: NDArray) -> NDArray:
        """ Find and fix sections starting and ending with double step
            where parity prediction strongly prefers alternating instead.

            Options
            -------
            start_threshold_p: Min. predicted probability of flipping limb,
                used to find candidate starts for double double steps
            len_limit: Max length of double doublestep length range to flip
            min_improvement_per_arrow: Minimum score improvement to accept
                a proposed flip, divided by flip length
        """
        start_threshold_p = args.setdefault('tactic.fix_double_doublestep.start_flip_prob', 0.9)
        len_limit = args.setdefault('tactic.fix_double_doublestep.len_limit', 12)
        min_improvement_per_arrow = args.setdefault('tactic.fix_double_doublestep.min_improvement_per_arrow', 2)

        # find low-scoring doublesteps
        start_threshold = np.log(1 - start_threshold_p)
        pred_matches_next = get_matches_next(pred_limbs)
        match_logp = self.predict_matchnext(logp = True)
        applied_logp = apply_index(match_logp, pred_matches_next)
        start_cands = np.where(applied_logp < start_threshold)[0]

        def try_flip(pred_limbs: NDArray, cand_idx: int):
            start = cand_idx + 1

            end_cands = start + np.where(applied_logp[start:start + len_limit] < -0.2)[0]
            best_improvement = 0
            base_score = self.score(pred_limbs)
            best_limbs = pred_limbs
            best_end = None
            for end_idx in end_cands:
                end = end_idx + 1
                pl = pred_limbs.copy()
                pl[start:end] = 1 - pl[start:end]
                score = self.score(pl)

                improvement = (score - base_score) / (end - start)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_limbs = pl
                    best_end = end

            if best_improvement >= min_improvement_per_arrow:
                return best_limbs, (start, best_end)
            return pred_limbs, None

        # logger.debug(f'{start_cands=}')
        n_flips = 0
        flipped_ranges = []
        while(len(start_cands) > 0):
            pred_limbs, found_range = try_flip(pred_limbs, start_cands[0])
            start_cands = start_cands[1:]
            if found_range is not None:
                n_flips += 1
                flipped_ranges.append(found_range)

        if self.verbose:
            logger.debug(f'Flipped {n_flips} sections: {flipped_ranges}')
        return pred_limbs

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

    def detect_impossible_multihit(self, pred_limbs: NDArray):
        """ Find parts of `pred_limbs` implying physically impossible
            limb combo to hit any single line with multiple downpresses;
            primarily fixes brackets.
            Does not consider holds.
        """
        pred_limbs = pred_limbs.copy()        
        n_lines_fixed = 0

        for row_idx, pc_idxs in self.row_idx_to_pcs.items():
            if len(pc_idxs) > 1:
                pcs = [self.pred_coords[i] for i in pc_idxs]
                limbs = pred_limbs[pc_idxs]

                lefts = [pc.arrow_pos for pc, limb in zip(pcs, limbs) if limb == 0]
                rights = [pc.arrow_pos for pc, limb in zip(pcs, limbs) if limb == 1]

                left_ok = notelines.one_foot_multihit_possible(lefts)
                right_ok = notelines.one_foot_multihit_possible(rights)

                if not (left_ok and right_ok):
                    limb_combos = notelines.multihit_to_valid_limbs([pc.arrow_pos for pc in pcs])

                    if len(limb_combos) == 0:
                        continue

                    n_lines_fixed += 1
                    score_to_limbs = dict()
                    for limb_combo in limb_combos:
                        pl = pred_limbs.copy()
                        for limb, pc_idx in zip(limb_combo, pc_idxs):
                            pl[pc_idx] = limb
                        score_to_limbs[self.score(pl)] = limb_combo
                    best_combo = score_to_limbs[max(score_to_limbs)]

                    for limb, pc_idx in zip(best_combo, pc_idxs):
                        pred_limbs[pc_idx] = limb
        if self.verbose and n_lines_fixed > 0:
            logger.debug(f'Fixed {n_lines_fixed} impossible multihit lines')
        return pred_limbs

    def detect_impossible_lines_with_holds(self, pred_limbs: NDArray):
        """ Find parts of `pred_limbs` implying physically impossible
            limb combo to hit any single line, when considering active holds too.
            Attempts to adjust downpresses at the line to make it possible;
            does not adjust prior holds.

            There may still be impossible lines considering active holds after this,
            if an incorrect limb is used for prior holds.
        """
        pred_limbs = pred_limbs.copy()
        adps = self.fcs.arrowdatapoints_without_3

        # get row idxs with active holds
        pc_idx_active_holds = [i for i in range(len(adps)) if adps[i].active_hold_idxs]
        rows_with_active_holds = sorted(set(self.pred_coords[i].row_idx for i in pc_idx_active_holds))

        n_lines_fixed = 0
        edited_pc_idxs = []
        for row_idx in rows_with_active_holds:
            row_pc_idxs = self.row_idx_to_pcs[row_idx]

            # get pc idxs of active holds
            active_hold_panel_pos = adps[row_pc_idxs[0]].active_hold_idxs
            all_prev_pc_idxs = adps[row_pc_idxs[0]].prev_pc_idxs
            hold_pc_idxs = [all_prev_pc_idxs[p] for p in active_hold_panel_pos]

            hold_pcs = [self.pred_coords[i] for i in hold_pc_idxs]
            hold_limbs = pred_limbs[hold_pc_idxs]
            hold_left = [pc.arrow_pos for pc, limb in zip(hold_pcs, hold_limbs) if limb == 0]
            hold_right = [pc.arrow_pos for pc, limb in zip(hold_pcs, hold_limbs) if limb == 1]

            curr_pcs = [self.pred_coords[i] for i in row_pc_idxs]
            curr_limbs = pred_limbs[row_pc_idxs]
            curr_left = [pc.arrow_pos for pc, limb in zip(curr_pcs, curr_limbs) if limb == 0]
            curr_right = [pc.arrow_pos for pc, limb in zip(curr_pcs, curr_limbs) if limb == 1]

            all_left = hold_left + curr_left
            all_right = hold_right + curr_right

            left_ok = notelines.one_foot_multihit_possible(all_left)
            right_ok = notelines.one_foot_multihit_possible(all_right)
            if not (left_ok and right_ok):
                all_arrows = sorted(all_left + all_right)
                limb_combos = notelines.multihit_to_valid_limbs(all_arrows)

                pos_to_pc_idxs = {pc.arrow_pos: self.pred_coords.index(pc)
                                  for pc in hold_pcs + curr_pcs}
                all_pc_idxs = [pos_to_pc_idxs[pos] for pos in all_arrows]

                def limbs_match_holds(limb_combo: tuple[int]) -> bool:
                    for pos, limb in zip(all_arrows, limb_combo):
                        if pos in hold_left and limb != 0:
                            return False
                        if pos in hold_right and limb != 1:
                            return False
                    return True

                # filter limb combos to those consistent with previous holds
                valid_lcs = [lc for lc in limb_combos if limbs_match_holds(lc)]
                if len(valid_lcs) == 0:
                    logger.warning(f'Found impossible line with holds with no valid alternate')
                    continue
                
                n_lines_fixed += 1
                score_to_limbs = dict()
                for limb_combo in valid_lcs:
                    pl = pred_limbs.copy()
                    for limb, pc_idx in zip(limb_combo, all_pc_idxs):
                        pl[pc_idx] = limb
                    score_to_limbs[self.score(pl)] = limb_combo
                best_combo = score_to_limbs[max(score_to_limbs)]

                for limb, pc_idx in zip(best_combo, all_pc_idxs):
                    pred_limbs[pc_idx] = limb
                edited_pc_idxs += row_pc_idxs

        if self.verbose and n_lines_fixed > 0:
            logger.debug(f'Fixed {n_lines_fixed} impossible lines with holds: {edited_pc_idxs=}')
        return pred_limbs

    def remove_doublesteps_in_long_nojack_runs(self, pred_limbs: NDArray) -> NDArray:
        """ Remove doublesteps in long runs with no jacks, with constant `time_since`,
            and each line has only one downpress.

            Options in args
            min_run_length: Minimum run length to consider
            min_time_since: Minimum time_since to consider -- do not consider
                runs with very low time_since as they may be staggered brackets
            max_time_since: Max time_since to consider - do not consider notes
                very far apart
            max_frac_doublestep: Skip removing doublesteps for sections with many
                predicted doublesteps - these may be staggered brackets
        """
        max_frac_doublestep = args.setdefault('tactic.remove_doublesteps.max_frac_doublestep', 0.40)
        # find long runs with nojacks, using arrowdatapoints
        adps = self.fcs.arrowdatapoints_without_3
        runs = run_reasoning.find_runs_without_jacks(adps, verbose = self.verbose)

        n_edits_reasoner = 0
        edited_runs_reasoner = []
        n_edits_mlscore = 0
        edited_runs_mlscore = []
        for run in runs:
            limbs = pred_limbs[run[0]:run[1]]
            n_double_steps = sum([len(list(g))-1 for k, g in itertools.groupby(limbs)])
            if n_double_steps / len(limbs) >= max_frac_doublestep:
                continue            
            if n_double_steps == 0:
                continue
            
            start_left_limbs = np.tile([0, 1], len(limbs) // 2 + 1)[:len(limbs)]
            start_right_limbs = np.tile([1, 0], len(limbs) // 2 + 1)[:len(limbs)]

            # 1. try using run pattern reasoner to decide limbs
            run_adps = adps[run[0]:run[1]]
            reason_left_score = run_reasoning.score_run(run_adps, start_left_limbs)
            reason_right_score = run_reasoning.score_run(run_adps, start_right_limbs)
            if reason_left_score > reason_right_score:
                pred_limbs[run[0]:run[1]] = start_left_limbs
                n_edits_reasoner += 1
                edited_runs_reasoner.append(run)
                continue
            elif reason_left_score < reason_right_score:
                pred_limbs[run[0]:run[1]] = start_right_limbs
                n_edits_reasoner += 1
                edited_runs_reasoner.append(run)
                continue
            # if tie, then proceed to #2

            # 2. score starting w left vs right using ML models
            left_pl = pred_limbs.copy()
            left_pl[run[0]:run[1]] = start_left_limbs
            start_left_score = self.score(left_pl)

            right_pl = pred_limbs.copy()
            right_pl[run[0]:run[1]] = start_right_limbs
            start_right_score = self.score(right_pl)

            # hacky; using 0 = left and 1 = right
            initial_limb = limbs[0]
            limb_scores = [start_left_score, start_right_score]
            initial_limb_score = limb_scores[initial_limb]
            change_limb_score = limb_scores[1 - initial_limb]
            pls = [left_pl, right_pl]
            pl_using_initial_limb = pls[initial_limb]
            pl_using_changed_limb = pls[1 - initial_limb]

            # trust limb on initial note, unless starting with other limb is better by threshold
            if change_limb_score > initial_limb_score + 1:
                pred_limbs = pl_using_changed_limb
            else:
                pred_limbs = pl_using_initial_limb
            
            n_edits_mlscore += 1
            edited_runs_mlscore.append(run)

        if self.verbose:
            logger.debug(f'Corrected no-jack run sections ...')
            logger.debug(f'Corrected {n_edits_reasoner} sections with reasoner: {edited_runs_reasoner=}')
            logger.debug(f'Corrected {n_edits_mlscore} sections with ML scorer: {edited_runs_mlscore=}')
        return pred_limbs

    """
        Model predictions
    """
    @functools.lru_cache
    def predict_arrow(self, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        if logp:
            return self.models.model_arrows_to_limb.predict_log_prob(points)
        else:
            return self.models.model_arrows_to_limb.predict(points)

    def predict_arrowlimbs(self, limb_array: NDArray, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrowlimbs_with_context(limb_array)
        if logp:
            return self.models.model_arrowlimbs_to_limb.predict_log_prob(points)
        else:
            return self.models.model_arrowlimbs_to_limb.predict(points)

    @functools.lru_cache
    def predict_matchnext(self, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        if logp:
            return self.models.model_arrows_to_matchnext.predict_log_prob(points)
        else:
            return self.models.model_arrows_to_matchnext.predict(points)

    @functools.lru_cache
    def predict_matchprev(self, logp: bool = False) -> NDArray:
        points = self.fcs.featurize_arrows_with_context()
        if logp:
            return self.models.model_arrows_to_matchprev.predict_log_prob(points)
        else:
            return self.models.model_arrows_to_matchprev.predict(points)
