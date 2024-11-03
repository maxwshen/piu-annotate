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
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.datapoints import ArrowDataPoint
from piu_annotate.formats import notelines
from piu_annotate.reasoning.reasoners import LimbReusePattern


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
    def __init__(
        self, 
        cs: ChartStruct, 
        fcs: ChartStructFeaturizer, 
        model_suite: ModelSuite, 
        verbose: bool = False
    ):
        """ Tactician uses a suite of ML models and a set of tactics to
            optimize predicted limb annotations for a given ChartStruct.
        """
        self.cs = cs
        self.fcs = fcs
        self.models = model_suite
        self.verbose = verbose
        self.pred_coords = self.cs.get_prediction_coordinates()
        self.singles_or_doubles = self.cs.singles_or_doubles()

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
    def initial_predict(
        self, 
        init_pred_limbs: NDArray | None = None,
        abstained_lr_patterns: list[LimbReusePattern] = [],
    ) -> NDArray:
        """ Initial prediction using arrow_to_limb and arrowlimbs_to_limb models
            
            If optional `init_pred_limbs` is provided, use those values,
            filling in missing entries (-1) with arrow_to_limb model prediction.
            Then, run arrowlimbs_to_limb model on all.
        """
        pred_limbs = self.predict_arrow()

        # Combine init_pred_limbs from reasoner with pred_limbs
        if init_pred_limbs is not None:
            # overwrite
            mask = (init_pred_limbs != -1)
            pred_limbs[mask] = init_pred_limbs[mask]

        pred_limbs = self.enforce_arrow_after_hold_release(pred_limbs)
        pred_limbs = self.predict_arrowlimbs(pred_limbs)
        pred_limbs = self.enforce_arrow_after_hold_release(pred_limbs)

        # use ML scoring to decide starting limb on abstained_lr_patterns
        # this enforces prediction follows reasoned limb reuse pattern
        for lr_pattern in abstained_lr_patterns:
            pred_limbs = self.decide_limb_reuse_pattern(lr_pattern, pred_limbs)

        if self.verbose:
            logger.debug(f'Used ML scoring to fill in {len(abstained_lr_patterns)} abstained LimbReusePatterns')
        return pred_limbs

    """
        Handle limb reuse patterns
    """
    def decide_limb_reuse_pattern(
        self, 
        lr_pattern: LimbReusePattern, 
        pred_limbs: NDArray
    ) -> NDArray:
        """ Decide starting limb on `lr_pattern` using several approaches,
            including ML scoring, and looking at previous line
            This enforces that the prediction adheres to limb reuse pattern from reasoner
        """
        dp_idxs = lr_pattern.downpress_idxs

        # if lr_pattern starts on very first note
        if dp_idxs[0] == 0:
            start_limb = None
            first_arrow = self.pred_coords[0].arrow_pos
            if self.singles_or_doubles == 'singles':
                if first_arrow < 2:
                    start_limb = 'left'
                elif first_arrow > 2:
                    start_limb = 'right'
            elif self.singles_or_doubles == 'doubles':
                if first_arrow < 5:
                    start_limb = 'left'
                else:
                    start_limb = 'right'
            if start_limb is not None:
                pred_limbs[dp_idxs] = lr_pattern.fill_limbs(start_limb)
                return pred_limbs
        
        if dp_idxs[0] > 0:
            init_idx = self.pred_coords[dp_idxs[0]].row_idx
            prev_line = self.cs.df.at[init_idx-1, 'Line with active holds'].replace('`', '')
            init_arrow = self.pred_coords[dp_idxs[0]].arrow_pos

            limb_map = {0: 'left', 1: 'right'}

            # if previous line has single hold release only
            row_idx_to_prev_pc = self.fcs.row_idx_to_prevs
            if notelines.has_one_hold_release(prev_line):
                # get pred_coord of hold release
                hold_release_arrow = prev_line.index('3')
                hold_pc_idx = row_idx_to_prev_pc[init_idx - 1][hold_release_arrow]
                hold_limb = pred_limbs[hold_pc_idx]

                if hold_release_arrow == init_arrow:
                    start_limb = limb_map[hold_limb]
                else:
                    start_limb = limb_map[1 - hold_limb]
                pred_limbs[dp_idxs] = lr_pattern.fill_limbs(start_limb)
                return pred_limbs
            elif notelines.is_hold_release(prev_line):
                # multiple hold releases
                hold_release_arrows = [i for i, s in enumerate(prev_line) if s == '3']
                if init_arrow in hold_release_arrows:
                    hold_pc_idx = row_idx_to_prev_pc[init_idx - 1][init_arrow]
                    hold_limb = pred_limbs[hold_pc_idx]

                    # reuse same limb as hold
                    start_limb = limb_map[hold_limb]
                    pred_limbs[dp_idxs] = lr_pattern.fill_limbs(start_limb)
                    return pred_limbs

        return self.ml_score_limb_reuse_pattern(lr_pattern, pred_limbs)

    def ml_score_limb_reuse_pattern(
        self, 
        lr_pattern: LimbReusePattern, 
        pred_limbs: NDArray
    ) -> NDArray:
        """ Decide limb use for `lr_pattern` using ML scoring.
            This preserves limb reuse pattern (alternate/same) from reasoner.
        """
        dp_idxs = lr_pattern.downpress_idxs

        start_left_limbs = lr_pattern.fill_limbs('left')
        start_right_limbs = lr_pattern.fill_limbs('right')

        left_pl = pred_limbs.copy()
        left_pl[dp_idxs] = start_left_limbs
        left_score = self.score(left_pl)

        right_pl = pred_limbs.copy()
        right_pl[dp_idxs] = start_right_limbs
        right_score = self.score(right_pl)

        if left_score > right_score:
            pred_limbs = left_pl
        elif left_score < right_score:
            pred_limbs = right_pl
        return pred_limbs

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

    def enforce_arrow_after_hold_release(self, pred_limbs: NDArray) -> NDArray:
        """ For line with exactly one arrow (1 or 2)
            following a line comprising one hold release,
            enforce:
            - Alternate limb if new arrow is different than hold arrow
            - Same limb if new arrow matches hold arrow
        """
        # Find candidate lines
        lines = [l.replace('`', '') for l in self.cs.df['Line with active holds']]
        
        row_idx_to_prev_pc = self.fcs.row_idx_to_prevs
        n_edits = 0
        for row_idx, (line1, line2) in enumerate(itertools.pairwise(lines)):
            if all([
                notelines.has_one_hold_release(line1),
                notelines.num_downpress(line2) == 1,
                not notelines.has_active_hold(line1),
                not notelines.has_active_hold(line2),
            ]):
                # check if any hold release very close in time before line1
                prev_line = self.cs.df.at[row_idx-1, 'Line with active holds']
                if notelines.is_hold_release(prev_line):
                    time_elapse = self.cs.df.at[row_idx, 'Time'] - self.cs.df.at[row_idx - 1, 'Time']
                    if time_elapse < 0.1:
                        import code; code.interact(local=dict(globals(), **locals()))

                hold_release_arrow = line1.index('3')
                next_arrow = [i for i, s in enumerate(line2) if s != '0'][0]

                hold_pc_idx = row_idx_to_prev_pc[row_idx][hold_release_arrow]
                downpress_pc_idx = self.row_idx_to_pcs[row_idx + 1][0]

                hold_limb = pred_limbs[hold_pc_idx]
                downpress_limb = pred_limbs[downpress_pc_idx]

                if hold_release_arrow == next_arrow:
                    if hold_limb != downpress_limb:
                        pred_limbs[downpress_pc_idx] = hold_limb
                        n_edits += 1
                
                elif hold_release_arrow != next_arrow:
                    if hold_limb == downpress_limb:
                        pred_limbs[downpress_pc_idx] = 1 - hold_limb
                        n_edits += 1
                
        if self.verbose:
            logger.debug(f'Changed {n_edits} arrows after line with one hold release')
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
