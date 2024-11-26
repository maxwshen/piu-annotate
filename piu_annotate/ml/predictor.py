from loguru import logger

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.tactics import Tactician
from piu_annotate.ml.models import ModelSuite
from piu_annotate.reasoning.reasoners import PatternReasoner


def predict(
    cs: ChartStruct, 
    model_suite: ModelSuite,
    verbose: bool = False,
) -> ChartStruct:
    """ Use Tactician and PatternReasoner to predict limb annotations for `cs`
    """
    fcs = featurizers.ChartStructFeaturizer(cs)
    reasoner = PatternReasoner(cs, verbose = verbose)
    tactics = Tactician(cs, fcs, model_suite, verbose = verbose)

    """
        1. Use PatternReasoner
    """
    pred_limbs, abstained_lr_patterns = reasoner.propose_limbs()

    """
        2. Use Tactician with ML models
    """
    score_to_limbs = dict()
    pred_limbs = tactics.initial_predict(pred_limbs, abstained_lr_patterns)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, initial pred: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.enforce_arrow_after_hold_release(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, arrow after hold release: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.flip_labels_by_score(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.flip_jack_sections(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip jacks: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.beam_search(score_to_limbs[max(score_to_limbs)], width = 5, n_iter = 3)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, beam search: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.fix_double_doublestep(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, fix double doublestep: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    # best score
    if verbose:
        best_score = max(score_to_limbs.keys())
        logger.success(f'Found {best_score=:.3f}')

    """
        Final tactics, to guarantee they are obeyed
    """
    pred_limbs = score_to_limbs[max(score_to_limbs)]

    pred_limbs = tactics.enforce_arrow_after_hold_release(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, arrow after hold release: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.detect_impossible_multihit(pred_limbs)
    if verbose:
        logger.info(f'Score, fix impossible multihit: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.detect_impossible_lines_with_holds(pred_limbs)
    if verbose:
        logger.info(f'Score, fix impossible lines with holds: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    if cs.get_chart_level() <= 15:
        pred_limbs = tactics.remove_unforced_brackets(pred_limbs)
        if verbose:
            logger.info(f'Score, fix low-level unforced brackets: {tactics.score(pred_limbs):.3f}')
            fcs.evaluate(pred_limbs, verbose = True)

    return cs, fcs, pred_limbs