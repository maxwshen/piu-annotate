from loguru import logger

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.tactics import Tactician
from piu_annotate.ml.models import ModelSuite


def predict(
    cs: ChartStruct, 
    model_suite: ModelSuite,
    verbose: bool = False,
) -> ChartStruct:
    """ Use tactician to predict limb annotations for `cs` """
    tactics = Tactician(cs, model_suite, verbose = verbose)
    fcs = featurizers.ChartStructFeaturizer(cs)

    score_to_limbs = dict()

    pred_limbs = tactics.initial_predict()
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, initial pred: {tactics.score(pred_limbs):.3f}')
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

    pred_limbs = tactics.remove_doublesteps_in_long_nojack_runs(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, remove doublesteps in long nojack runs: {tactics.score(pred_limbs):.3f}')
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
        Attempt to fix impossible lines
    """
    pred_limbs = tactics.detect_impossible_multihit(score_to_limbs[max(score_to_limbs)])
    if verbose:
        logger.info(f'Score, fix impossible multihit: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    pred_limbs = tactics.detect_impossible_lines_with_holds(pred_limbs)
    if verbose:
        logger.info(f'Score, fix impossible lines with holds: {tactics.score(pred_limbs):.3f}')
        fcs.evaluate(pred_limbs, verbose = True)

    return cs, fcs, pred_limbs