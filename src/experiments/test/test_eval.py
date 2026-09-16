'''
Phase-8 evaluation: standalone eval.py scoring on PyTorch 2.7.

Two independent verifications:

(1) Score the trained Seq2Seq model's iter-20000 result JSONs using
    eval.py's Evaluation class (the same scorer used during training,
    but run here as a standalone entry point). Cross-check the metrics
    against the training CSV log.

(2) Run eval_simple_agents() (eval.py's __main__ entry) to verify the
    Stop / Shortest / Random baselines work on the modern stack and
    produce sensible reference numbers.

Run from repo root inside the container:
    python3 tests/test_eval.py
'''

import csv
import os
import sys

# tasks/R2R must be on the path for `from eval import Evaluation`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tasks', 'R2R'))

from eval import Evaluation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO_ROOT, 'tasks', 'R2R', 'results')
CSV_LOG = os.path.join(REPO_ROOT, 'tasks', 'R2R', 'plots',
                       'seq2seq_sample_imagenet_log.csv')


def load_csv_final_row():
    '''Read the last row of the training loss log.'''
    with open(CSV_LOG) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-1]


def test_trained_model_eval():
    '''Score the iter-20000 result JSONs with eval.py's Evaluation.'''
    print('=== (1) Standalone Evaluation scoring (iter 20000) ===')
    csv_row = load_csv_final_row()
    print('Training CSV final row: iter=%s' % csv_row['iteration'])

    for split in ['val_seen', 'val_unseen']:
        result_file = os.path.join(
            RESULT_DIR,
            'seq2seq_sample_imagenet_%s_iter_20000.json' % split)
        assert os.path.exists(result_file), 'missing: %s' % result_file

        ev = Evaluation([split])
        score_summary, _ = ev.score(result_file)

        print('\n  %s (eval.py standalone):' % split)
        for k in ['length', 'nav_error', 'oracle success_rate',
                  'success_rate', 'spl']:
            print('    %-22s %.4f' % (k, score_summary[k]))

        # Cross-check against the training CSV (same Evaluation scorer
        # was called during training, so values should match exactly)
        prefix = '%s ' % split
        for metric in ['nav_error', 'oracle success_rate',
                       'success_rate', 'spl']:
            csv_key = prefix + metric
            csv_val = float(csv_row[csv_key])
            eval_val = score_summary[metric]
            # Floating-point tolerance for re-scoring
            assert abs(csv_val - eval_val) < 1e-4, \
                '%s %s: CSV=%.6f eval=%.6f mismatch' % (
                    split, metric, csv_val, eval_val)
        print('  [OK] all metrics match training CSV within 1e-4')

    print('\nTRAINED MODEL EVAL: PASSED\n')


def test_simple_agents():
    '''Run eval.py's __main__ entry (Stop/Shortest/Random baselines).'''
    print('=== (2) eval_simple_agents (Stop/Shortest/Random) ===')
    from eval import eval_simple_agents
    eval_simple_agents()
    print('\nSIMPLE AGENTS EVAL: PASSED\n')


if __name__ == '__main__':
    test_trained_model_eval()
    test_simple_agents()
    print('ALL EVAL TESTS PASSED')
