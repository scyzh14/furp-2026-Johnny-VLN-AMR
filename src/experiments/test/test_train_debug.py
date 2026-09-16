'''
Phase-6 validation: the REAL train.py entry point on PyTorch 2.7.

Launches the actual CLI entry `python3 tasks/R2R/train.py --debug` as a
subprocess (no imports, no mocks) and asserts the full chain:

    argparse --debug config (batch=4, n_iters=3, train split only)
      -> setup() seeds
      -> R2RBatch initialization (features + nav graphs)
      -> Encoder/Decoder .cuda()
      -> 3 full training iterations (forward + loss + backward + Adam step)
      -> iteration counter reaches n_iters (3/3 = 100%)
      -> loss log CSV written; NO snapshot checkpoints written

Default behavior of `python3 tasks/R2R/train.py` (no flag) is unchanged:
batch 100, full 20000-iteration baseline with validation and snapshots.
Algorithm code paths are untouched; the debug flag only gates validation
environment creation and snapshot saving.

Run from the repo root inside the modern container:
    python3 tests/test_train_debug.py
'''

import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR = os.path.join(REPO_ROOT, 'tasks', 'R2R', 'snapshots')
PLOT_CSV = os.path.join(REPO_ROOT, 'tasks', 'R2R', 'plots', 'seq2seq_sample_imagenet_log.csv')


def main():
    cmd = [sys.executable, 'tasks/R2R/train.py', '--debug']
    print('[run] %s (cwd=%s)' % (' '.join(cmd), REPO_ROOT))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)

    # Full stdout is the test log
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit('train.py --debug exited with code %d' % proc.returncode)

    out = proc.stdout

    # Stage assertions, in order of appearance in the stdout stream
    def find(pattern):
        m = re.search(pattern, out)
        assert m, 'expected output missing: %r' % pattern
        return m

    find(r'R2RBatch loaded with \d+ instructions, using splits: train')
    print('[1] R2RBatch initialized on train split: OK')

    find(r'Training with sample feedback')
    print('[2] Seq2SeqAgent created with original default feedback (sample): OK')

    # No CUDA errors and iterations completed: the print happens after
    # forward/backward/step for all intervals
    m = find(r'\((\d+) (\d+)%\) train loss: ([\d.]+)')
    iters, pct, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    assert iters == 3 and pct == 100, 'expected 3/3 iterations, got %d (%d%%)' % (iters, pct)
    print('[3] iterations completed: 3/3 (100%%), train loss %.4f: OK' % loss)
    assert loss > 0, 'loss must be a valid positive number'
    print('[4] valid loss produced (forward+backward+optimizer.step all ran): OK')

    assert os.path.exists(PLOT_CSV), 'training log CSV missing: %s' % PLOT_CSV
    print('[5] training log written: tasks/R2R/plots/seq2seq_sample_imagenet_log.csv')

    snapshots = [f for f in os.listdir(SNAPSHOT_DIR) if f.startswith('seq2seq')]
    assert not snapshots, 'debug run must not write snapshots, found: %s' % snapshots
    print('[6] no snapshot checkpoints written (snapshots dir unchanged): OK')

    print('')
    print('TRAIN.PY DEBUG ENTRY TEST PASSED')


if __name__ == '__main__':
    main()
