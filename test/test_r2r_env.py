'''
Phase-4 validation: R2R environment data flow.

Verifies the ORIGINAL data flow end to end, with MatterSim still in the loop
(the simulator must NOT be replaced by pre-computed teacher action tables):

    R2R_*.json
      -> R2RBatch (tokenize instructions, build per-scan nav graphs,
                   all-pairs shortest paths)
      -> EnvBatch -> MatterSim.newEpisode (live simulator state)
      -> MatterSim.getState (live state: viewpoint / viewIndex / navLocs)
      -> cached ResNet-152 feature lookup keyed by (scan, viewpoint, viewIndex)
      -> observation dict (feature + instruction encoding + teacher action)
      -> MatterSim.makeAction driven by teacher -> next live state

Run from the repo root inside the modern container:
    python3 tests/test_r2r_env.py

The only R2R code change required to run on modern networkx is
G.node -> G.nodes in env.py (tracked in git / MIGRATION_LOG.md).
'''

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)                 # so 'build' relative paths resolve
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'R2R'))  # env / utils imports
os.chdir(REPO_ROOT)

import numpy as np
import MatterSim
from utils import read_vocab, Tokenizer
from env import EnvBatch, R2RBatch

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
FEATURE_STORE = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80
BATCH_SIZE = 4
MAX_TEACHER_STEPS = 40


def main():
    # ---- Data: vocab + tokenizer ----
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    print('[1] vocab size=%d, tokenizer ready' % len(vocab))

    # ---- R2RBatch construction (loads full TSV feature store + nav graphs) ----
    env = R2RBatch(FEATURE_STORE, batch_size=BATCH_SIZE, splits=['train'], tokenizer=tok)
    print('[2] R2RBatch ready: %d instructions across %d scans' % (len(env.data), len(env.scans)))

    # The simulator object must exist and be the real MatterSim binding
    assert isinstance(env.env, EnvBatch)
    assert isinstance(env.env.sim, MatterSim.Simulator), 'simulator must be a live MatterSim instance'
    assert env.env.features is not None and len(env.env.features) > 0
    print('[3] EnvBatch wraps a live MatterSim.Simulator; cached feature store loaded: %d viewpoints'
          % len(env.env.features))

    # ---- reset(): data JSON -> newEpisode -> getState -> feature lookup -> obs ----
    obs = env.reset()
    assert len(obs) == BATCH_SIZE
    goals = [item['path'][-1] for item in env.batch]
    print('[4] reset() produced %d observations' % BATCH_SIZE)

    for i, ob in enumerate(obs):
        # observation keys defined by the original env.py
        for key in ['instr_id', 'scan', 'viewpoint', 'viewIndex', 'heading',
                    'elevation', 'feature', 'step', 'navigableLocations',
                    'instructions', 'teacher', 'instr_encoding']:
            assert key in ob, 'missing observation key: %s' % key

        # instruction encoding: reversed, EOS/PAD padded, fixed length
        assert ob['instr_encoding'].shape == (MAX_INPUT_LENGTH,), \
            'instr_encoding shape %s' % (ob['instr_encoding'].shape,)

        # cached feature must match the feature store keyed by LIVE sim state
        long_id = ob['scan'] + '_' + ob['viewpoint']
        expected_feature = env.env.features[long_id][ob['viewIndex'], :]
        assert ob['feature'].shape == (2048,), 'feature shape %s' % (ob['feature'].shape,)
        assert np.array_equal(ob['feature'], expected_feature), \
            'feature does not match cached store for live sim state'

        # teacher action validity: index references a real navigable location
        ix, h, e = ob['teacher']
        assert 0 <= ix < len(ob['navigableLocations']), 'teacher index out of range'

        # start state must equal the path's first viewpoint
        assert ob['viewpoint'] == env.batch[i]['path'][0]
        print('    obs[%d]: scan=%s vp=%s viewIndex=%2d feature=%s teacher=%s instr=%r'
              % (i, ob['scan'], ob['viewpoint'][:8], ob['viewIndex'],
                 ob['feature'].shape, ob['teacher'], ob['instructions'][:40]))

    # ---- Teacher-forced rollout: actions drive the LIVE simulator ----
    reached = 0
    for t in range(MAX_TEACHER_STEPS):
        actions = [ob['teacher'] for ob in obs]
        obs = env.step(actions)
        # step counter comes from the live simulator
        assert all(ob['step'] == t + 1 for ob in obs), 'simulator step mismatch'
        if all(ob['teacher'] == (0, 0, 0) for ob in obs):
            break

    # After teacher rollout every trajectory should stand at its goal viewpoint
    for i, ob in enumerate(obs):
        dist = env.distances[ob['scan']][ob['viewpoint']][goals[i]]
        ok = (ob['viewpoint'] == goals[i]) and (dist == 0)
        reached += int(ok)
        print('    end[%d]: vp=%s goal=%s graph_dist=%d %s'
              % (i, ob['viewpoint'][:8], goals[i][:8], dist, 'REACHED' if ok else 'NOT REACHED'))
    assert reached == BATCH_SIZE, 'teacher policy should reach every goal'

    print('[5] teacher-forced rollout through MatterSim: all %d/%d trajectories reached goal'
          % (reached, BATCH_SIZE))
    print('')
    print('ALL R2R ENVIRONMENT DATA-FLOW TESTS PASSED')


if __name__ == '__main__':
    main()
