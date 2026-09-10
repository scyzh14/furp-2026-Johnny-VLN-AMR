'''
Minimal MatterSim reproduction test for the modern (CUDA 12.8 / PyTorch 2.x)
migration environment.

Scope (phase 3): verify the Python binding and the navigation STATE MACHINE
only, with rendering DISABLED (the R2R baseline trains with
setRenderingEnabled(False)).

Checks:
  TEST 2: import MatterSim
  TEST 3: simulator construction + initialize()
  TEST 4: load a scan / episode (newEpisode + getState)
  TEST 5: state transition via makeAction
          - turn left/right changes discretized heading (30 deg steps)
          - look up/down changes elevation / viewIndex
          - forward moves to the expected navigable viewpoint
  Navigation graph / state machine do NOT depend on GL or skybox images,
  so no GPU rendering is exercised here.

Run from repo root inside the modern container:
    PYTHONPATH=build python3 tests/test_mattersim.py
(or simply: python3 tests/test_mattersim.py  -- it adds build/ to sys.path)
'''

import os
import sys
import json
import math

sys.path.append('build')

DATASET_PATH = os.environ.get('MP3D_SCAN_PATH', '/mp3d_data/v1/scans')
NAVGRAPH_PATH = os.environ.get('MP3D_NAVGRAPH_PATH', 'connectivity')

HEADING_INCREMENT = math.pi * 2.0 / 12  # 30 degrees, as in upstream env.py
ELEVATION_INCREMENT = math.pi / 6.0     # 30 degrees


def load_scan_and_viewpoint():
    with open(os.path.join(NAVGRAPH_PATH, 'scans.txt')) as f:
        scan = f.readline().strip()
    with open(os.path.join(NAVGRAPH_PATH, '%s_connectivity.json' % scan)) as f:
        conns = json.load(f)
    viewpoints = [c['image_id'] for c in conns if c['included']]
    return scan, viewpoints


def main():
    # TEST 2: import
    import MatterSim
    print('[TEST 2] import MatterSim: OK')

    scan, viewpoints = load_scan_and_viewpoint()
    print('         scan=%s, %d included viewpoints' % (scan, len(viewpoints)))

    # TEST 3: construct + initialize
    sim = MatterSim.Simulator()
    sim.setDatasetPath(DATASET_PATH)
    sim.setNavGraphPath(NAVGRAPH_PATH)
    sim.setRenderingEnabled(False)            # R2R baseline setting
    sim.setDiscretizedViewingAngles(True)     # R2R baseline setting
    sim.setBatchSize(1)
    sim.initialize()
    print('[TEST 3] Simulator constructed and initialized (rendering off): OK')

    # TEST 4: find a viewpoint with a visible forward edge at some heading
    move_case = None
    for vp in viewpoints[:200]:
        sim.newEpisode([scan], [vp], [0.0], [0.0])
        state = sim.getState()[0]
        assert state.scanId == scan
        assert state.step == 0
        assert state.location.viewpointId == vp
        nav = state.navigableLocations
        if len(nav) > 1:
            move_case = (vp, nav[1].viewpointId)
            break
    assert move_case is not None, 'no viewpoint with a navigable neighbour found'
    print('[TEST 4] newEpisode/getState: OK (vp %s has neighbour %s)' % move_case)

    # TEST 5a: forward action lands on the expected navigable viewpoint
    vp, target_vp = move_case
    sim.newEpisode([scan], [vp], [0.0], [0.0])
    sim.makeAction([1], [0.0], [0.0])       # index 1 == forward (navigableLocations[1])
    s = sim.getState()[0]
    assert s.step == 1, 'step should increment to 1, got %d' % s.step
    assert s.location.viewpointId == target_vp, \
        'forward should move to %s, got %s' % (target_vp, s.location.viewpointId)
    print('[TEST 5a] forward action -> expected viewpoint, step=1: OK')

    # TEST 5b: turn right snaps heading to +30 degrees
    sim.newEpisode([scan], [vp], [0.0], [0.0])
    sim.makeAction([0], [1.0], [0.0])       # turn right
    s = sim.getState()[0]
    assert abs(s.heading - HEADING_INCREMENT) < 1e-6, \
        'heading after right turn should be %.6f, got %.6f' % (HEADING_INCREMENT, s.heading)
    assert s.viewIndex == 13, 'viewIndex should be 13 after right turn, got %d' % s.viewIndex

    # TEST 5c: turn left from 0 wraps to heading 11*30deg
    sim.newEpisode([scan], [vp], [0.0], [0.0])
    sim.makeAction([0], [-1.0], [0.0])      # turn left
    s = sim.getState()[0]
    assert abs(s.heading - 11 * HEADING_INCREMENT) < 1e-6, \
        'heading after left turn should be 11*30deg, got %.6f' % s.heading

    # TEST 5d: look up changes elevation / viewIndex band
    sim.newEpisode([scan], [vp], [0.0], [0.0])
    sim.makeAction([0], [0.0], [1.0])       # look up
    s = sim.getState()[0]
    assert abs(s.elevation - ELEVATION_INCREMENT) < 1e-6, \
        'elevation after look up should be %.6f, got %.6f' % (ELEVATION_INCREMENT, s.elevation)
    assert s.viewIndex == 0 + 2 * 12, 'viewIndex should be 24 after look up, got %d' % s.viewIndex

    # TEST 5e: look down changes elevation band
    sim.newEpisode([scan], [vp], [0.0], [0.0])
    sim.makeAction([0], [0.0], [-1.0])      # look down
    s = sim.getState()[0]
    assert abs(s.elevation - (-ELEVATION_INCREMENT)) < 1e-6
    assert s.viewIndex == 0, 'viewIndex should be 0 after look down, got %d' % s.viewIndex

    sim.close()
    print('[TEST 5b-5e] turn/look state transitions (heading, elevation, viewIndex): OK')
    print('')
    print('ALL MATTERSIM STATE-MACHINE TESTS PASSED')


if __name__ == '__main__':
    main()
