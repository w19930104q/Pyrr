#!/usr/bin/env python
"""Diagnostic report for matrix44.decompose.

Runs a fixed set of matrices (normal, negative determinant, zero-scale and
sheared) through matrix44.decompose and verifies:

  * outputs are finite (no NaN/inf from zero-scale axes);
  * recomposition error is within tolerance for TRS-representable matrices;
  * sheared matrices decompose to the nearest TRS with a reported residual
    and reach a fixed point (no drift on repeated round trips);
  * the reflection sign convention is deterministic (always on the x scale);
  * normal matrices decompose bit-identically to the original algorithm.

Exits with status 1 if any check fails.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyrr import matrix44, quaternion


def original_decompose(m):
    """The decompose algorithm as it was before the fix (commit 6acd3ee)."""
    m = np.asarray(m)
    scale = np.linalg.norm(m[:3, :3], axis=1)
    det = np.linalg.det(m)
    if det < 0:
        scale[0] *= -1
    position = m[3, :3]
    rotation = m[:3, :3] * (1 / scale)[:, None]
    return scale, quaternion.create_from_matrix(rotation), position


def compose(scale, rotation, translation, dtype='f8'):
    s = matrix44.create_from_scale(scale, dtype=dtype)
    r = matrix44.create_from_quaternion(rotation, dtype=dtype)
    t = matrix44.create_from_translation(translation, dtype=dtype)
    return s.dot(r).dot(t)


def trs(scale, euler, translation, dtype='f8'):
    q = quaternion.create_from_eulers(np.array(euler, dtype=dtype), dtype=dtype)
    return compose(np.array(scale, dtype=dtype), q,
                   np.array(translation, dtype=dtype), dtype=dtype)


def fixed_matrix_set():
    """The fixed acceptance matrix set. Each entry is (name, matrix, kind)."""
    cases = []
    add = cases.append

    # normal matrices (TRS-representable, det > 0)
    add(('normal/identity', np.eye(4), 'normal'))
    add(('normal/trs', trs([1, 2, 3], [0.3, -0.7, 1.1], [10, 0, -5]), 'normal'))
    add(('normal/trs_f4', trs([1, 1, 2], [0, 0, np.pi], [10, 0, -5], dtype='f4'), 'normal'))
    add(('normal/double_neg_scale', trs([-1, -1, 2], [0.5, 0, 0], [1, 2, 3]), 'normal'))
    add(('normal/tiny_scale', trs([1e-4, 1, 1e4], [0.2, 0.4, 0.6], [0, 0, 0]), 'normal'))

    # negative determinant (odd number of negative scales)
    add(('negdet/neg_x', trs([-3, 1, 1], [0, 0.7, 0], [1, 2, 3]), 'negdet'))
    add(('negdet/neg_y', trs([1, -2, 1], [0, 0.7, 0], [1, 2, 3]), 'negdet'))
    add(('negdet/neg_z', trs([1, 1, -1], [0.3, 0.7, -0.2], [1, 2, 3]), 'negdet'))

    # zero scale axes
    add(('zero/z', trs([1, 2, 0], [0, 0.7, 0], [1, 2, 3]), 'zero'))
    add(('zero/x', trs([0, 2, 3], [0.4, 0, 0.9], [1, 2, 3]), 'zero'))
    add(('zero/yz', trs([3, 0, 0], [0, 0.7, 0], [1, 2, 3]), 'zero'))
    add(('zero/xyz', trs([0, 0, 0], [0, 0.7, 0], [1, 2, 3]), 'zero'))

    # shear (not TRS-representable)
    m = np.eye(4); m[0, 1] = 0.5
    add(('shear/xy', m, 'shear'))
    m = trs([1, 2, 3], [0.3, -0.7, 1.1], [10, 0, -5])
    m[0, 1] += 0.4; m[1, 2] += 0.3
    add(('shear/trs+shear', m, 'shear'))
    m = np.eye(4); m[0, 1] = 0.5; m[2, 2] = -1.0
    add(('shear/neg_det', m, 'shear'))

    return cases


def main():
    tol = {'f8': 1e-12, 'f4': 1e-5}
    failures = []
    rows = []

    for name, m, kind in fixed_matrix_set():
        m = np.asarray(m)
        eps = tol['f4' if m.dtype == np.dtype('f4') else 'f8']
        notes = []

        scale, rot, trans = matrix44.decompose(m)
        recomposed = compose(scale, rot, trans, dtype=m.dtype)
        recon_err = np.abs(recomposed - m).max()

        # 1. finite outputs
        finite = all(np.all(np.isfinite(x)) for x in (scale, rot, trans))
        if not finite:
            failures.append((name, 'non-finite output'))

        # 2. determinism: identical output on repeated calls
        again = matrix44.decompose(m)
        deterministic = all(np.array_equal(a, b) for a, b in zip((scale, rot, trans), again))
        if not deterministic:
            failures.append((name, 'non-deterministic output'))

        # 3. fixed point: decompose(compose(decompose(m))) == decompose(m)
        scale2, rot2, trans2 = matrix44.decompose(recomposed)
        drift = np.abs(compose(scale2, rot2, trans2, dtype=m.dtype) - recomposed).max()
        if drift > eps:
            failures.append((name, 'round-trip drift %.3e > %.0e' % (drift, eps)))

        # 4. per-kind checks
        if kind == 'normal':
            # recomposition within tolerance
            if recon_err > eps:
                failures.append((name, 'recompose error %.3e > %.0e' % (recon_err, eps)))
            # output unchanged vs the original algorithm
            o_scale, o_rot, o_trans = original_decompose(m)
            same = (np.array_equal(scale, o_scale) and np.array_equal(rot, o_rot)
                    and np.array_equal(trans, o_trans))
            if not same:
                failures.append((name, 'normal-matrix output changed'))
            notes.append('output==original: %s' % same)
        elif kind == 'negdet':
            if recon_err > eps:
                failures.append((name, 'recompose error %.3e > %.0e' % (recon_err, eps)))
            # deterministic sign convention: reflection on x, others positive
            if not (scale[0] < 0 and scale[1] > 0 and scale[2] > 0):
                failures.append((name, 'sign convention violated: %s' % (scale,)))
            notes.append('sign convention: x<0, y>0, z>0')
        elif kind == 'zero':
            if recon_err > eps:
                failures.append((name, 'recompose error %.3e > %.0e' % (recon_err, eps)))
        elif kind == 'shear':
            # shear cannot be represented exactly; the residual must be
            # bounded by the shear magnitude (distance to the nearest
            # axis-aligned-scale matrix is an upper bound) and stable
            bound = np.abs(m[:3, :3] - np.diag(np.diag(m[:3, :3]))).max() + eps
            if recon_err > bound:
                failures.append((name, 'shear residual %.3e exceeds bound %.3e'
                                 % (recon_err, bound)))
            notes.append('residual bounded by shear magnitude %.3e' % bound)

        rows.append((name, kind, finite, deterministic, recon_err, drift, '; '.join(notes)))

    print('decompose diagnostic report')
    print('===========================')
    print('%-26s %-7s %-6s %-6s %-12s %-12s %s'
          % ('matrix', 'kind', 'finite', 'determ', 'recompose_err', 'fixedpt_drift', 'notes'))
    for name, kind, finite, deterministic, recon_err, drift, notes in rows:
        print('%-26s %-7s %-6s %-6s %-12.3e %-12.3e %s'
              % (name, kind, 'yes' if finite else 'NO', 'yes' if deterministic else 'NO',
                 recon_err, drift, notes))
    print()
    if failures:
        print('FAIL: %d check(s) failed' % len(failures))
        for name, msg in failures:
            print('  - %s: %s' % (name, msg))
        return 1
    print('PASS: all %d matrices satisfy the acceptance criteria' % len(rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
