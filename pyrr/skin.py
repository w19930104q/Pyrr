# -*- coding: utf-8 -*-
"""Skeletal animation chain: quaternion interpolation, matrix rebuild and
vertex skinning.

This module wires three existing pieces of Pyrr into one sign-consistent
pipeline:

1.  **quaternion shortest path interpolation** (:func:`interpolate_keyframes`
    / :func:`slerp_shortest_path`) -- keyframe rotations are blended along
    the short arc of the 4D sphere.
2.  **matrix rebuild** (:func:`joint_matrix`) -- an interpolated rotation,
    translation and scale are rebuilt into a single row-major
    :class:`numpy.ndarray` of shape ``(4, 4)``, using the exact same
    conventions as :mod:`pyrr.matrix44`.
3.  **vertex transformation** (:func:`skin_vertices` /
    :func:`transform_vertex`) -- vertices (optionally weighted over several
    joints) are transformed with the rebuilt matrices.

Conventions (identical to the rest of Pyrr)
--------------------------------------------

* Matrices are **row-major** and points are **row vectors**; a point is
  transformed with ``v' = v @ M`` (see
  :func:`pyrr.matrix44.apply_to_vector`).
* A joint's local transform is the row-vector order
  ``M = T @ R @ S``, so a vertex passes through translation first, then
  rotation, then scale::

      v' = ((v @ T) @ R) @ S

  Because the scale is applied last (outermost on the right), it scales the
  translated position as well; with a non-unit scale the translation values
  visible in the matrix's last row are the per-axis-scaled translation,
  ``[tx*sx, ty*sy, tz*sz, 1]``. With a unit scale the last row holds the
  raw translation, exactly like
  :func:`pyrr.matrix44.create_from_translation`.

* Quaternions are stored as ``[x, y, z, w]``. ``q`` and ``-q`` describe the
  **same rotation** (the unit quaternion double covers the rotation group).
  The interpolation functions below normalise that ambiguity explicitly by
  forcing the dot product of the two keyframes to be non-negative, so the
  blend always follows the shortest path regardless of which cover each
  keyframe happens to use.
* Rotating a vector with :func:`pyrr.quaternion.apply_to_vector` is
  equivalent to multiplying it by
  :func:`pyrr.matrix44.create_from_inverse_of_quaternion`. This module uses
  that matrix builder exclusively for the rotation stage, which keeps the
  quaternion stage and the matrix stage bit-for-bit consistent. The other
  builder, :func:`pyrr.matrix44.create_from_quaternion`, stays untouched and
  keeps its own existing semantics.
* Scale matrices (:func:`pyrr.matrix44.create_from_scale`) hold the per-axis
  scale factors on the diagonal ``diag(sx, sy, sz, 1)``. They only ever
  contain scale (the bottom row remains ``[0, 0, 0, 1]``), so the
  translation stored in the last row is never interpreted as a scale.

Every function here is composable: running the whole chain on a pose gives
the same vertex positions as applying each stage separately by hand with the
individual :mod:`pyrr.quaternion` and :mod:`pyrr.matrix44` functions.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from . import matrix44, quaternion
from .utils import all_parameters_as_numpy_arrays, parameters_as_numpy_arrays


def slerp_shortest_path(quat1, quat2, t):
    """Spherically interpolate between two unit quaternions along the
    shortest path.

    Unlike :func:`pyrr.quaternion.slerp`, the double-cover ambiguity
    (``q`` and ``-q`` encode the same rotation) is resolved explicitly:
    the second quaternion is flipped whenever its dot product with the
    first one is negative, guaranteeing that the interpolation travels
    the short arc even at exactly 180 degrees of separation, where the
    generic formula is numerically unstable.

    The parameter ``t`` is clamped to the range ``[0, 1]``. Inputs do not
    have to be pre-normalised; they are normalised internally.

    :param numpy.array quat1: The start quaternion, shape ``(4,)`` in
        ``[x, y, z, w]`` order.
    :param numpy.array quat2: The end quaternion, shape ``(4,)`` in
        ``[x, y, z, w]`` order. May use either cover (``q`` or ``-q``).
    :param float t: The interpolation factor, ``0.0`` returns ``quat1``
        and ``1.0`` returns a quaternion equivalent to ``quat2``.
    :rtype: numpy.array
    :return: The interpolated unit quaternion, shape ``(4,)``.
    """
    quat1 = quaternion.normalize(np.asarray(quat1, dtype=np.float64))
    quat2 = quaternion.normalize(np.asarray(quat2, dtype=np.float64))

    t = float(np.clip(t, 0.0, 1.0))

    # resolve the q / -q double cover: always interpolate towards the
    # cover of quat2 that lies on the same hemisphere as quat1
    dot = float(np.dot(quat1, quat2))
    if dot < 0.0:
        quat2 = -quat2
        dot = -dot

    # clamp against rounding that would push the arccos argument past 1.0
    dot = min(dot, 1.0)

    if dot > 0.9995:
        # nearly identical orientations: linear blend is numerically
        # stabler than the sine-ratio form and visually identical
        result = quat1 * (1.0 - t) + quat2 * t
    else:
        angle = np.arccos(dot)
        sin_angle = np.sin(angle)
        result = (
            quat1 * (np.sin(angle * (1.0 - t)) / sin_angle)
            + quat2 * (np.sin(angle * t) / sin_angle)
        )

    return quaternion.normalize(result)


@all_parameters_as_numpy_arrays
def lerp_shortest_path(quat1, quat2, t):
    """Normalised linear interpolation between two quaternions along the
    shortest path.

    Cheaper than :func:`slerp_shortest_path` at the cost of non-constant
    angular velocity. The same double-cover resolution is applied first,
    so the blend never takes the long way around.

    The parameter ``t`` is clamped to the range ``[0, 1]``.

    :param numpy.array quat1: The start quaternion, shape ``(4,)``.
    :param numpy.array quat2: The end quaternion, shape ``(4,)``.
    :param float t: The interpolation factor.
    :rtype: numpy.array
    :return: The interpolated unit quaternion, shape ``(4,)``.
    """
    quat1 = quaternion.normalize(quat1)
    quat2 = quaternion.normalize(quat2)

    t = float(np.clip(t, 0.0, 1.0))

    if np.dot(quat1, quat2) < 0.0:
        quat2 = -quat2

    return quaternion.normalize(quat1 * (1.0 - t) + quat2 * t)


def interpolate_keyframes(times, rotations, t):
    """Interpolate a quaternion keyframe sequence at time ``t`` using
    shortest-path slerp.

    ``times`` must be sorted in ascending order. Outside the keyframe
    range the first / last rotation is held (clamped extrapolation).

    :param numpy.array times: Keyframe times, shape ``(n,)``.
    :param numpy.array rotations: Keyframe quaternions, shape ``(n, 4)`` in
        ``[x, y, z, w]`` order. Individual keyframes may use either cover
        (``q`` or ``-q``); the shortest path is selected per segment.
    :param float t: The sample time.
    :rtype: numpy.array
    :return: The interpolated unit quaternion, shape ``(4,)``.
    """
    times = np.asarray(times, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)

    if times.shape[0] != rotations.shape[0]:
        raise ValueError(
            "times and rotations must contain the same number of keyframes"
        )
    if times.ndim != 1 or rotations.ndim != 2 or rotations.shape[1] != 4:
        raise ValueError(
            "expected times of shape (n,) and rotations of shape (n, 4)"
        )

    # clamped extrapolation
    if t <= times[0]:
        return quaternion.normalize(rotations[0])
    if t >= times[-1]:
        return quaternion.normalize(rotations[-1])

    # np.searchsorted gives the first keyframe time strictly after t
    upper = int(np.searchsorted(times, t, side='right'))
    lower = upper - 1
    span = times[upper] - times[lower]
    local_t = 0.0 if span == 0.0 else (t - times[lower]) / span

    return slerp_shortest_path(rotations[lower], rotations[upper], local_t)


@parameters_as_numpy_arrays('translation', 'scale', 'rotation')
def joint_matrix(translation, rotation, scale=None, dtype=None):
    """Rebuild a joint's local transformation matrix from separate
    translation, rotation and scale components.

    The components are combined in Pyrr's row-vector order::

        M = T @ R @ S        so that        v' = v @ T @ R @ S

    i.e. a vertex is translated first, then rotated, then scaled.

    The rotation matrix is built with
    :func:`pyrr.matrix44.create_from_inverse_of_quaternion`, which applies
    exactly the rotation described by
    :func:`pyrr.quaternion.apply_to_vector`. Both covers of the rotation
    (``rotation`` and ``-rotation``) therefore produce the same matrix.

    The scale matrix is :func:`pyrr.matrix44.create_from_scale`: a diagonal
    matrix with the per-axis factors on the diagonal and a ``1`` in the
    bottom-right corner; it never contributes translation. Under the
    row-vector ``T @ R @ S`` order the scale additionally scales the
    translated position, so the matrix's last row is the raw translation
    for a unit scale and ``[tx*sx, ty*sy, tz*sz, 1]`` otherwise.

    :param numpy.array translation: The joint translation, shape ``(3,)``.
    :param numpy.array rotation: The joint rotation quaternion, shape
        ``(4,)`` in ``[x, y, z, w]`` order.
    :param numpy.array scale: Optional per-axis scale, shape ``(3,)``.
        Defaults to ``(1, 1, 1)``.
    :param dtype: The numpy dtype of the returned matrix. Defaults to the
        component dtype.
    :rtype: numpy.array
    :return: A row-major matrix with shape ``(4, 4)``.
    """
    if scale is None:
        scale = np.ones(3, dtype=translation.dtype)
    scale = np.asarray(scale)

    translation_matrix = matrix44.create_from_translation(translation, dtype)
    # inverse-of-quaternion is the builder whose v @ M matches
    # quaternion.apply_to_vector(rotation, v)
    rotation_matrix = matrix44.create_from_inverse_of_quaternion(rotation, dtype)
    scale_matrix = matrix44.create_from_scale(scale, dtype)

    return matrix44.multiply(
        matrix44.multiply(translation_matrix, rotation_matrix),
        scale_matrix
    )


def joint_matrices(translations, rotations, scales=None, dtype=None):
    """Rebuild a batch of joint matrices with :func:`joint_matrix`.

    :param numpy.array translations: Translations, shape ``(j, 3)``.
    :param numpy.array rotations: Rotation quaternions, shape ``(j, 4)``.
    :param numpy.array scales: Optional per-axis scales, shape ``(j, 3)``.
    :param dtype: The numpy dtype of the returned matrices.
    :rtype: numpy.array
    :return: Matrices with shape ``(j, 4, 4)``.
    """
    translations = np.asarray(translations)
    rotations = np.asarray(rotations)
    count = translations.shape[0]

    if scales is None:
        scales = np.ones((count, 3), dtype=translations.dtype)
    scales = np.asarray(scales)

    return np.stack([
        joint_matrix(translations[i], rotations[i], scales[i], dtype)
        for i in range(count)
    ])


@parameters_as_numpy_arrays('vertex')
def transform_vertex(vertex, matrix):
    """Transform a single vertex with a joint matrix.

    This is a thin wrapper over :func:`pyrr.matrix44.apply_to_vector` so
    that every stage of the chain goes through the same row-vector
    (``v @ M``) convention.

    :param numpy.array vertex: A vertex, shape ``(3,)`` or ``(4,)``.
    :param numpy.array matrix: A row-major matrix, shape ``(4, 4)``.
    :rtype: numpy.array
    :return: The transformed vertex with the same shape as the input.
    """
    return matrix44.apply_to_vector(matrix, vertex)


def skin_vertices(vertices, matrices, weights=None, indices=None):
    """Skin a mesh with one or more weighted joints.

    Two modes are supported:

    * **Rigid** (``weights`` and ``indices`` omitted): every vertex is
      transformed by ``matrices[0]``.
    * **Weighted** (both supplied): each vertex lists the joints it is
      bound to and a weight per joint. Weights are normalised to sum to
      one per vertex, and the skinned position is the weighted average of
      the positions obtained from each joint matrix. All averaging happens
      after transformation, in the same space, so the result is
      independent of how the weights happen to be scaled on input.

    :param numpy.array vertices: The bind-pose vertices, shape ``(v, 3)``.
    :param numpy.array matrices: Joint matrices, shape ``(j, 4, 4)``, as
        produced by :func:`joint_matrix` / :func:`joint_matrices`.
    :param numpy.array weights: Optional per-vertex weights, shape
        ``(v, k)``.
    :param numpy.array indices: Optional per-vertex joint indices, shape
        ``(v, k)``. Entries index into ``matrices``.
    :rtype: numpy.array
    :return: The skinned vertices, shape ``(v, 3)``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    matrices = np.asarray(matrices, dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (vertex_count, 3)")

    if weights is None or indices is None:
        if weights is not None or indices is not None:
            raise ValueError(
                "weights and indices must be supplied together"
            )
        # rigid skinning with the first joint matrix
        return matrix44.apply_to_vector(matrices[0], vertices)

    weights = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)

    if weights.shape != indices.shape:
        raise ValueError(
            "weights and indices must have the same shape (vertex_count, k)"
        )
    if weights.shape[0] != vertices.shape[0]:
        raise ValueError(
            "one weight / index row is required per vertex"
        )

    # normalise weights per vertex so callers may pass unnormalised values
    weight_sums = weights.sum(axis=1, keepdims=True)
    if np.any(weight_sums <= 0.0):
        raise ValueError("each vertex must have a positive total weight")
    normalised_weights = weights / weight_sums

    influence_count = weights.shape[1]
    accumulated = np.zeros_like(vertices)
    for influence in range(influence_count):
        joint_indices = indices[:, influence]
        influence_matrices = matrices[joint_indices]
        transformed = np.einsum(
            'vi,vij->vj',
            np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1),
            influence_matrices
        )[:, :3]
        accumulated += normalised_weights[:, influence, np.newaxis] * transformed

    return accumulated


@all_parameters_as_numpy_arrays
def pose_vertex(vertex, translation, rotation, scale=None):
    """Run the full chain on a single vertex.

    Equivalent to (and implemented as)::

        transform_vertex(vertex, joint_matrix(translation, rotation, scale))

    Provided as a convenience so callers can read the pipeline
    "interpolate -> rebuild -> transform" top to bottom in one call.

    :param numpy.array vertex: A vertex, shape ``(3,)``.
    :param numpy.array translation: The joint translation, shape ``(3,)``.
    :param numpy.array rotation: The joint rotation quaternion, shape
        ``(4,)``.
    :param numpy.array scale: Optional per-axis scale, shape ``(3,)``.
    :rtype: numpy.array
    :return: The transformed vertex, shape ``(3,)``.
    """
    matrix = joint_matrix(translation, rotation, scale)
    return transform_vertex(vertex, matrix)
