# -*- coding: utf-8 -*-
"""Camera pipeline helpers.

Composes the three stages required to go from camera parameters to
screen coordinates into a single entry point:

#. the view matrix (:func:`pyrr.matrix44.create_look_at`),
#. the perspective projection matrix
   (:func:`pyrr.matrix44.create_perspective_projection`),
#. the viewport transform (:func:`create_viewport_matrix`).

The composed functions call the existing segment functions internally,
so their results are bit-for-bit identical to calling the segments
manually. Unlike the segment functions, the composed entry points
validate their inputs and raise :class:`ValueError` for degenerate
configurations (for example an ``up`` vector parallel to the view
direction, or a non-positive ``near`` plane) instead of silently
producing NaN matrices.

The behaviour of the individual segment functions in
:mod:`pyrr.matrix44` is unchanged.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from . import matrix44

#: Tolerance used to detect an up vector (nearly) parallel to the
#: view direction. The magnitude of the cross product of the
#: normalised vectors is compared against this value.
_PARALLEL_TOLERANCE = 1e-12


def create_viewport_matrix(x, y, width, height, min_depth=0.0, max_depth=1.0, dtype=None):
    """Creates a viewport transform matrix.

    Maps normalised device coordinates (the output of the perspective
    division, in the range [-1, 1]) to window coordinates, equivalent
    to OpenGL's ``glViewport`` / ``glDepthRange``.

    :param float x: The left edge of the viewport in window coordinates.
    :param float y: The bottom edge of the viewport in window coordinates.
    :param float width: The width of the viewport. Must be positive.
    :param float height: The height of the viewport. Must be positive.
    :param float min_depth: The window coordinate depth that NDC -1 maps to.
    :param float max_depth: The window coordinate depth that NDC 1 maps to.
    :rtype: numpy.array
    :return: A 4x4 matrix mapping NDC to window coordinates.
    :raises ValueError: If width or height is not positive, or
        min_depth == max_depth.
    """
    if not width > 0:
        raise ValueError("viewport width must be positive")
    if not height > 0:
        raise ValueError("viewport height must be positive")
    if min_depth == max_depth:
        raise ValueError("viewport min_depth and max_depth must differ")

    return np.array((
        (width / 2.0, 0., 0., 0.),
        (0., height / 2.0, 0., 0.),
        (0., 0., (max_depth - min_depth) / 2.0, 0.),
        (x + width / 2.0, y + height / 2.0, (max_depth + min_depth) / 2.0, 1.),
    ), dtype=dtype)


def _validate_camera_parameters(eye, target, up, fovy, aspect, near, far):
    """Validates camera parameters for the composed pipeline functions.

    Raises ValueError for degenerate inputs that would otherwise
    silently produce NaN or inf matrices.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward_length = np.linalg.norm(forward)
    if forward_length == 0.0:
        raise ValueError("eye and target must not be the same point")

    up_length = np.linalg.norm(up)
    if up_length == 0.0:
        raise ValueError("up vector must be non-zero")

    parallel = np.linalg.norm(np.cross(forward / forward_length, up / up_length))
    if parallel <= _PARALLEL_TOLERANCE:
        raise ValueError("up vector must not be parallel to the view direction")

    if not 0.0 < fovy < 180.0:
        raise ValueError("fovy must be in the range (0, 180) degrees")
    if aspect == 0.0:
        raise ValueError("aspect ratio must be non-zero")
    if not near > 0.0:
        raise ValueError("near plane distance must be positive")
    if not far > near:
        raise ValueError("far plane distance must be greater than near")


def create_look_at_perspective(eye, target, up, fovy, aspect, near, far, dtype=None):
    """Creates a combined view-projection matrix.

    Equivalent to::

        matrix44.multiply(
            matrix44.create_look_at(eye, target, up, dtype=dtype),
            matrix44.create_perspective_projection(fovy, aspect, near, far, dtype=dtype),
        )

    The result is bit-for-bit identical to performing the two calls
    manually. Degenerate inputs raise :class:`ValueError` instead of
    silently producing NaN values.

    :param numpy.array eye: Position of the camera in world coordinates.
    :param numpy.array target: The position in world coordinates that the
        camera is looking at.
    :param numpy.array up: The up vector of the camera.
    :param float fovy: Field of view in y direction in degrees.
        Must be in the range (0, 180).
    :param float aspect: Aspect ratio of the view (width / height).
        Must be non-zero.
    :param float near: Distance from the viewer to the near clipping
        plane. Must be positive.
    :param float far: Distance from the viewer to the far clipping
        plane. Must be greater than near.
    :rtype: numpy.array
    :return: A 4x4 view-projection matrix.
    :raises ValueError: If the parameters are degenerate.
    """
    _validate_camera_parameters(eye, target, up, fovy, aspect, near, far)
    view = matrix44.create_look_at(eye, target, up, dtype=dtype)
    projection = matrix44.create_perspective_projection(fovy, aspect, near, far, dtype=dtype)
    return matrix44.multiply(view, projection)


def create_look_at_perspective_viewport(eye, target, up, fovy, aspect, near, far, viewport, dtype=None):
    """Creates a combined view-projection-viewport matrix.

    Equivalent to::

        matrix44.multiply(
            matrix44.multiply(
                matrix44.create_look_at(eye, target, up, dtype=dtype),
                matrix44.create_perspective_projection(fovy, aspect, near, far, dtype=dtype),
            ),
            camera.create_viewport_matrix(*viewport, dtype=dtype),
        )

    The result is bit-for-bit identical to performing the three calls
    manually. Degenerate inputs raise :class:`ValueError` instead of
    silently producing NaN values.

    Applying this matrix to a homogeneous world coordinate and
    dividing by ``w`` yields window coordinates (see
    :func:`world_to_screen`).

    :param numpy.array eye: Position of the camera in world coordinates.
    :param numpy.array target: The position in world coordinates that the
        camera is looking at.
    :param numpy.array up: The up vector of the camera.
    :param float fovy: Field of view in y direction in degrees.
        Must be in the range (0, 180).
    :param float aspect: Aspect ratio of the view (width / height).
        Must be non-zero.
    :param float near: Distance from the viewer to the near clipping
        plane. Must be positive.
    :param float far: Distance from the viewer to the far clipping
        plane. Must be greater than near.
    :param viewport: A sequence of ``(x, y, width, height)`` describing
        the viewport in window coordinates. Width and height must be
        positive.
    :rtype: numpy.array
    :return: A 4x4 view-projection-viewport matrix.
    :raises ValueError: If the parameters are degenerate.
    """
    view_projection = create_look_at_perspective(
        eye, target, up, fovy, aspect, near, far, dtype=dtype
    )
    viewport_matrix = create_viewport_matrix(*viewport, dtype=dtype)
    return matrix44.multiply(view_projection, viewport_matrix)


def world_to_screen(points, eye, target, up, fovy, aspect, near, far, viewport, dtype=None):
    """Transforms world coordinates to screen (window) coordinates.

    Applies the full camera pipeline (look at, perspective projection,
    perspective division and viewport transform) to one or more
    points. This is the composed equivalent of::

        m = camera.create_look_at_perspective_viewport(
            eye, target, up, fovy, aspect, near, far, viewport, dtype=dtype)
        matrix44.apply_to_vector(m, points)

    :param numpy.array points: A vector3, or an Nx3 array of vectors,
        in world coordinates.
    :param numpy.array eye: Position of the camera in world coordinates.
    :param numpy.array target: The position in world coordinates that the
        camera is looking at.
    :param numpy.array up: The up vector of the camera.
    :param float fovy: Field of view in y direction in degrees.
        Must be in the range (0, 180).
    :param float aspect: Aspect ratio of the view (width / height).
        Must be non-zero.
    :param float near: Distance from the viewer to the near clipping
        plane. Must be positive.
    :param float far: Distance from the viewer to the far clipping
        plane. Must be greater than near.
    :param viewport: A sequence of ``(x, y, width, height)`` describing
        the viewport in window coordinates. Width and height must be
        positive.
    :rtype: numpy.array
    :return: The screen coordinates ``(x, y, depth)`` of each point,
        where ``depth`` is the window coordinate depth.
    :raises ValueError: If the parameters are degenerate.
    """
    m = create_look_at_perspective_viewport(
        eye, target, up, fovy, aspect, near, far, viewport, dtype=dtype
    )
    return matrix44.apply_to_vector(m, np.asarray(points))
