# -*- coding: utf-8 -*-
"""Provides functions to create and test view frustums.

A frustum is represented by 6 planes (left, right, bottom, top, near, far).
Each plane is a numpy.array of shape (4,) as used by :mod:`pyrr.plane`,
with the plane normals pointing toward the **inside** of the frustum.

A point is inside the frustum if it is in front of (or on) all 6 planes,
ie. ``point_height_above_plane( point, plane ) >= 0.0`` for every plane.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .utils import all_parameters_as_numpy_arrays, parameters_as_numpy_arrays


class index:
    #: The index of the left plane within the frustum
    left = 0

    #: The index of the right plane within the frustum
    right = 1

    #: The index of the bottom plane within the frustum
    bottom = 2

    #: The index of the top plane within the frustum
    top = 3

    #: The index of the near plane within the frustum
    near = 4

    #: The index of the far plane within the frustum
    far = 5


@parameters_as_numpy_arrays('matrix')
def create_from_matrix(matrix, dtype=None):
    """Creates a frustum from a view-projection matrix.

    The matrix should be the combined view and projection matrix as used
    by Pyrr (row-major, vectors multiplied as ``vector * matrix``), ie::

        view_projection = numpy.dot( view, projection )

    The planes are extracted using the Gribb-Hartmann method. Because Pyrr
    multiplies vectors from the left, the planes are combinations of the
    matrix columns. The clip space is the OpenGL style one used by
    :func:`pyrr.matrix44.create_perspective_projection`
    (``-w <= x, y, z <= w``).

    :param numpy.array matrix: The combined view-projection matrix, shape (4, 4).
    :rtype: numpy.array
    :return: The 6 frustum planes, shape (6, 4), normals pointing inward.
        Planes are ordered as per the :class:`index` class.
    """
    dtype = dtype or matrix.dtype

    # columns of the matrix
    c0 = matrix[:, 0]
    c1 = matrix[:, 1]
    c2 = matrix[:, 2]
    c3 = matrix[:, 3]

    # each plane is of the form a*x + b*y + c*z + d >= 0 for points
    # inside the frustum
    planes = np.array(
        [
            c3 + c0,    # left
            c3 - c0,    # right
            c3 + c1,    # bottom
            c3 - c1,    # top
            c3 + c2,    # near
            c3 - c2,    # far
        ],
        dtype=dtype
    )

    # pyrr planes store [A, B, C, D] with n.p - D = 0 on the plane,
    # so D = -d
    planes[:, 3] = -planes[:, 3]

    # normalise the planes so distances are in world units
    lengths = np.linalg.norm(planes[:, :3], axis=1)
    planes /= lengths[:, np.newaxis]

    return planes

@all_parameters_as_numpy_arrays
def intersects_point(planes, point):
    """Checks if a point is inside the frustum.

    :param numpy.array planes: The frustum planes, shape (6, 4).
    :param numpy.array point: The point to test, shape (3,).
    :rtype: boolean
    :return: True if the point is inside or on the edge of the frustum.
    """
    distances = np.sum(planes[:, :3] * point, axis=1) - planes[:, 3]
    return bool(np.all(distances >= 0.0))

@all_parameters_as_numpy_arrays
def intersects_sphere(planes, sphere):
    """Checks if a sphere intersects the frustum.

    :param numpy.array planes: The frustum planes, shape (6, 4).
    :param numpy.array sphere: The sphere to test, shape (4,) with the
        first 3 values being the centre and the last being the radius.
    :rtype: boolean
    :return: True if the sphere touches or is inside the frustum.
    """
    distances = np.sum(planes[:, :3] * sphere[:3], axis=1) - planes[:, 3]
    return bool(np.all(distances >= -sphere[3]))

@all_parameters_as_numpy_arrays
def intersects_aabb(planes, aabb):
    """Checks if an AABB intersects the frustum.

    Uses the p-vertex test: for each plane, the corner of the AABB
    farthest along the plane normal is tested. If that corner is behind
    any plane, the AABB is entirely outside the frustum.

    :param numpy.array planes: The frustum planes, shape (6, 4).
    :param numpy.array aabb: The AABB to test, shape (2, 3).
    :rtype: boolean
    :return: True if the AABB touches or is inside the frustum.
    """
    normals = planes[:, :3]

    # the p-vertex is the corner farthest along each plane normal
    p_vertices = np.where(normals >= 0.0, aabb[1], aabb[0])
    distances = np.sum(normals * p_vertices, axis=1) - planes[:, 3]
    return bool(np.all(distances >= 0.0))
