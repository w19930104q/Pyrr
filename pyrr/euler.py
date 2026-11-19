# -*- coding: utf-8 -*-
"""Provide functions for the creation and manipulation of Euler angles.

Eulers represent 3 rotations: Pitch, Roll and Yaw.

Eulers are represented using a numpy.array of shape (3,).
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .utils import parameters_as_numpy_arrays


class index:
    """Defines the indices used to store the Euler values in the numpy array.
    """
    #: The index of the roll value within the euler.
    roll = 0

    #: The index of the pitch value within the euler.
    pitch = 1

    #: The index of the yaw value within the euler.
    yaw = 2


def create(roll=0., pitch=0., yaw=0., dtype=None):
    """Creates an array storing the specified euler angles.

    Input values are in radians.

    :param float pitch: The pitch in radians.
    :param float roll: The roll in radians.
    :param float yaw: The yaw in radians.
    :rtype: numpy.array
    """
    return np.array((roll, pitch, yaw), dtype=dtype)


def create_from_x_rotation(theta, dtype=None):
    return np.array([theta, 0., 0.], dtype=dtype)


def create_from_y_rotation(theta, dtype=None):
    return np.array([0., theta, 0.], dtype=dtype)


def create_from_z_rotation(theta, dtype=None):
    return np.array([0., 0., theta], dtype=dtype)


@parameters_as_numpy_arrays('quat')
def create_from_quaternion(quat, dtype=None):
    """Creates euler angles from a quaternion.

    This is the exact inverse of
    :py:func:`pyrr.quaternion.create_from_eulers`, which produces the
    rotation ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``.

    The result is deterministic:

    * ``quat`` and ``-quat`` (the same rotation) yield identical eulers.
    * Pitch is always in the range [-pi/2, pi/2]; the arcsine argument
      is clamped to [-1, 1] so numerical noise at the poles cannot
      produce NaN.
    * At gimbal lock (pitch = +/- pi/2), roll and yaw are degenerate —
      only their combined rotation is observable. The deterministic
      choice is ``yaw = 0`` with the entire remaining rotation folded
      into roll.

    This makes quaternion -> euler -> quaternion round-trips stable,
    including at and around gimbal lock.

    :param numpy.array quat: The quaternion to convert (shape 4).
    :rtype: numpy.array
    :return: The euler angles in the format [roll, pitch, yaw].
    """
    dtype = dtype or quat.dtype

    # the quaternion must be normalized
    # (the elements below are quadratic in the components, so the
    # result is invariant under quat -> -quat)
    norm = np.linalg.norm(quat)
    if not np.isclose(norm, 1.):
        quat = quat / norm

    qx, qy, qz, qw = quat

    # elements of the active rotation matrix
    # R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    # sin(pitch) = R[0,2]; clamp so |sin| slightly above 1.0 near the
    # poles (from floating point noise) cannot produce NaN
    sin_pitch = np.clip(r02, -1.0, 1.0)

    # gimbal lock threshold: cos(pitch)**2 at or below a small multiple
    # of machine epsilon. At exact lock, floating point noise in the
    # quaternion leaves cos(pitch)**2 as large as ~6*eps, so 16*eps
    # gives a safe, deterministic margin.
    try:
        tol = 16.0 * np.finfo(dtype).eps
    except (ValueError, TypeError):
        tol = 16.0 * np.finfo(np.float64).eps

    if 1.0 - sin_pitch * sin_pitch > tol:
        # regular pose
        pitch = np.arcsin(sin_pitch)
        roll = np.arctan2(-r12, r22)
        yaw = np.arctan2(-r01, r00)
    else:
        # gimbal lock: pitch = +/- pi/2, cos(pitch) = 0.
        # roll and yaw cannot be separated; deterministically set
        # yaw = 0 and fold the combined rotation into roll:
        #   pitch = +pi/2 -> roll = atan2( R[1,0], R[1,1])
        #   pitch = -pi/2 -> roll = atan2(-R[1,0], R[1,1])
        # pitch is snapped to exactly +/- pi/2: the quaternion cannot
        # resolve it more finely than the lock threshold anyway.
        pitch = np.copysign(np.pi / 2.0, sin_pitch)
        yaw = 0.0
        roll = np.arctan2(sin_pitch * r10, r11)

    return np.array([roll, pitch, yaw], dtype=dtype)


def roll(eulers):
    """Extracts the roll value from the euler.

    :rtype: float.
    """
    return eulers[0]


def yaw(eulers):
    """Extracts the yaw value from the euler.

    :rtype: float.
    """
    return eulers[2]


def pitch(eulers):
    """Extracts the pitch value from the euler.

    :rtype: float.
    """
    return eulers[1]
