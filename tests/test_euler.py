try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
from pyrr import euler, quaternion


def quaternion_distance(q1, q2):
    """Sign-invariant distance between two quaternions."""
    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))


class test_euler(unittest.TestCase):
    def test_import(self):
        import pyrr
        pyrr.euler
        from pyrr import euler

    def test_create(self):
        self.assertTrue(np.array_equal(euler.create(), [0., 0., 0.]))
        e = euler.create(roll=1., pitch=2., yaw=3.)
        self.assertEqual(euler.roll(e), 1.)
        self.assertEqual(euler.pitch(e), 2.)
        self.assertEqual(euler.yaw(e), 3.)
        self.assertTrue(np.array_equal(e, [1., 2., 3.]))

    def test_create_from_quaternion_regular_poses(self):
        # euler -> quaternion -> euler recovers canonical eulers
        # (pitch kept away from the poles)
        rng = np.random.default_rng(0)
        for _ in range(1000):
            e = np.array([
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi / 2. + 1e-3, np.pi / 2. - 1e-3),
                rng.uniform(-np.pi, np.pi),
            ])
            result = euler.create_from_quaternion(quaternion.create_from_eulers(e))
            np.testing.assert_allclose(result, e, atol=1e-10)

    def test_create_from_quaternion_known_values(self):
        # single axis rotations
        e = euler.create_from_x_rotation(0.5)
        np.testing.assert_allclose(
            euler.create_from_quaternion(quaternion.create_from_eulers(e)), e)
        e = euler.create_from_y_rotation(0.5)
        np.testing.assert_allclose(
            euler.create_from_quaternion(quaternion.create_from_eulers(e)), e)
        e = euler.create_from_z_rotation(0.5)
        np.testing.assert_allclose(
            euler.create_from_quaternion(quaternion.create_from_eulers(e)), e)

    def test_create_from_quaternion_gimbal_lock(self):
        # at exact gimbal lock the branch is deterministic:
        # pitch snapped to +/- pi/2, yaw == 0, combined rotation in roll
        for pitch in (np.pi / 2., -np.pi / 2.):
            for roll, yaw in [(0.7, -1.2), (-2.0, 0.3), (3.0, 2.9), (0., 0.)]:
                e = np.array([roll, pitch, yaw])
                q = quaternion.create_from_eulers(e)
                result = euler.create_from_quaternion(q)
                self.assertEqual(euler.yaw(result), 0.)
                self.assertEqual(euler.pitch(result), np.copysign(np.pi / 2., pitch))
                # the round trip preserves the rotation exactly
                q2 = quaternion.create_from_eulers(result)
                self.assertLess(quaternion_distance(q, q2), 1e-12)
                # determinism: identical input -> identical output
                np.testing.assert_array_equal(
                    result, euler.create_from_quaternion(q))

    def test_create_from_quaternion_near_gimbal_lock(self):
        # round trip error stays bounded around the poles
        rng = np.random.default_rng(1)
        for pole in (np.pi / 2., -np.pi / 2.):
            for _ in range(1000):
                e = np.array([
                    rng.uniform(-np.pi, np.pi),
                    pole + rng.uniform(-1e-6, 1e-6),
                    rng.uniform(-np.pi, np.pi),
                ])
                q = quaternion.create_from_eulers(e)
                result = euler.create_from_quaternion(q)
                self.assertFalse(np.any(np.isnan(result)))
                q2 = quaternion.create_from_eulers(result)
                self.assertLess(quaternion_distance(q, q2), 1e-6)

    def test_create_from_quaternion_sign_invariant(self):
        # q and -q are the same rotation and must give identical eulers
        rng = np.random.default_rng(2)
        for _ in range(1000):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            np.testing.assert_array_equal(
                euler.create_from_quaternion(q),
                euler.create_from_quaternion(-q))

    def test_create_from_quaternion_full_pose_sampling(self):
        # sampling the whole rotation space: no NaN, no round trip jump
        rng = np.random.default_rng(3)
        for _ in range(10000):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            result = euler.create_from_quaternion(q)
            self.assertFalse(np.any(np.isnan(result)))
            q2 = quaternion.create_from_eulers(result)
            self.assertLess(quaternion_distance(q, q2), 1e-10)

    def test_create_from_quaternion_no_jump_across_lock(self):
        # sweeping pitch through the pole must not jump the rotation
        pitches = np.pi / 2. + np.linspace(-1e-6, 1e-6, 501)
        quats = []
        for pitch in pitches:
            e = euler.create_from_quaternion(
                quaternion.create_from_eulers([0.7, pitch, -1.2]))
            quats.append(quaternion.create_from_eulers(e))
        quats = np.array(quats)
        steps = np.linalg.norm(np.diff(quats, axis=0), axis=1)
        self.assertLess(steps.max(), 1e-6)

    def test_create_from_quaternion_non_unit(self):
        q = quaternion.create_from_eulers([0.3, 0.5, -0.7])
        result = euler.create_from_quaternion(q * 2.5)
        q2 = quaternion.create_from_eulers(result)
        self.assertLess(quaternion_distance(q, q2), 1e-12)

    def test_create_from_quaternion_dtype(self):
        q = quaternion.create_from_eulers([0.3, 0.5, -0.7]).astype(np.float32)
        result = euler.create_from_quaternion(q)
        self.assertEqual(result.dtype, np.float32)
        q2 = quaternion.create_from_eulers(result)
        self.assertLess(quaternion_distance(q, q2), 1e-5)


if __name__ == '__main__':
    unittest.main()
