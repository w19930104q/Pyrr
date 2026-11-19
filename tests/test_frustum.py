try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
from pyrr import frustum, matrix44


class test_frustum(unittest.TestCase):
    def test_import(self):
        import pyrr
        pyrr.frustum
        from pyrr import frustum

    def setUp(self):
        # camera at the origin looking down -Z
        # fovy 90 degrees, aspect 1, near 1, far 10
        self.projection = matrix44.create_perspective_projection(90.0, 1.0, 1.0, 10.0)
        self.planes = frustum.create_from_matrix(self.projection)

    def test_create_from_matrix_shape(self):
        self.assertEqual(self.planes.shape, (6, 4))

    def test_create_from_matrix_dtype(self):
        planes = frustum.create_from_matrix(self.projection, dtype=np.float32)
        self.assertEqual(planes.dtype, np.float32)

    def test_create_from_matrix_normals_normalized(self):
        lengths = np.linalg.norm(self.planes[:, :3], axis=1)
        np.testing.assert_almost_equal(lengths, np.ones(6))

    def test_create_from_matrix_orthogonal(self):
        projection = matrix44.create_orthogonal_projection(-5.0, 5.0, -5.0, 5.0, 1.0, 10.0)
        planes = frustum.create_from_matrix(projection)
        self.assertTrue(frustum.intersects_point(planes, [0.0, 0.0, -5.0]))
        self.assertTrue(frustum.intersects_point(planes, [4.9, 4.9, -5.0]))
        self.assertFalse(frustum.intersects_point(planes, [5.1, 0.0, -5.0]))
        self.assertFalse(frustum.intersects_point(planes, [0.0, 0.0, -11.0]))
        self.assertFalse(frustum.intersects_point(planes, [0.0, 0.0, 1.0]))

    def test_intersects_point_inside(self):
        self.assertTrue(frustum.intersects_point(self.planes, [0.0, 0.0, -5.0]))
        self.assertTrue(frustum.intersects_point(self.planes, [4.0, 4.0, -5.0]))
        self.assertTrue(frustum.intersects_point(self.planes, [-4.0, -4.0, -9.0]))

    def test_intersects_point_outside(self):
        # behind the camera
        self.assertFalse(frustum.intersects_point(self.planes, [0.0, 0.0, 5.0]))
        # in front of the near plane
        self.assertFalse(frustum.intersects_point(self.planes, [0.0, 0.0, -0.5]))
        # beyond the far plane
        self.assertFalse(frustum.intersects_point(self.planes, [0.0, 0.0, -50.0]))
        # outside the sides
        self.assertFalse(frustum.intersects_point(self.planes, [10.0, 0.0, -5.0]))
        self.assertFalse(frustum.intersects_point(self.planes, [0.0, -10.0, -5.0]))

    def test_intersects_point_with_view_matrix(self):
        # camera at (0, 0, 10) looking at the origin
        view = matrix44.create_look_at(
            np.array([0.0, 0.0, 10.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        )
        planes = frustum.create_from_matrix(np.dot(view, self.projection))
        self.assertTrue(frustum.intersects_point(planes, [0.0, 0.0, 0.0]))
        self.assertTrue(frustum.intersects_point(planes, [0.0, 0.0, 5.0]))
        # behind the camera
        self.assertFalse(frustum.intersects_point(planes, [0.0, 0.0, 20.0]))
        # closer than the near plane
        self.assertFalse(frustum.intersects_point(planes, [0.0, 0.0, 9.5]))

    def test_intersects_aabb(self):
        # fully inside
        self.assertTrue(frustum.intersects_aabb(
            self.planes, [[-1.0, -1.0, -5.0], [1.0, 1.0, -3.0]]))
        # fully outside
        self.assertFalse(frustum.intersects_aabb(
            self.planes, [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]]))
        # straddling the near plane
        self.assertTrue(frustum.intersects_aabb(
            self.planes, [[-0.5, -0.5, -2.0], [0.5, 0.5, 0.5]]))
        # straddling the left plane
        self.assertTrue(frustum.intersects_aabb(
            self.planes, [[-10.0, -1.0, -5.0], [-4.0, 1.0, -3.0]]))
        # outside the left plane
        self.assertFalse(frustum.intersects_aabb(
            self.planes, [[-10.0, -1.0, -5.0], [-6.0, 1.0, -3.0]]))
        # beyond the far plane
        self.assertFalse(frustum.intersects_aabb(
            self.planes, [[-1.0, -1.0, -20.0], [1.0, 1.0, -11.0]]))

    def test_intersects_sphere(self):
        # fully inside
        self.assertTrue(frustum.intersects_sphere(self.planes, [0.0, 0.0, -5.0, 1.0]))
        # fully outside
        self.assertFalse(frustum.intersects_sphere(self.planes, [0.0, 0.0, 5.0, 1.0]))
        # touching the near plane
        self.assertTrue(frustum.intersects_sphere(self.planes, [0.0, 0.0, -0.5, 1.0]))
        # outside the right plane
        self.assertFalse(frustum.intersects_sphere(self.planes, [100.0, 0.0, -5.0, 1.0]))
