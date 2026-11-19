try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
from pyrr import camera, matrix44


class test_camera(unittest.TestCase):
    def test_import(self):
        import pyrr
        pyrr.camera
        from pyrr import camera

    # fixed camera used by the acceptance tests
    eye = [0., 0., 5.]
    target = [0., 0., 0.]
    up = [0., 1., 0.]
    fovy = 60.
    aspect = 4. / 3.
    near = 0.1
    far = 100.
    viewport = (0., 0., 800., 600.)

    def test_create_viewport_matrix(self):
        result = camera.create_viewport_matrix(0., 0., 800., 600.)
        expected = np.array([
            [400., 0., 0., 0.],
            [0., 300., 0., 0.],
            [0., 0., 0.5, 0.],
            [400., 300., 0.5, 1.],
        ])
        np.testing.assert_almost_equal(result, expected, decimal=5)
        self.assertTrue(result.dtype == float)

    def test_create_viewport_matrix_offset_and_depth(self):
        result = camera.create_viewport_matrix(10., 20., 640., 480., min_depth=-1., max_depth=1.)
        expected = np.array([
            [320., 0., 0., 0.],
            [0., 240., 0., 0.],
            [0., 0., 1., 0.],
            [330., 260., 0., 1.],
        ])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_create_viewport_matrix_invalid(self):
        self.assertRaises(ValueError, camera.create_viewport_matrix, 0., 0., 0., 600.)
        self.assertRaises(ValueError, camera.create_viewport_matrix, 0., 0., 800., -1.)
        self.assertRaises(ValueError, camera.create_viewport_matrix, 0., 0., 800., 600., 1., 1.)

    def test_create_look_at_perspective_fixed(self):
        # acceptance against fixed camera matrices
        result = camera.create_look_at_perspective(
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
        )
        expected = np.array([
            [1.2990381057, 0., 0., 0.],
            [0., 1.7320508076, 0., 0.],
            [0., 0., -1.002002002, -1.],
            [0., 0., 4.8098098098, 5.],
        ])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_create_look_at_perspective_viewport_fixed(self):
        # acceptance against fixed camera matrices
        result = camera.create_look_at_perspective_viewport(
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )
        expected = np.array([
            [5.1961524227e+02, 0., 0., 0.],
            [0., 5.1961524227e+02, 0., 0.],
            [-4.0000000000e+02, -3.0000000000e+02, -1.0010010010e+00, -1.],
            [2.0000000000e+03, 1.5000000000e+03, 4.9049049049e+00, 5.],
        ])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_create_look_at_perspective_viewport_fixed_2(self):
        # a second fixed camera, off-axis, to guard against
        # coincidentally symmetric expectations
        result = camera.create_look_at_perspective_viewport(
            [3., 4., 5.], [0., 1., 0.], [0., 1., 0.],
            45., 16. / 9., 1., 50.,
            (10., 20., 640., 480.),
        )
        expected = np.array([
            [2.2165720453e+02, -2.5533047512e+02, -4.6683235816e-01, -4.5749571100e-01],
            [-1.5097358463e+02, 3.9627045613e+02, -4.6683235816e-01, -4.5749571100e-01],
            [-4.7520111455e+02, -4.2555079187e+02, -7.7805393027e-01, -7.6249285166e-01],
            [2.3149282976e+03, 1.3086635602e+03, 6.1376879952e+00, 7.0149342353e+00],
        ])
        np.testing.assert_almost_equal(result, expected, decimal=4)

    def test_create_look_at_perspective_bit_identical(self):
        # composed call must be bit-for-bit identical to the segments
        result = camera.create_look_at_perspective(
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
        )
        view = matrix44.create_look_at(self.eye, self.target, self.up)
        projection = matrix44.create_perspective_projection(
            self.fovy, self.aspect, self.near, self.far
        )
        expected = matrix44.multiply(view, projection)
        self.assertTrue(np.array_equal(result, expected))

    def test_create_look_at_perspective_viewport_bit_identical(self):
        # composed call must be bit-for-bit identical to the segments
        result = camera.create_look_at_perspective_viewport(
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )
        view = matrix44.create_look_at(self.eye, self.target, self.up)
        projection = matrix44.create_perspective_projection(
            self.fovy, self.aspect, self.near, self.far
        )
        viewport = camera.create_viewport_matrix(*self.viewport)
        expected = matrix44.multiply(matrix44.multiply(view, projection), viewport)
        self.assertTrue(np.array_equal(result, expected))

    def test_create_look_at_perspective_dtype(self):
        result = camera.create_look_at_perspective(
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            dtype='float32',
        )
        self.assertTrue(result.dtype == np.float32)

    def test_create_look_at_perspective_viewport_dtype(self):
        result = camera.create_look_at_perspective_viewport(
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
            dtype='float32',
        )
        self.assertTrue(result.dtype == np.float32)

    def test_degenerate_eye_equals_target(self):
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            [1., 2., 3.], [1., 2., 3.], [0., 1., 0.],
            self.fovy, self.aspect, self.near, self.far,
        )

    def test_degenerate_up_parallel_to_view_direction(self):
        # up parallel to the view direction must raise, not return NaN
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            [0., 0., 5.], [0., 0., 0.], [0., 0., -1.],
            self.fovy, self.aspect, self.near, self.far,
        )
        # anti-parallel must also raise
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective_viewport,
            [0., 0., 5.], [0., 0., 0.], [0., 0., 2.],
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )

    def test_degenerate_up_zero(self):
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, [0., 0., 0.],
            self.fovy, self.aspect, self.near, self.far,
        )

    def test_degenerate_near_not_positive(self):
        # negative near must raise, not silently produce NaN
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            self.fovy, self.aspect, -0.1, self.far,
        )
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            self.fovy, self.aspect, 0., self.far,
        )

    def test_degenerate_far_not_greater_than_near(self):
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.near,
        )
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            self.fovy, self.aspect, 10., 1.,
        )

    def test_degenerate_fovy(self):
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            0., self.aspect, self.near, self.far,
        )
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            180., self.aspect, self.near, self.far,
        )

    def test_degenerate_aspect(self):
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective,
            self.eye, self.target, self.up,
            self.fovy, 0., self.near, self.far,
        )

    def test_degenerate_viewport(self):
        self.assertRaises(
            ValueError,
            camera.create_look_at_perspective_viewport,
            self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            (0., 0., 0., 600.),
        )

    def test_segment_functions_unchanged(self):
        # the existing segment functions must keep their previous
        # behaviour: no validation, degenerate input silently
        # produces NaN instead of raising
        with np.errstate(all='ignore'):
            m = matrix44.create_perspective_projection(self.fovy, self.aspect, 0., self.far)
            self.assertTrue(np.isnan(m).any())
            m = matrix44.create_perspective_projection(self.fovy, self.aspect, -0.1, self.far)
            self.assertFalse(np.isnan(m).any())
            m = matrix44.create_look_at(self.eye, self.target, [0., 0., -1.])
            self.assertTrue(np.isnan(m).any())

    def test_world_to_screen_fixed(self):
        # the target projects to the centre of the viewport
        result = camera.world_to_screen(
            self.target, self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )
        np.testing.assert_almost_equal(result, [400., 300., 0.980980981], decimal=5)

    def test_world_to_screen_fixed_point(self):
        result = camera.world_to_screen(
            [1., 1., 0.], self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )
        np.testing.assert_almost_equal(
            result, [503.9230484541, 403.9230484541, 0.980980981], decimal=5
        )

    def test_world_to_screen_batch(self):
        result = camera.world_to_screen(
            [[0., 1., 0.], [1., 2., 3.]],
            [3., 4., 5.], [0., 1., 0.], [0., 1., 0.],
            45., 16. / 9., 1., 50.,
            (10., 20., 640., 480.),
        )
        expected = np.array([
            [330., 260., 0.8647973772],
            [241.1453275569, 169.6652632337, 0.7162598087],
        ])
        np.testing.assert_almost_equal(result, expected, decimal=4)

    def test_world_to_screen_matches_segments(self):
        # composed call must match applying the segment matrices manually
        points = np.array([[1., 2., 3.], [-4., 0.5, 2.], [0., 0., 0.]])
        result = camera.world_to_screen(
            points, self.eye, self.target, self.up,
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )
        view = matrix44.create_look_at(self.eye, self.target, self.up)
        projection = matrix44.create_perspective_projection(
            self.fovy, self.aspect, self.near, self.far
        )
        viewport = camera.create_viewport_matrix(*self.viewport)
        m = matrix44.multiply(matrix44.multiply(view, projection), viewport)
        expected = matrix44.apply_to_vector(m, points)
        self.assertTrue(np.array_equal(result, expected))

    def test_world_to_screen_degenerate(self):
        self.assertRaises(
            ValueError,
            camera.world_to_screen,
            [0., 0., 0.], self.eye, self.target, [0., 0., -1.],
            self.fovy, self.aspect, self.near, self.far,
            self.viewport,
        )


if __name__ == '__main__':
    unittest.main()
