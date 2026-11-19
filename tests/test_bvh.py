try:
    import unittest2 as unittest
except:
    import unittest
import time
import numpy as np
from pyrr import bvh, frustum, geometric_tests, matrix44


def random_scene(seed, count):
    """Creates a random scene of AABBs, including some degenerate ones.
    """
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-100.0, 100.0, (count, 3))
    sizes = rng.uniform(0.0, 4.0, (count, 3))
    return np.stack([centres - sizes, centres + sizes], axis=1)


def random_frustum(seed):
    """Creates a frustum from a random view-projection matrix.
    """
    rng = np.random.RandomState(seed)
    eye = rng.uniform(-150.0, 150.0, 3)
    target = rng.uniform(-50.0, 50.0, 3)
    up = np.array([0.0, 1.0, 0.0])
    # avoid a degenerate look-at
    if np.abs(np.dot(eye - target, up)) > 0.99 * np.linalg.norm(eye - target):
        up = np.array([1.0, 0.0, 0.0])
    view = matrix44.create_look_at(eye, target, up)
    projection = matrix44.create_perspective_projection(
        rng.uniform(30.0, 90.0), rng.uniform(0.5, 2.0), 1.0, 500.0)
    return frustum.create_from_matrix(np.dot(view, projection))


class test_bvh(unittest.TestCase):
    def test_import(self):
        import pyrr
        pyrr.bvh
        from pyrr import bvh

    def test_empty(self):
        tree = bvh.BoundingVolumeHierarchy(np.zeros((0, 2, 3)))
        self.assertEqual(len(tree), 0)
        self.assertIsNone(tree.bounds)
        self.assertEqual(tree.query_frustum(random_frustum(0)), [])
        self.assertEqual(tree.query_aabb([[-1, -1, -1], [1, 1, 1]]), [])
        self.assertEqual(tree.query_ray([[0, 0, 10], [0, 0, -1]]), [])

    def test_single_object(self):
        boxes = np.array([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
        tree = bvh.BoundingVolumeHierarchy(boxes)
        self.assertEqual(len(tree), 1)
        np.testing.assert_equal(tree.bounds, boxes[0])
        self.assertEqual(tree.query_aabb([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]), [0])
        self.assertEqual(tree.query_aabb([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]), [])

    def test_invalid_shape(self):
        self.assertRaises(ValueError, bvh.BoundingVolumeHierarchy, np.zeros((4, 3)))

    def test_create_from_aabbs(self):
        boxes = random_scene(1, 10)
        tree = bvh.create_from_aabbs(boxes)
        self.assertIsInstance(tree, bvh.BoundingVolumeHierarchy)
        self.assertEqual(len(tree), 10)

    def test_bounds_contains_all_objects(self):
        boxes = random_scene(2, 100)
        tree = bvh.BoundingVolumeHierarchy(boxes)
        self.assertTrue(np.all(tree.bounds[0] <= boxes[:, 0].min(axis=0)))
        self.assertTrue(np.all(tree.bounds[1] >= boxes[:, 1].max(axis=0)))

    def test_query_frustum_matches_per_object_tests(self):
        # the culled result must be identical to testing every object
        for scene_seed, count in [(3, 1), (4, 7), (5, 200), (6, 2000)]:
            boxes = random_scene(scene_seed, count)
            for leaf_size in (1, 4, 8):
                tree = bvh.BoundingVolumeHierarchy(boxes, leaf_size=leaf_size)
                for frustum_seed in range(5):
                    planes = random_frustum(frustum_seed)
                    result = tree.query_frustum(planes)
                    expected = [
                        i for i in range(count)
                        if frustum.intersects_aabb(planes, boxes[i])
                    ]
                    self.assertEqual(result, expected)

    def test_query_aabb_matches_per_object_tests(self):
        boxes = random_scene(7, 1000)
        tree = bvh.BoundingVolumeHierarchy(boxes)
        rng = np.random.RandomState(8)
        for _ in range(10):
            centre = rng.uniform(-100.0, 100.0, 3)
            size = rng.uniform(0.0, 30.0, 3)
            query = np.array([centre - size, centre + size])
            result = tree.query_aabb(query)
            expected = [
                i for i in range(len(boxes))
                if geometric_tests.aabb_intersect_aabb(query, boxes[i])
            ]
            self.assertEqual(result, expected)

    def test_query_ray_matches_per_object_tests(self):
        boxes = random_scene(9, 1000)
        tree = bvh.BoundingVolumeHierarchy(boxes)
        rng = np.random.RandomState(10)
        for _ in range(10):
            origin = rng.uniform(-100.0, 100.0, 3)
            direction = rng.uniform(-1.0, 1.0, 3)
            ray = np.array([origin, direction])
            result = tree.query_ray(ray)
            expected = [
                i for i in range(len(boxes))
                if geometric_tests.ray_intersect_aabb(ray, boxes[i]) is not None
            ]
            self.assertEqual(result, expected)

    def test_query_frustum_performance(self):
        # a frustum that culls most of the scene: the hierarchy query
        # must return the same result as the per-object tests while
        # being significantly faster
        count = 40000
        rng = np.random.RandomState(11)
        centres = rng.uniform(-500.0, 500.0, (count, 3))
        sizes = rng.uniform(0.5, 2.0, (count, 3))
        boxes = np.stack([centres - sizes, centres + sizes], axis=1)

        view = matrix44.create_look_at(
            np.array([0.0, 0.0, 200.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]))
        projection = matrix44.create_perspective_projection(30.0, 1.0, 1.0, 300.0)
        planes = frustum.create_from_matrix(np.dot(view, projection))

        tree = bvh.BoundingVolumeHierarchy(boxes)

        start = time.perf_counter()
        expected = [
            i for i in range(count)
            if frustum.intersects_aabb(planes, boxes[i])
        ]
        per_object_time = time.perf_counter() - start

        # best of a few runs to avoid timer noise
        result = None
        query_time = min(
            self._time_query(tree, planes)
            for _ in range(5)
        )
        result = test_bvh._last_result

        self.assertEqual(result, expected)
        self.assertLess(query_time * 3.0, per_object_time)

    @staticmethod
    def _time_query(tree, planes):
        start = time.perf_counter()
        result = tree.query_frustum(planes)
        elapsed = time.perf_counter() - start
        test_bvh._last_result = result
        return elapsed

    _last_result = None
