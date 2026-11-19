# -*- coding: utf-8 -*-
"""Provides a Bounding Volume Hierarchy (BVH) for accelerating
intersection queries against large numbers of AABBs.

Testing every object in a scene individually is O(n) per query.
A BVH organises the objects into a binary tree of nested AABBs, so
whole subtrees can be rejected with a single test. Queries such as
frustum culling then run in roughly O(log n + k) time, where k is the
number of objects actually reported.

The hierarchy is built once from a set of AABBs and is static
afterwards. If the objects move, rebuild the hierarchy.

Queries only ever *prune* nodes whose bounds are entirely outside the
query volume. Because a node's bounds exactly contain the bounds of all
of its descendants, a pruned node is guaranteed to contain no hits, and
every candidate leaf object is re-tested with the same single-object
functions used by a naive per-object loop. Query results are therefore
identical to testing every object individually.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from . import frustum, geometric_tests


class BoundingVolumeHierarchy(object):
    """A static bounding volume hierarchy over a set of AABBs.

    :param numpy.array aabbs: The object AABBs, shape (n, 2, 3).
    :param int leaf_size: The maximum number of objects stored in a
        leaf node.
    """

    def __init__(self, aabbs, leaf_size=8):
        aabbs = np.asarray(aabbs)
        if aabbs.size == 0:
            aabbs = aabbs.reshape((0, 2, 3))
        if aabbs.ndim != 3 or aabbs.shape[1:] != (2, 3):
            raise ValueError('aabbs must have shape (n, 2, 3)')

        self._aabbs = aabbs
        self._leaf_size = max(1, int(leaf_size))

        count = aabbs.shape[0]
        # object indices, reordered during construction so that the
        # objects of each leaf form a contiguous range
        self._indices = np.arange(count, dtype=np.intp)

        # node storage, converted to arrays after construction
        bounds = []
        left = []
        right = []
        leaf_start = []
        leaf_count = []

        if count > 0:
            centroids = (aabbs[:, 0] + aabbs[:, 1]) * 0.5
            indices = self._indices
            leaf_size = self._leaf_size

            def build(start, end):
                node = len(bounds)
                left.append(-1)
                right.append(-1)
                leaf_start.append(0)
                leaf_count.append(0)

                boxes = aabbs[indices[start:end]]
                node_bounds = np.array(
                    [
                        boxes[:, 0].min(axis=0),
                        boxes[:, 1].max(axis=0),
                    ],
                    dtype=aabbs.dtype
                )
                bounds.append(node_bounds)

                if end - start <= leaf_size:
                    leaf_start[node] = start
                    leaf_count[node] = end - start
                    return node

                # split at the median centroid along the longest axis
                axis = int(np.argmax(node_bounds[1] - node_bounds[0]))
                mid = (start + end) // 2
                keys = centroids[indices[start:end], axis]
                order = np.argpartition(keys, mid - start)
                indices[start:end] = indices[start:end][order]

                left[node] = build(start, mid)
                right[node] = build(mid, end)
                return node

            build(0, count)

        dtype = aabbs.dtype
        self._bounds = np.array(bounds, dtype=dtype).reshape((-1, 2, 3))
        self._left = np.array(left, dtype=np.intp)
        self._right = np.array(right, dtype=np.intp)
        self._leaf_start = np.array(leaf_start, dtype=np.intp)
        self._leaf_count = np.array(leaf_count, dtype=np.intp)

    def __len__(self):
        """The number of objects in the hierarchy.
        """
        return self._aabbs.shape[0]

    @property
    def aabbs(self):
        """The AABBs the hierarchy was built from, shape (n, 2, 3).
        """
        return self._aabbs

    @property
    def bounds(self):
        """The AABB of the root node, encompassing all objects.

        Returns None if the hierarchy is empty.
        """
        if self._bounds.shape[0] == 0:
            return None
        return self._bounds[0].copy()

    def query_frustum(self, planes):
        """Finds all objects whose AABB intersects the frustum.

        The result is identical to testing every object with
        :func:`pyrr.frustum.intersects_aabb`, but subtrees that lie
        entirely outside the frustum are skipped.

        :param numpy.array planes: The frustum planes, shape (6, 4),
            as created by :func:`pyrr.frustum.create_from_matrix`.
        :rtype: list
        :return: A sorted list of the indices of the intersecting objects.
        """
        planes = np.asarray(planes)
        normals = planes[:, :3]
        offsets = planes[:, 3]

        result = []
        if self._bounds.shape[0] == 0:
            return result

        stack = [0]
        while stack:
            node = stack.pop()
            node_bounds = self._bounds[node]

            # p-vertex test, equivalent to frustum.intersects_aabb
            p_vertices = np.where(normals >= 0.0, node_bounds[1], node_bounds[0])
            distances = np.sum(normals * p_vertices, axis=1) - offsets
            if np.any(distances < 0.0):
                continue

            count = self._leaf_count[node]
            if count:
                start = self._leaf_start[node]
                for i in range(start, start + count):
                    obj = self._indices[i]
                    if frustum.intersects_aabb(planes, self._aabbs[obj]):
                        result.append(int(obj))
            else:
                stack.append(self._left[node])
                stack.append(self._right[node])

        result.sort()
        return result

    def query_aabb(self, aabb):
        """Finds all objects whose AABB overlaps the given AABB.

        The result is identical to testing every object with
        :func:`pyrr.geometric_tests.aabb_intersect_aabb`.

        :param numpy.array aabb: The query AABB, shape (2, 3).
        :rtype: list
        :return: A sorted list of the indices of the overlapping objects.
        """
        aabb = np.asarray(aabb)

        result = []
        if self._bounds.shape[0] == 0:
            return result

        stack = [0]
        while stack:
            node = stack.pop()
            node_bounds = self._bounds[node]

            if np.any(node_bounds[0] > aabb[1]) or np.any(node_bounds[1] < aabb[0]):
                continue

            count = self._leaf_count[node]
            if count:
                start = self._leaf_start[node]
                for i in range(start, start + count):
                    obj = self._indices[i]
                    if geometric_tests.aabb_intersect_aabb(aabb, self._aabbs[obj]):
                        result.append(int(obj))
            else:
                stack.append(self._left[node])
                stack.append(self._right[node])

        result.sort()
        return result

    def query_ray(self, ray):
        """Finds all objects whose AABB is intersected by the ray.

        The result is identical to testing every object with
        :func:`pyrr.geometric_tests.ray_intersect_aabb`.

        :param numpy.array ray: The query ray, shape (2, 3).
        :rtype: list
        :return: A sorted list of the indices of the intersected objects.
        """
        ray = np.asarray(ray)

        result = []
        if self._bounds.shape[0] == 0:
            return result

        stack = [0]
        while stack:
            node = stack.pop()
            if geometric_tests.ray_intersect_aabb(ray, self._bounds[node]) is None:
                continue

            count = self._leaf_count[node]
            if count:
                start = self._leaf_start[node]
                for i in range(start, start + count):
                    obj = self._indices[i]
                    if geometric_tests.ray_intersect_aabb(ray, self._aabbs[obj]) is not None:
                        result.append(int(obj))
            else:
                stack.append(self._left[node])
                stack.append(self._right[node])

        result.sort()
        return result


def create_from_aabbs(aabbs, leaf_size=8):
    """Creates a bounding volume hierarchy from a set of AABBs.

    :param numpy.array aabbs: The object AABBs, shape (n, 2, 3).
    :param int leaf_size: The maximum number of objects stored in a
        leaf node.
    :rtype: BoundingVolumeHierarchy
    """
    return BoundingVolumeHierarchy(aabbs, leaf_size=leaf_size)


#: Short alias for BoundingVolumeHierarchy.
BVH = BoundingVolumeHierarchy
