#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Acceptance tests for the skeletal animation chain (pyrr.skin).

The same fixed pose sequence is used throughout:

    t = 0.0  identity orientation
    t = 1.0  +90 degrees around Y
    t = 2.0  +180 degrees around Y

with a fixed translation and scale per keyframe. Every number below is
derived from that sequence by hand using Pyrr's own low-level functions,
so a failure pinpoints which link of the chain broke its convention.
"""
try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
from pyrr import quaternion, matrix33, matrix44, skin


SQRT_HALF = np.sqrt(0.5)


class test_slerp_shortest_path(unittest.TestCase):
    def test_endpoints(self):
        q1 = quaternion.create_from_y_rotation(0.3)
        q2 = quaternion.create_from_y_rotation(1.2)
        np.testing.assert_allclose(
            skin.slerp_shortest_path(q1, q2, 0.0), q1, atol=1e-12
        )
        np.testing.assert_allclose(
            skin.slerp_shortest_path(q1, q2, 1.0), quaternion.normalize(q2),
            atol=1e-12
        )

    def test_t_clamped(self):
        q1 = quaternion.create_from_y_rotation(0.3)
        q2 = quaternion.create_from_y_rotation(1.2)
        np.testing.assert_allclose(
            skin.slerp_shortest_path(q1, q2, -5.0),
            skin.slerp_shortest_path(q1, q2, 0.0), atol=1e-12
        )
        np.testing.assert_allclose(
            skin.slerp_shortest_path(q1, q2, 5.0),
            skin.slerp_shortest_path(q1, q2, 1.0), atol=1e-12
        )

    def test_midpoint_of_fixed_sequence(self):
        # midpoint of identity and +90 deg around Y is +45 deg around Y
        q0 = quaternion.create()
        q1 = quaternion.create_from_y_rotation(np.pi / 2.0)
        result = skin.slerp_shortest_path(q0, q1, 0.5)
        expected = quaternion.create_from_y_rotation(np.pi / 4.0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_double_cover_gives_same_rotation(self):
        # -q encodes the same rotation; flipping the cover must not change
        # the geometric result of the interpolation
        q1 = quaternion.create_from_axis_rotation([0.3, 0.8, -0.2], 0.7)
        q2 = quaternion.create_from_axis_rotation([-0.5, 0.2, 1.1], 1.3)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            a = skin.slerp_shortest_path(q1, q2, t)
            b = skin.slerp_shortest_path(q1, -q2, t)
            np.testing.assert_allclose(
                matrix33.create_from_inverse_of_quaternion(a),
                matrix33.create_from_inverse_of_quaternion(b),
                atol=1e-12
            )
            # the two results must literally be the same cover as well
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_antipodal_keyframes(self):
        # exactly 180 degrees apart: the generic slerp formula is
        # undefined; we must still get a 90 degree rotation on the
        # shortest arc without NaNs
        q1 = quaternion.create_from_y_rotation(0.0)
        q2 = quaternion.create_from_y_rotation(np.pi)
        result = skin.slerp_shortest_path(q1, q2, 0.5)
        self.assertFalse(np.any(np.isnan(result)))
        np.testing.assert_allclose(
            quaternion.rotation_angle(result), np.pi / 2.0, atol=1e-12
        )
        # and it stays a rotation on the shortest (Y) axis
        np.testing.assert_allclose(
            np.abs(quaternion.rotation_axis(result)), [0.0, 1.0, 0.0],
            atol=1e-12
        )

    def test_antipodal_generic_axis(self):
        q1 = quaternion.create()
        q2 = quaternion.create_from_axis_rotation([1.0, 2.0, 3.0], np.pi)
        result = skin.slerp_shortest_path(q1, q2, 0.5)
        self.assertFalse(np.any(np.isnan(result)))
        np.testing.assert_allclose(
            quaternion.rotation_angle(result), np.pi / 2.0, atol=1e-12
        )

    def test_non_normalised_inputs(self):
        q1 = quaternion.create_from_y_rotation(0.4) * 3.0
        q2 = quaternion.create_from_y_rotation(1.1) * 0.25
        result = skin.slerp_shortest_path(q1, q2, 0.5)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    def test_matches_quaternion_slerp_for_normal_case(self):
        q1 = quaternion.create_from_y_rotation(0.2)
        q2 = quaternion.create_from_y_rotation(0.9)
        np.testing.assert_allclose(
            skin.slerp_shortest_path(q1, q2, 0.37),
            quaternion.slerp(q1, q2, 0.37), atol=1e-12
        )


class test_lerp_shortest_path(unittest.TestCase):
    def test_double_cover(self):
        q1 = quaternion.create_from_axis_rotation([1.0, 1.0, 0.0], 0.5)
        q2 = quaternion.create_from_axis_rotation([0.0, 1.0, 1.0], 1.0)
        a = skin.lerp_shortest_path(q1, q2, 0.5)
        b = skin.lerp_shortest_path(q1, -q2, 0.5)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(a), 1.0, atol=1e-12)

    def test_endpoints(self):
        q1 = quaternion.create_from_y_rotation(0.3)
        q2 = quaternion.create_from_y_rotation(1.2)
        np.testing.assert_allclose(
            skin.lerp_shortest_path(q1, q2, 0.0), q1, atol=1e-12
        )


class test_interpolate_keyframes(unittest.TestCase):
    # fixed pose sequence: identity -> +90 deg Y (t=1) -> +180 deg Y (t=2)
    TIMES = np.array([0.0, 1.0, 2.0])
    ROTATIONS = np.array([
        quaternion.create(),
        quaternion.create_from_y_rotation(np.pi / 2.0),
        quaternion.create_from_y_rotation(np.pi),
    ])

    def test_keyframe_times_return_keyframes(self):
        for index, time in enumerate(self.TIMES):
            np.testing.assert_allclose(
                skin.interpolate_keyframes(self.TIMES, self.ROTATIONS, time),
                quaternion.normalize(self.ROTATIONS[index]), atol=1e-12
            )

    def test_half_segment(self):
        # t = 0.5 must be +45 deg around Y
        result = skin.interpolate_keyframes(self.TIMES, self.ROTATIONS, 0.5)
        np.testing.assert_allclose(
            result, quaternion.create_from_y_rotation(np.pi / 4.0), atol=1e-12
        )
        # t = 1.5 must be +135 deg around Y
        result = skin.interpolate_keyframes(self.TIMES, self.ROTATIONS, 1.5)
        np.testing.assert_allclose(
            result, quaternion.create_from_y_rotation(3.0 * np.pi / 4.0),
            atol=1e-12
        )

    def test_clamped_extrapolation(self):
        np.testing.assert_allclose(
            skin.interpolate_keyframes(self.TIMES, self.ROTATIONS, -1.0),
            self.ROTATIONS[0], atol=1e-12
        )
        np.testing.assert_allclose(
            skin.interpolate_keyframes(self.TIMES, self.ROTATIONS, 3.0),
            self.ROTATIONS[-1], atol=1e-12
        )

    def test_mixed_covers_stay_shortest_path(self):
        # same geometric sequence but the middle keyframe is stored as -q
        rotations = self.ROTATIONS.copy()
        rotations[1] = -rotations[1]
        for t in (0.25, 0.5, 0.75, 1.25, 1.5, 1.75):
            canonical = skin.interpolate_keyframes(self.TIMES, self.ROTATIONS, t)
            covered = skin.interpolate_keyframes(self.TIMES, rotations, t)
            np.testing.assert_allclose(
                matrix33.create_from_inverse_of_quaternion(canonical),
                matrix33.create_from_inverse_of_quaternion(covered),
                atol=1e-12
            )

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            skin.interpolate_keyframes(np.array([0.0, 1.0]),
                                       self.ROTATIONS, 0.5)
        with self.assertRaises(ValueError):
            skin.interpolate_keyframes(self.TIMES,
                                       self.ROTATIONS[:, :3], 0.5)


class test_joint_matrix(unittest.TestCase):
    def test_default_scale_is_identity(self):
        translation = np.array([4.0, 5.0, 6.0])
        rotation = quaternion.create_from_y_rotation(0.7)
        result = skin.joint_matrix(translation, rotation)
        expected = np.dot(
            matrix44.create_from_translation(translation),
            matrix44.create_from_inverse_of_quaternion(rotation)
        )
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_double_cover_produces_identical_matrix(self):
        rotation = quaternion.create_from_axis_rotation([0.3, 0.8, -0.2], 1.1)
        translation = np.array([1.0, -2.0, 3.0])
        scale = np.array([2.0, 3.0, 4.0])
        m1 = skin.joint_matrix(translation, rotation, scale)
        m2 = skin.joint_matrix(translation, -rotation, scale)
        np.testing.assert_allclose(m1, m2, atol=1e-12)

    def test_scale_matrix_semantics(self):
        # the scale stage only touches the diagonal and never the
        # translation row; diagonal factors are the scale vector
        rotation = quaternion.create()
        translation = np.array([0.0, 0.0, 0.0])
        scale = np.array([2.0, 3.0, 4.0])
        matrix = skin.joint_matrix(translation, rotation, scale)
        np.testing.assert_allclose(
            np.diag(matrix), [2.0, 3.0, 4.0, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
        # and a unit vertex is scaled component-wise
        np.testing.assert_allclose(
            matrix44.apply_to_vector(matrix, [1.0, 1.0, 1.0]),
            [2.0, 3.0, 4.0], atol=1e-12
        )

    def test_translation_row_follows_row_vector_order(self):
        # unit scale: the last row holds the raw translation
        translation = np.array([1.5, -2.0, 0.5])
        unit = skin.joint_matrix(translation, quaternion.create())
        np.testing.assert_allclose(
            unit[3], [1.5, -2.0, 0.5, 1.0], atol=1e-12
        )

        # non-unit scale: v @ T @ R @ S means the translated position is
        # scaled too, so the last row is the per-axis-scaled translation
        scale = np.array([2.0, 3.0, 4.0])
        scaled = skin.joint_matrix(translation, quaternion.create(), scale)
        np.testing.assert_allclose(
            scaled[3], [3.0, -6.0, 2.0, 1.0], atol=1e-12
        )
        # the origin vertex lands on the scaled translation
        np.testing.assert_allclose(
            matrix44.apply_to_vector(scaled, [0.0, 0.0, 0.0]),
            translation * scale, atol=1e-12
        )

    def test_rotation_stage_matches_quaternion_apply_to_vector(self):
        # this is the critical sign-convention check: the matrix rebuilt
        # from a quaternion must rotate a point exactly like
        # quaternion.apply_to_vector
        rotation = quaternion.create_from_axis_rotation([0.4, -1.2, 0.7], 1.3)
        matrix = skin.joint_matrix(np.zeros(3), rotation)
        for vertex in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0], [0.3, -0.7, 1.2]):
            vertex = np.array(vertex)
            np.testing.assert_allclose(
                matrix44.apply_to_vector(matrix, vertex),
                quaternion.apply_to_vector(rotation, vertex), atol=1e-12
            )


class test_chain_composition(unittest.TestCase):
    """The whole chain must equal its hand-computed segments."""

    def setUp(self):
        self.rotation = quaternion.create_from_axis_rotation(
            [0.3, 0.8, -0.2], 1.1
        )
        self.translation = np.array([4.0, 5.0, 6.0])
        self.scale = np.array([2.0, 3.0, 4.0])
        self.vertex = np.array([0.7, -1.3, 2.1])

    def test_combined_matrix_equals_segment_chain(self):
        matrix = skin.joint_matrix(
            self.translation, self.rotation, self.scale
        )

        # hand-built segment chain, row-vector order: v @ T @ R @ S
        t_matrix = matrix44.create_from_translation(self.translation)
        r_matrix = matrix44.create_from_inverse_of_quaternion(self.rotation)
        s_matrix = matrix44.create_from_scale(self.scale)
        staged = self.vertex.copy()
        staged = matrix44.apply_to_vector(t_matrix, staged)
        staged = matrix44.apply_to_vector(r_matrix, staged)
        staged = matrix44.apply_to_vector(s_matrix, staged)

        combined = matrix44.apply_to_vector(matrix, self.vertex)
        np.testing.assert_allclose(combined, staged, atol=1e-12)

        # explicit T @ R @ S assembly
        hand_matrix = np.dot(np.dot(t_matrix, r_matrix), s_matrix)
        np.testing.assert_allclose(matrix, hand_matrix, atol=1e-12)

    def test_pose_vertex_equals_manual_quaternion_then_affine(self):
        # fully manual path using quaternion semantics directly:
        # translate, rotate with the quaternion, then scale
        translated = self.vertex + self.translation
        rotated = quaternion.apply_to_vector(self.rotation, translated)
        manual = rotated * self.scale

        result = skin.pose_vertex(
            self.vertex, self.translation, self.rotation, self.scale
        )
        np.testing.assert_allclose(result, manual, atol=1e-12)

        # transform_vertex on the rebuilt matrix gives the same answer
        matrix = skin.joint_matrix(
            self.translation, self.rotation, self.scale
        )
        np.testing.assert_allclose(
            skin.transform_vertex(self.vertex, matrix), manual, atol=1e-12
        )

    def test_full_pose_sequence_pipeline(self):
        """Fixed pose sequence end to end.

        For every sample time: interpolate the rotation, rebuild the
        matrix, transform the vertex -- and compare against a pipeline
        assembled by hand from individual pyrr calls.
        """
        times = np.array([0.0, 1.0, 2.0])
        rotations = np.array([
            quaternion.create(),
            quaternion.create_from_y_rotation(np.pi / 2.0),
            quaternion.create_from_y_rotation(np.pi),
        ])
        translation = np.array([1.0, 2.0, 3.0])
        scale = np.array([1.5, 2.0, 0.5])
        vertex = np.array([0.6, -0.4, 1.3])

        for sample_time in np.linspace(0.0, 2.0, 17):
            rotation = skin.interpolate_keyframes(times, rotations,
                                                  sample_time)
            matrix = skin.joint_matrix(translation, rotation, scale)
            result = skin.transform_vertex(vertex, matrix)

            # hand pipeline with the low-level functions
            staged = matrix44.apply_to_vector(
                matrix44.create_from_translation(translation), vertex
            )
            staged = quaternion.apply_to_vector(rotation, staged)
            staged = staged * scale
            np.testing.assert_allclose(result, staged, atol=1e-12)

    def test_pose_sequence_with_negated_keyframes_identical_vertices(self):
        # storing every second keyframe as -q must not move a single vertex
        times = np.array([0.0, 1.0, 2.0])
        rotations = np.array([
            quaternion.create(),
            quaternion.create_from_y_rotation(np.pi / 2.0),
            quaternion.create_from_y_rotation(np.pi),
        ])
        covered = rotations.copy()
        covered[1] = -covered[1]
        covered[2] = -covered[2]

        translation = np.array([1.0, 2.0, 3.0])
        scale = np.array([1.5, 2.0, 0.5])
        vertex = np.array([0.6, -0.4, 1.3])

        for sample_time in np.linspace(0.0, 2.0, 17):
            q_a = skin.interpolate_keyframes(times, rotations, sample_time)
            q_b = skin.interpolate_keyframes(times, covered, sample_time)
            v_a = skin.pose_vertex(vertex, translation, q_a, scale)
            v_b = skin.pose_vertex(vertex, translation, q_b, scale)
            np.testing.assert_allclose(v_a, v_b, atol=1e-12)


class test_skinning(unittest.TestCase):
    def setUp(self):
        # two joints with known, distinct matrices
        self.joint_a = skin.joint_matrix(
            np.array([1.0, 0.0, 0.0]),
            quaternion.create(),
            np.array([1.0, 1.0, 1.0])
        )
        self.joint_b = skin.joint_matrix(
            np.array([0.0, 2.0, 0.0]),
            quaternion.create_from_y_rotation(np.pi / 2.0),
            np.array([2.0, 2.0, 2.0])
        )
        self.matrices = np.stack([self.joint_a, self.joint_b])
        self.vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def test_rigid_skinning(self):
        result = skin.skin_vertices(self.vertices, self.matrices)
        expected = matrix44.apply_to_vector(self.joint_a, self.vertices)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_full_weight_matches_single_joint(self):
        # weight 1.0 on joint B must equal rigid skinning with joint B
        weights = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        indices = np.array([
            [0, 1],
            [0, 1],
            [0, 1],
        ])
        result = skin.skin_vertices(
            self.vertices, self.matrices, weights, indices
        )
        expected = matrix44.apply_to_vector(self.joint_b, self.vertices)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_half_weights_are_average_of_transforms(self):
        weights = np.full((3, 2), 0.5)
        indices = np.broadcast_to(np.array([0, 1]), (3, 2)).copy()
        result = skin.skin_vertices(
            self.vertices, self.matrices, weights, indices
        )
        a = matrix44.apply_to_vector(self.joint_a, self.vertices)
        b = matrix44.apply_to_vector(self.joint_b, self.vertices)
        np.testing.assert_allclose(result, 0.5 * (a + b), atol=1e-12)

    def test_unnormalised_weights_are_normalised(self):
        weights = np.full((3, 2), 3.0)
        indices = np.broadcast_to(np.array([0, 1]), (3, 2)).copy()
        result = skin.skin_vertices(
            self.vertices, self.matrices, weights, indices
        )
        a = matrix44.apply_to_vector(self.joint_a, self.vertices)
        b = matrix44.apply_to_vector(self.joint_b, self.vertices)
        np.testing.assert_allclose(result, 0.5 * (a + b), atol=1e-12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            # only one of weights / indices supplied
            skin.skin_vertices(self.vertices, self.matrices,
                               weights=np.ones((3, 2)))
        with self.assertRaises(ValueError):
            # mismatched shapes
            skin.skin_vertices(
                self.vertices, self.matrices,
                np.ones((3, 2)), np.ones((3, 3), dtype=np.int64)
            )
        with self.assertRaises(ValueError):
            # zero total weight
            skin.skin_vertices(
                self.vertices, self.matrices,
                np.zeros((3, 2)),
                np.zeros((3, 2), dtype=np.int64)
            )

    def test_skinned_chain_matches_manual_blend(self):
        # weighted skinning over matrices rebuilt from the pose sequence
        times = np.array([0.0, 1.0, 2.0])
        rotations = np.array([
            quaternion.create(),
            quaternion.create_from_y_rotation(np.pi / 2.0),
            quaternion.create_from_y_rotation(np.pi),
        ])
        sample_time = 1.25
        rotation = skin.interpolate_keyframes(times, rotations, sample_time)

        matrices = skin.joint_matrices(
            translations=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
            rotations=np.array([rotation, -rotation]),
            scales=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        )
        # the two joints differ only by translation and by q vs -q;
        # first check the double cover made their rotation blocks equal
        np.testing.assert_allclose(
            matrices[0][:3, :3], matrices[1][:3, :3], atol=1e-12
        )

        vertices = np.array([[0.5, 0.5, 0.5], [1.0, -1.0, 2.0]])
        weights = np.array([[0.75, 0.25], [0.4, 0.6]])
        indices = np.array([[0, 1], [0, 1]])
        result = skin.skin_vertices(vertices, matrices, weights, indices)

        manual = np.zeros_like(vertices)
        for vertex_index in range(vertices.shape[0]):
            blended = np.zeros(3)
            total = weights[vertex_index].sum()
            for influence in range(2):
                transformed = matrix44.apply_to_vector(
                    matrices[indices[vertex_index, influence]],
                    vertices[vertex_index]
                )
                blended += (weights[vertex_index, influence] / total) * transformed
            manual[vertex_index] = blended
        np.testing.assert_allclose(result, manual, atol=1e-12)


class test_existing_functions_unchanged(unittest.TestCase):
    def test_quaternion_slerp_still_present(self):
        q1 = quaternion.create()
        q2 = quaternion.create_from_y_rotation(np.pi / 2.0)
        result = quaternion.slerp(q1, q2, 0.5)
        np.testing.assert_allclose(
            result, quaternion.create_from_y_rotation(np.pi / 4.0), atol=1e-6
        )

    def test_matrix_builders_keep_their_semantics(self):
        q = quaternion.create_from_x_rotation(np.pi / 2.0)
        # create_from_quaternion keeps its existing (transpose) semantics
        expected_from_quaternion = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_allclose(
            matrix33.create_from_quaternion(q),
            expected_from_quaternion, atol=1e-12
        )
        # create_from_inverse_of_quaternion keeps its existing semantics,
        # which is the one consistent with quaternion.apply_to_vector
        np.testing.assert_allclose(
            matrix33.create_from_inverse_of_quaternion(q),
            expected_from_quaternion.T, atol=1e-12
        )


if __name__ == '__main__':
    unittest.main()
