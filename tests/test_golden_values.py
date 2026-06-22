"""
Golden-value regression tests.

These tests lock in the exact numeric output of every metric function
against hand-computed or pre-validated reference values. Any refactor
that silently changes the math will break these tests.
"""
import pytest
import torch
import numpy as np
from latent_trajectories.trajectories import HiddenStateTrajectory
from latent_trajectories.metrics import (
    compute_trajectory_length,
    compute_curvature,
    compute_layer_velocity,
    compute_convergence_matrix,
    compute_convergence_score,
    compute_per_prompt_convergence_score,
    compute_layerwise_silhouette,
    compute_rsa_matrix,
)


def _make_traj(trajectory_data, prompt_id=0):
    """Helper to create a HiddenStateTrajectory from raw tensor."""
    t = torch.tensor(trajectory_data, dtype=torch.float32)
    return HiddenStateTrajectory(
        prompt_id=prompt_id,
        prompt="test",
        model="test",
        embedding_state=torch.zeros(t.shape[1]),
        trajectory=t,
    )


class TestTrajectoryLength:
    """Lock in trajectory length computation."""

    def test_straight_line(self):
        # Straight line along x-axis: [0,0] -> [1,0] -> [2,0]
        # Raw length = 1.0 + 1.0 = 2.0
        traj = _make_traj([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        lengths = compute_trajectory_length([traj], normalized=False)
        assert np.isclose(lengths[0], 2.0, atol=1e-6)

    def test_diagonal(self):
        # Diagonal: [0,0] -> [1,1] -> [2,2]
        # Raw length = sqrt(2) + sqrt(2) = 2*sqrt(2)
        traj = _make_traj([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        lengths = compute_trajectory_length([traj], normalized=False)
        assert np.isclose(lengths[0], 2 * np.sqrt(2), atol=1e-6)

    def test_normalized(self):
        # After L2 normalization, [1,0] and [2,0] both become [1,0],
        # so normalized distance between them is 0.
        traj = _make_traj([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        lengths = compute_trajectory_length([traj], normalized=True)
        assert np.isclose(lengths[0], 0.0, atol=1e-6)

    def test_orthogonal_steps_normalized(self):
        # [1,0] -> [0,1]: both unit vectors, distance = sqrt(2)
        # [0,1] -> [1,0]: distance = sqrt(2)
        # Total = 2*sqrt(2)
        traj = _make_traj([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        lengths = compute_trajectory_length([traj], normalized=True)
        assert np.isclose(lengths[0], 2 * np.sqrt(2), atol=1e-6)


class TestCurvature:
    """Lock in curvature computation."""

    def test_straight_line_zero_curvature(self):
        # Perfectly straight trajectory: curvature = 0
        traj = _make_traj([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        curvatures = compute_curvature([traj])
        assert np.isclose(curvatures[0], 0.0, atol=1e-5)

    def test_right_angle(self):
        # 90-degree turn: [0,0] -> [1,0] -> [1,1]
        # v1 = [1,0], v2 = [0,1], angle = pi/2
        traj = _make_traj([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        curvatures = compute_curvature([traj])
        assert np.isclose(curvatures[0], np.pi / 2, atol=1e-5)

    def test_reversal(self):
        # 180-degree turn: [0,0] -> [1,0] -> [0,0]
        # v1 = [1,0], v2 = [-1,0], angle = pi
        traj = _make_traj([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        curvatures = compute_curvature([traj])
        assert np.isclose(curvatures[0], np.pi, atol=1e-5)

    def test_too_few_layers(self):
        traj = _make_traj([[0.0, 0.0], [1.0, 0.0]])
        curvatures = compute_curvature([traj])
        assert np.isclose(curvatures[0], 0.0)


class TestLayerVelocity:
    """Lock in layer velocity computation."""

    def test_uniform_velocity(self):
        t1 = _make_traj([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        t2 = _make_traj([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        velocities = compute_layer_velocity([t1, t2])
        assert np.allclose(velocities, [1.0, 1.0], atol=1e-6)


class TestConvergenceMatrix:
    """Lock in convergence matrix computation."""

    def test_convergence_pair(self):
        # Two trajectories that converge
        t1 = _make_traj([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        t2 = _make_traj([[10.0, 0.0], [5.5, 0.0], [2.0, 0.0]])
        matrix = compute_convergence_matrix([t1, t2])
        # d_init = |0 - 10| = 10, d_final = |1 - 2| = 1
        # convergence = 10 - 1 = 9
        assert np.isclose(matrix[0, 1], 9.0, atol=1e-5)
        assert np.isclose(matrix[1, 0], 9.0, atol=1e-5)


class TestRSAMatrix:
    """Lock in RSA matrix properties."""

    def test_diagonal_is_one(self):
        # Need ≥3 trajectories for meaningful Spearman correlation on RDMs
        t1 = _make_traj([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        t2 = _make_traj([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        t3 = _make_traj([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
        rsa = compute_rsa_matrix([t1, t2, t3])
        for i in range(rsa.shape[0]):
            assert np.isclose(rsa[i, i], 1.0, atol=1e-5)

    def test_symmetry(self):
        # Need ≥3 trajectories so RDM upper-triangle has >1 element
        # (Spearman on single-element vectors returns NaN)
        t1 = _make_traj([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        t2 = _make_traj([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        t3 = _make_traj([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
        rsa = compute_rsa_matrix([t1, t2, t3])
        assert np.allclose(rsa, rsa.T, atol=1e-5)


class TestStatsGoldenValues:
    """Lock in statistical function outputs."""

    def test_cohens_d_perfect_separation(self):
        from latent_trajectories.stats import cohens_d
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        d = cohens_d(a, b)
        # Pooled SD = sqrt(2.5), diff = -5, d = -5/sqrt(2.5) ≈ -3.162
        assert np.isclose(d, -5.0 / np.sqrt(2.5), atol=1e-3)

    def test_permutation_separable(self):
        from latent_trajectories.stats import permutation_test
        # Use 5 samples per group and 10k permutations for sufficient power
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        p = permutation_test(a, b, num_permutations=10000, random_seed=42)
        assert p < 0.05

    def test_bootstrap_ci_contains_mean(self):
        from latent_trajectories.stats import bootstrap_ci
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = bootstrap_ci(data, num_bootstraps=1000, random_seed=42)
        assert lo <= np.mean(data) <= hi
