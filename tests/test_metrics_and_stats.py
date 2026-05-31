import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import numpy as np
import torch
from trajectories import HiddenStateTrajectory
from metrics import (
    compute_trajectory_length,
    compute_curvature,
    compute_layer_velocity,
    compute_convergence_matrix,
    compute_layerwise_silhouette,
    compute_rsa_matrix
)
from stats import (
    bootstrap_ci,
    permutation_test,
    cohens_d
)

@pytest.fixture
def mock_trajectories():
    traj1 = HiddenStateTrajectory(
        prompt_id=1,
        prompt="cat",
        model="test",
        embedding_state=torch.zeros(4),
        trajectory=torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0]
        ])
    )
    
    traj2 = HiddenStateTrajectory(
        prompt_id=2,
        prompt="dog",
        model="test",
        embedding_state=torch.zeros(4),
        trajectory=torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0, 0.0]
        ])
    )
    
    return [traj1, traj2]

def test_compute_trajectory_length(mock_trajectories):
    lengths = compute_trajectory_length(mock_trajectories)
    assert len(lengths) == 2
    # traj1 length: dist between layers is 1.0 each, so total length = 2.0
    assert np.isclose(lengths[0], 2.0)
    # traj2 length: layer1->2 dist is 1.0, layer2->3 dist is sqrt(1^2 + 0.5^2) = 1.118
    assert np.isclose(lengths[1], 1.0 + np.sqrt(1.25))

def test_compute_curvature(mock_trajectories):
    curvatures = compute_curvature(mock_trajectories)
    assert len(curvatures) == 2
    # traj1 moves in a straight line, curvature = 0
    assert np.isclose(curvatures[0], 0.0, atol=1e-5)
    # traj2 has a bend, curvature should be > 0
    assert curvatures[1] > 0.0

def test_stats():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    
    # Cohen's d should be large and negative (since mean(A) < mean(B))
    d = cohens_d(a, b)
    assert d < -1.0
    
    # Permutation test should give a small p-value
    p = permutation_test(a, b, num_permutations=1000)
    assert p < 0.05
    
    # Bootstrap CI
    ci_low, ci_high = bootstrap_ci(a, num_bootstraps=100, random_seed=42)
    assert ci_low <= np.mean(a) <= ci_high