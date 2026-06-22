import pytest
import torch
import numpy as np
from latent_trajectories.trajectories import HiddenStateTrajectory


@pytest.fixture
def mock_trajectories():
    """Create a set of mock trajectories with known geometry for testing."""
    traj1 = HiddenStateTrajectory(
        prompt_id=1,
        prompt="The cat sat on the mat",
        model="test_model",
        embedding_state=torch.zeros(4),
        trajectory=torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ]),
    )
    traj2 = HiddenStateTrajectory(
        prompt_id=2,
        prompt="A dog runs in the park",
        model="test_model",
        embedding_state=torch.zeros(4),
        trajectory=torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0, 0.0],
        ]),
    )
    traj3 = HiddenStateTrajectory(
        prompt_id=3,
        prompt="What is 2 + 2?",
        model="test_model",
        embedding_state=torch.ones(4),
        trajectory=torch.tensor([
            [5.0, 5.0, 0.0, 0.0],
            [6.0, 5.0, 0.0, 0.0],
            [7.0, 5.0, 0.0, 0.0],
        ]),
    )
    traj4 = HiddenStateTrajectory(
        prompt_id=4,
        prompt="Explain quantum mechanics",
        model="test_model",
        embedding_state=torch.ones(4),
        trajectory=torch.tensor([
            [5.0, 6.0, 0.0, 0.0],
            [6.0, 6.0, 0.0, 0.0],
            [7.0, 5.5, 0.0, 0.0],
        ]),
    )
    return [traj1, traj2, traj3, traj4]


@pytest.fixture
def mock_labels():
    """Labels corresponding to mock_trajectories fixture."""
    return ["animals", "animals", "reasoning", "reasoning"]
