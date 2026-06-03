import torch
import numpy as np
from src.trajectories import HiddenStateTrajectory
from src.metrics import compute_trajectory_length

t = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0]
])

traj = HiddenStateTrajectory(
    trajectory=t,
    metadata={"id": "test"}
)

l_raw = compute_trajectory_length([traj], normalized=False)
l_norm = compute_trajectory_length([traj], normalized=True)

print("Raw:", l_raw)
print("Norm:", l_norm)
