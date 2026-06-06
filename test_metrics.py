import torch
import numpy as np
from src.trajectories import HiddenStateTrajectory
from src.metrics import compute_per_prompt_convergence_score, compute_convergence_score

t1 = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0]
])

t2 = torch.tensor([
    [2.0, 0.0],
    [0.0, 2.0],
    [-2.0, 0.0]
])

t3 = torch.tensor([
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0]
])

t4 = torch.tensor([
    [0.0, 2.0],
    [-2.0, 0.0],
    [0.0, -2.0]
])


traj1 = HiddenStateTrajectory(
    prompt_id=1, prompt="", model="", embedding_state=torch.zeros(2), trajectory=t1
)
traj2 = HiddenStateTrajectory(
    prompt_id=2, prompt="", model="", embedding_state=torch.zeros(2), trajectory=t2
)
traj3 = HiddenStateTrajectory(
    prompt_id=3, prompt="", model="", embedding_state=torch.zeros(2), trajectory=t3
)
traj4 = HiddenStateTrajectory(
    prompt_id=4, prompt="", model="", embedding_state=torch.zeros(2), trajectory=t4
)


labels = ["cat1", "cat1", "cat2", "cat2"]
scores = compute_convergence_score([traj1, traj2, traj3, traj4], labels)
print("Scores:", scores)