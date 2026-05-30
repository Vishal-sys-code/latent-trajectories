import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

class HiddenStateTrajectory:
    def __init__(
        self,
        prompt_id: int,
        prompt: str,
        model: str,
        embedding_state: torch.Tensor,
        trajectory: torch.Tensor,
        model_family: str = "unknown"
    ):
        """
        Initializes a HiddenStateTrajectory.

        Args:
            prompt_id: The unique identifier of the prompt.
            prompt: The text of the prompt.
            model: The name of the model.
            embedding_state: The initial embedding state before the transformer blocks (shape: [D]).
            trajectory: The hidden states for the transformer blocks (shape: [L, D]).
            model_family: The family of the model (optional).
        """
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.model = model
        self.model_family = model_family
        
        # Ensure tensors are on CPU for storage and standardized
        self.embedding_state = embedding_state.cpu().detach()
        self.trajectory = trajectory.cpu().detach()
        
        self.num_layers, self.hidden_dim = self.trajectory.shape

    def layer_distance(self) -> np.ndarray:
        """
        Computes the L2 distance between consecutive layers.
        
        Returns:
            np.ndarray of shape [L-1] containing the distances.
        """
        if self.num_layers < 2:
            return np.array([])
            
        distances = torch.norm(self.trajectory[1:] - self.trajectory[:-1], p=2, dim=1)
        return distances.numpy()

    def cosine_transition(self) -> np.ndarray:
        """
        Computes the cosine similarity between consecutive layers.
        
        Returns:
            np.ndarray of shape [L-1] containing the cosine similarities.
        """
        if self.num_layers < 2:
            return np.array([])
            
        cosines = F.cosine_similarity(self.trajectory[:-1], self.trajectory[1:], dim=1)
        return cosines.numpy()

    def save(self, filepath: str):
        """
        Saves the trajectory object and computed metrics to a standardized PyTorch file.
        
        Args:
            filepath: Path to save the file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "model": self.model,
            "model_family": self.model_family,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "embedding_state": self.embedding_state,
            "trajectory": self.trajectory,
            "layer_distances": self.layer_distance(),
            "layer_cosines": self.cosine_transition(),
        }
        
        torch.save(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HiddenStateTrajectory":
        """
        Loads a trajectory object from a saved file.
        
        Args:
            filepath: Path to the saved file.
            
        Returns:
            A new instance of HiddenStateTrajectory.
        """
        data = torch.load(filepath, weights_only=False)
        
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            model=data["model"],
            embedding_state=data["embedding_state"],
            trajectory=data["trajectory"],
            model_family=data.get("model_family", "unknown")
        )