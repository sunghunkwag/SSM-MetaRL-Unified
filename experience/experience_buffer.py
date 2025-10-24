"""
PyTorch-based Experience Buffer for Experience-Augmented Adaptation.

This module provides a simple circular buffer to store and sample
(observation, target) pairs as Tensors, inspired by the NumPy-based
ExperienceBuffer from the 'lownocompute-ai-baseline' repository.
"""

import torch
from collections import deque
import random
from typing import Tuple, List, Optional


class ExperienceBuffer:
    """
    A PyTorch-based circular memory buffer to store past experiences.

    This buffer stores experiences as (observation, target) tensors, enabling
    the Adapter to sample them during test-time adaptation.
    """

    def __init__(self, max_size: int = 10000, device: str = "cpu"):
        """
        Initialize the experience buffer.

        Args:
            max_size: Maximum number of experiences (tensor pairs) to store.
                      When full, oldest experiences are automatically removed.
            device: The torch device to store tensors on (e.g., 'cpu', 'cuda').
        """
        # deque with maxlen automatically drops oldest items
        self.buffer = deque(maxlen=max_size)
        self.device = torch.device(device)
        self.max_size = max_size

    def add(self, observations: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Add a batch of experiences to the buffer.

        Args:
            observations: A tensor of observations (B, *obs_shape).
            targets: A tensor of corresponding targets (B, *target_shape).
        """
        if observations.shape[0] != targets.shape[0]:
            raise ValueError("Observations and targets must have the same batch size.")

        # Move to the designated device and detach from any graph
        observations = observations.detach().to(self.device)
        targets = targets.detach().to(self.device)

        # Add individual (obs, target) pairs to the deque
        for i in range(observations.shape[0]):
            self.buffer.append((observations[i], targets[i]))

    def get_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a random batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            A tuple of (observations, targets) tensors, stacked into a batch.
            Returns None if the buffer is empty.
        """
        if len(self.buffer) == 0:
            return None

        actual_size = min(int(batch_size), len(self.buffer))

        # Randomly sample from the deque
        samples = random.sample(list(self.buffer), actual_size)

        try:
            # Unzip list of tuples into two lists
            obs_list, target_list = zip(*samples)

            # Stack tensors to create a batch
            obs_batch = torch.stack(obs_list)
            target_batch = torch.stack(target_list)

            return obs_batch, target_batch

        except Exception as e:
            print(f"Error stacking tensors in ExperienceBuffer: {e}")
            return None

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)
