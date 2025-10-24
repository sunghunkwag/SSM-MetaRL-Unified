"""
Hybrid Test-Time Adaptation Module with Experience Replay

This module provides an enhanced adaptation strategy that combines
current task data with past experiences stored in an ExperienceBuffer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from experience.experience_buffer import ExperienceBuffer


@dataclass
class HybridAdaptationConfig:
    """Configuration for the Hybrid Adapter with experience replay."""
    learning_rate: float = 0.01
    num_steps: int = 5
    grad_clip_norm: Optional[float] = 1.0
    
    # Experience replay specific parameters
    experience_batch_size: int = 32
    experience_weight: float = 0.1


class HybridAdapter:
    """
    Performs test-time adaptation using a hybrid loss function.
    
    This adapter leverages an ExperienceBuffer to combine the loss from
    the current task with a loss from sampled past experiences, enabling
    more robust adaptation through experience-augmented learning.
    """

    def __init__(
        self,
        model: nn.Module,
        config: HybridAdaptationConfig,
        experience_buffer: ExperienceBuffer,
        device: str = "cpu",
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for test-time adaptation")

        self.model = model
        self.config = config
        self.device = device
        self.experience_buffer = experience_buffer

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        self.loss_fn = nn.MSELoss()

    def update_step(
        self,
        x_current: torch.Tensor,
        y_current: torch.Tensor,
        hidden_state_current: torch.Tensor
    ) -> Tuple[float, int]:
        """
        Performs adaptation update steps using a hybrid loss function.
        
        The hybrid loss combines:
        1. Loss on current task data (primary objective)
        2. Loss on sampled past experiences (regularization/augmentation)
        
        Args:
            x_current: Input tensor for the current task (B_curr, input_dim)
            y_current: Target tensor for the current task (B_curr, output_dim)
            hidden_state_current: Current hidden state (B_curr, state_dim)
            
        Returns:
            Tuple[float, int]:
                - loss (float): The total hybrid loss from the final step.
                - steps (int): The number of steps taken.
        """

        self.model.train()
        total_loss_item = 0.0

        for step in range(self.config.num_steps):
            self.optimizer.zero_grad()

            # 1. Compute loss on current task data
            output_current, next_hidden_state = self.model(
                x_current, 
                hidden_state_current
            )
            loss_current = self.loss_fn(output_current, y_current)
            total_loss = loss_current

            # 2. Compute loss on past experience data (if available)
            experience_batch = self.experience_buffer.get_batch(
                self.config.experience_batch_size
            )

            if experience_batch is not None:
                x_exp, y_exp = experience_batch

                # Initialize hidden state for experience batch
                # (experiences are treated as independent sequences)
                B_exp = x_exp.shape[0]
                hidden_state_exp = self.model.init_hidden(batch_size=B_exp)

                # Forward pass for experience batch
                output_exp, _ = self.model(x_exp, hidden_state_exp)
                loss_experience = self.loss_fn(output_exp, y_exp)

                # 3. Combine losses with weighting
                total_loss = (
                    loss_current + 
                    self.config.experience_weight * loss_experience
                )

            # 4. Backpropagation on combined loss
            total_loss.backward()

            # 5. Gradient clipping (optional)
            if self.config.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip_norm
                )

            # 6. Optimizer step
            self.optimizer.step()

            # 7. Update hidden state for next iteration
            hidden_state_current = next_hidden_state.detach()
            total_loss_item = total_loss.item()

        return total_loss_item, self.config.num_steps

