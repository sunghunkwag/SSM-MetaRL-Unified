"""
Standard Test-Time Adaptation Module

This module provides the standard adaptation strategy without experience replay.
It performs gradient-based adaptation using only current task data.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StandardAdaptationConfig:
    """Configuration for the Standard Adapter."""
    learning_rate: float = 0.01
    num_steps: int = 5
    grad_clip_norm: Optional[float] = 1.0
    trust_region_eps: Optional[float] = None
    ema_decay: Optional[float] = None
    entropy_weight: Optional[float] = None
    max_steps_per_call: int = 5


class StandardAdapter:
    """
    Performs standard test-time adaptation without experience replay.
    
    This adapter uses only the current task data for adaptation,
    making it suitable as a baseline for comparison with hybrid approaches.
    """

    def __init__(
        self,
        model: nn.Module,
        config: StandardAdaptationConfig,
        device: str = 'cpu'
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for test-time adaptation")
        
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        self.loss_fn = nn.MSELoss()

    def update_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        hidden_state: torch.Tensor
    ) -> Tuple[float, int]:
        """
        Performs standard adaptation update steps using only current task data.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            y: Target tensor (batch_size, output_dim)
            hidden_state: Current hidden state (batch_size, state_dim)
            
        Returns:
            Tuple[float, int]:
                - loss (float): The loss value from the final adaptation step.
                - steps (int): The number of steps taken.
        """
        
        self.model.train()
        current_loss = 0.0
        
        for step in range(self.config.num_steps):
            self.optimizer.zero_grad()
            
            # Forward pass
            output, next_hidden_state = self.model(x, hidden_state)
            
            # Calculate loss
            loss = self.loss_fn(output, y)
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping if configured
            if self.config.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update hidden state for next iteration
            hidden_state = next_hidden_state.detach()
            current_loss = loss.item()

        return current_loss, self.config.num_steps

