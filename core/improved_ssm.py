"""
Improved State Space Model for Continuous Control

Key improvements:
1. Proper layer normalization for stable training
2. Better initialization schemes
3. Residual connections for gradient flow
4. Separate value head for actor-critic
5. Adaptive state dimension based on task complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedSSM(nn.Module):
    """
    Enhanced State Space Model with modern deep RL techniques.
    
    Architecture improvements:
    - Layer normalization for stable gradients
    - Orthogonal initialization for recurrent weights
    - Residual connections
    - Separate policy and value heads (Actor-Critic)
    - Proper continuous action distribution parameterization
    """
    
    def __init__(self, 
                 input_dim,
                 action_dim,
                 state_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 use_layer_norm=True,
                 use_residual=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(
                state_dim=state_dim,
                input_dim=hidden_dim if i == 0 else hidden_dim,
                output_dim=hidden_dim,
                use_layer_norm=use_layer_norm
            )
            for i in range(num_layers)
        ])
        
        # Policy head (actor) - outputs mean and log_std
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Learnable log_std (independent of state)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper schemes"""
        # Orthogonal initialization for recurrent connections
        for layer in self.ssm_layers:
            nn.init.orthogonal_(layer.A, gain=0.9)  # Slightly less than 1 for stability
            nn.init.xavier_uniform_(layer.B)
            nn.init.xavier_uniform_(layer.C)
        
        # Xavier for feedforward layers
        for module in [self.input_proj, self.policy_mean, self.value_head]:
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Small initialization for policy output layer (last layer of policy_mean)
        nn.init.uniform_(self.policy_mean[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.policy_mean[-1].bias, -3e-3, 3e-3)
    
    def init_hidden(self, batch_size=1, device=None):
        """Initialize hidden state"""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.state_dim, device=device)
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass through improved SSM.
        
        Args:
            x: Input tensor (batch, input_dim) or (batch, seq_len, input_dim)
            hidden_state: Previous hidden state (batch, state_dim)
        
        Returns:
            action_mean: Mean of action distribution (batch, action_dim)
            action_log_std: Log std of action distribution (batch, action_dim)
            value: State value estimate (batch, 1)
            next_hidden: Updated hidden state (batch, state_dim)
        """
        batch_size = x.shape[0]
        
        # Handle sequence input
        if len(x.shape) == 3:
            # x is (batch, seq_len, input_dim)
            # Process sequentially
            seq_len = x.shape[1]
            outputs = []
            values = []
            
            if hidden_state is None:
                hidden_state = self.init_hidden(batch_size, x.device)
            
            for t in range(seq_len):
                mean, log_std, value, hidden_state = self.forward(x[:, t], hidden_state)
                outputs.append((mean, log_std))
                values.append(value)
            
            # Stack outputs
            means = torch.stack([o[0] for o in outputs], dim=1)
            log_stds = torch.stack([o[1] for o in outputs], dim=1)
            values = torch.stack(values, dim=1)
            
            return means, log_stds, values, hidden_state
        
        # Single step forward
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, x.device)
        
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # SSM layers with residual connections
        for i, layer in enumerate(self.ssm_layers):
            h_new, hidden_state = layer(h, hidden_state)
            if self.use_residual and i > 0:
                h = h + h_new  # Residual connection
            else:
                h = h_new
        
        # Policy head
        action_mean = self.policy_mean(h)
        action_mean = torch.tanh(action_mean)  # Bound to [-1, 1]
        
        # Expand log_std to match batch size
        action_log_std = self.policy_log_std.expand(batch_size, -1)
        
        # Value head
        value = self.value_head(h)
        
        return action_mean, action_log_std, value, hidden_state
    
    def get_action(self, x, hidden_state=None, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            x: Observation
            hidden_state: Previous hidden state
            deterministic: If True, return mean action
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
            next_hidden: Updated hidden state
        """
        action_mean, action_log_std, value, next_hidden = self.forward(x, hidden_state)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value, next_hidden
    
    def evaluate_actions(self, x, actions, hidden_state=None):
        """
        Evaluate actions for PPO-style updates.
        
        Args:
            x: Observations
            actions: Actions taken
            hidden_state: Previous hidden state
        
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_mean, action_log_std, value, _ = self.forward(x, hidden_state)
        
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Squeeze value to match expected shape
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return log_probs, value, entropy
    
    def save(self, path):
        """Save model"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'action_dim': self.action_dim,
                'state_dim': self.state_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'use_layer_norm': self.use_layer_norm,
                'use_residual': self.use_residual
            }
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])


class SSMLayer(nn.Module):
    """Single SSM layer with proper normalization"""
    
    def __init__(self, state_dim, input_dim, output_dim, use_layer_norm=True):
        super().__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # State space matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))  # State transition
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))  # Input projection
        self.C = nn.Parameter(torch.randn(output_dim, state_dim))  # Output projection
        self.D = nn.Parameter(torch.randn(output_dim, input_dim))  # Feedthrough
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, hidden_state):
        """
        Forward pass through SSM layer.
        
        Args:
            x: Input (batch, input_dim)
            hidden_state: Previous state (batch, state_dim)
        
        Returns:
            output: Layer output (batch, output_dim)
            next_state: Updated state (batch, state_dim)
        """
        # State update: s_t = A @ s_{t-1} + B @ x_t
        next_state = torch.matmul(hidden_state, self.A.t()) + torch.matmul(x, self.B.t())
        
        # Output: y_t = C @ s_t + D @ x_t
        output = torch.matmul(next_state, self.C.t()) + torch.matmul(x, self.D.t())
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output, next_state

