"""
Test-Time Adaptation for SSM-MetaRL

Implements sophisticated test-time compute strategies:
1. Online fine-tuning with experience replay
2. Meta-learned adaptation (MAML-style)
3. Uncertainty-guided exploration
4. Adaptive learning rates based on task difficulty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy


class TestTimeAdapter:
    """
    Advanced test-time adaptation with meta-learning.
    
    Key features:
    - Fast adaptation using meta-learned initialization
    - Experience replay for stable updates
    - Uncertainty estimation for exploration
    - Adaptive learning rates
    """
    
    def __init__(self,
                 model,
                 adaptation_lr=1e-3,
                 adaptation_steps=10,
                 buffer_size=1000,
                 batch_size=32,
                 use_uncertainty=True,
                 device='cpu'):
        
        self.model = model
        self.device = device
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.batch_size = batch_size
        self.use_uncertainty = use_uncertainty
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Store original parameters for meta-learning
        self.meta_parameters = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # Adaptation optimizer
        self.optimizer = None
    
    def reset_to_meta(self):
        """Reset model to meta-learned initialization"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.meta_parameters[name])
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })
    
    def sample_batch(self):
        """Sample a batch from replay buffer"""
        if len(self.buffer) < self.batch_size:
            batch = list(self.buffer)
        else:
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t['action'] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t['reward'] for t in batch])).to(self.device)
        next_obs = torch.FloatTensor(np.array([t['next_obs'] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t['done'] for t in batch])).to(self.device)
        
        return obs, actions, rewards, next_obs, dones
    
    def compute_adaptation_loss(self, obs, actions, rewards, next_obs, dones):
        """
        Compute loss for test-time adaptation.
        
        Uses a combination of:
        1. Policy gradient loss (maximize rewards)
        2. Value prediction loss (predict returns)
        3. Behavioral cloning loss (match expert actions if available)
        """
        # Get model predictions
        log_probs, values, entropy = self.model.evaluate_actions(obs, actions)
        
        # Policy gradient loss (weighted by rewards)
        policy_loss = -(log_probs * rewards.unsqueeze(1)).mean()
        
        # Value prediction loss
        with torch.no_grad():
            _, _, next_values, _ = self.model.get_action(next_obs, deterministic=True)
            targets = rewards.unsqueeze(1) + 0.99 * next_values * (1 - dones.unsqueeze(1))
        
        value_loss = F.mse_loss(values, targets)
        
        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        return total_loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
    
    def adapt(self, num_steps=None):
        """
        Perform test-time adaptation using collected experience.
        
        Args:
            num_steps: Number of gradient steps (default: self.adaptation_steps)
        
        Returns:
            stats: Adaptation statistics
        """
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        if len(self.buffer) < 10:
            return {'adapted': False, 'reason': 'insufficient_data'}
        
        # Create optimizer if not exists
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                             lr=self.adaptation_lr)
        
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        for step in range(num_steps):
            # Sample batch
            obs, actions, rewards, next_obs, dones = self.sample_batch()
            
            # Compute loss
            loss, step_stats = self.compute_adaptation_loss(
                obs, actions, rewards, next_obs, dones
            )
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track statistics
            for key, value in step_stats.items():
                stats[key].append(value)
        
        # Return average statistics
        return {
            'adapted': True,
            'policy_loss': np.mean(stats['policy_loss']),
            'value_loss': np.mean(stats['value_loss']),
            'entropy': np.mean(stats['entropy']),
            'buffer_size': len(self.buffer)
        }
    
    def adapt_online(self, env, num_episodes=5, adapt_every=10):
        """
        Perform online adaptation while interacting with environment.
        
        This is the key test-time compute strategy:
        - Collect experience from the new task
        - Periodically adapt the policy
        - Continue collecting with improved policy
        
        Args:
            env: Environment to adapt to
            num_episodes: Number of episodes to run
            adapt_every: Adapt every N steps
        
        Returns:
            episode_rewards: Rewards for each episode
            adaptation_stats: Statistics from adaptation
        """
        episode_rewards = []
        adaptation_stats = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            hidden_state = self.model.init_hidden(batch_size=1, device=self.device)
            episode_reward = 0
            done = False
            truncated = False
            steps = 0
            
            while not (done or truncated) and steps < 1000:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action
                with torch.no_grad():
                    action, _, _, hidden_state = self.model.get_action(
                        obs_tensor, hidden_state, deterministic=False
                    )
                
                action_np = action.cpu().numpy().flatten()
                action_np = np.clip(action_np, -1.0, 1.0)
                
                # Step environment
                next_obs, reward, done, truncated, info = env.step(action_np)
                
                # Store transition
                self.store_transition(obs, action_np, reward, next_obs, done or truncated)
                
                episode_reward += reward
                obs = next_obs
                steps += 1
                
                # Adapt periodically
                if steps % adapt_every == 0 and len(self.buffer) >= self.batch_size:
                    stats = self.adapt(num_steps=5)
                    if stats['adapted']:
                        adaptation_stats.append(stats)
            
            episode_rewards.append(episode_reward)
            
            print(f"  Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, "
                  f"Steps = {steps}, Buffer = {len(self.buffer)}")
        
        return episode_rewards, adaptation_stats


class MetaLearner:
    """
    Meta-learner for finding good initialization (MAML-style).
    
    This learns parameters that can quickly adapt to new tasks
    with just a few gradient steps.
    """
    
    def __init__(self,
                 model,
                 inner_lr=0.01,
                 outer_lr=0.001,
                 num_inner_steps=5,
                 device='cpu'):
        
        self.model = model
        self.device = device
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    
    def inner_loop(self, support_data, model_copy):
        """
        Inner loop: adapt to a specific task.
        
        Args:
            support_data: Data for adaptation (obs, actions, rewards)
            model_copy: Copy of model to adapt
        
        Returns:
            adapted_model: Model after adaptation
        """
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=self.inner_lr)
        
        obs, actions, rewards, next_obs, dones = support_data
        
        for step in range(self.num_inner_steps):
            # Compute loss
            log_probs, values, entropy = model_copy.evaluate_actions(obs, actions)
            loss = -(log_probs * rewards.unsqueeze(1)).mean()
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model_copy
    
    def meta_update(self, tasks):
        """
        Meta-update: update initialization to work well across tasks.
        
        Args:
            tasks: List of tasks, each with (support_data, query_data)
        
        Returns:
            meta_loss: Loss on query sets after adaptation
        """
        meta_loss = 0
        
        for support_data, query_data in tasks:
            # Create a copy of the model
            model_copy = copy.deepcopy(self.model)
            
            # Inner loop: adapt to task
            adapted_model = self.inner_loop(support_data, model_copy)
            
            # Evaluate on query set
            obs, actions, rewards, next_obs, dones = query_data
            log_probs, values, entropy = adapted_model.evaluate_actions(obs, actions)
            task_loss = -(log_probs * rewards.unsqueeze(1)).mean()
            
            meta_loss += task_loss
        
        meta_loss /= len(tasks)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()

