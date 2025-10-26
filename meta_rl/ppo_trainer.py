"""
PPO Trainer for Improved SSM

Implements Proximal Policy Optimization with:
1. Generalized Advantage Estimation (GAE)
2. Clipped surrogate objective
3. Value function clipping
4. Entropy bonus
5. Gradient clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for SSM-based policies.
    
    This is a modern, stable RL algorithm that works well with
    continuous control tasks.
    """
    
    def __init__(self,
                 model,
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=10,
                 mini_batch_size=64,
                 device='cpu'):
        
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training statistics
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        # Convert to tensors
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Append next value
        if next_value.dim() == 0:
            next_value = next_value.unsqueeze(0)
        values = torch.cat([values, next_value])
        
        # Compute GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.stack(advantages)
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def collect_rollout(self, env, num_steps=2048):
        """
        Collect a rollout from the environment.
        
        Args:
            env: Gymnasium environment
            num_steps: Number of steps to collect
        
        Returns:
            rollout: Dictionary containing trajectory data
        """
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, _ = env.reset()
        hidden_state = self.model.init_hidden(batch_size=1, device=self.device)
        
        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value, hidden_state = self.model.get_action(
                    obs_tensor, hidden_state, deterministic=False
                )
            
            # Step environment
            action_np = action.cpu().numpy().flatten()
            action_np = np.clip(action_np, -1.0, 1.0)
            next_obs, reward, done, truncated, info = env.step(action_np)
            
            # Store transition
            observations.append(obs)
            actions.append(action.cpu())
            rewards.append(reward)
            values.append(value.cpu().item())
            log_probs.append(log_prob.cpu())
            dones.append(done or truncated)
            
            obs = next_obs
            
            if done or truncated:
                obs, _ = env.reset()
                hidden_state = self.model.init_hidden(batch_size=1, device=self.device)
        
        # Get final value estimate
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, next_value, _ = self.model.get_action(obs_tensor, hidden_state)
        next_value = next_value.cpu().item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, 
                                               torch.FloatTensor([next_value]))
        
        rollout = {
            'observations': torch.FloatTensor(np.array(observations)),
            'actions': torch.cat(actions),
            'old_log_probs': torch.cat(log_probs),
            'advantages': advantages,
            'returns': returns,
            'values': torch.FloatTensor(values)
        }
        
        return rollout, sum(rewards)
    
    def update(self, rollout):
        """
        Update policy using PPO.
        
        Args:
            rollout: Dictionary containing trajectory data
        
        Returns:
            stats: Training statistics
        """
        observations = rollout['observations'].to(self.device)
        actions = rollout['actions'].to(self.device)
        old_log_probs = rollout['old_log_probs'].to(self.device)
        advantages = rollout['advantages'].to(self.device)
        returns = rollout['returns'].to(self.device)
        old_values = rollout['values'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Mini-batch updates
            num_samples = len(observations)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.model.evaluate_actions(
                    batch_obs, batch_actions, hidden_state=None
                )
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                   1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss1 = (values - batch_returns) ** 2
                value_loss2 = (values_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                
                self.stats['policy_loss'].append(policy_loss.item())
                self.stats['value_loss'].append(value_loss.item())
                self.stats['entropy'].append(-entropy_loss.item())
                self.stats['total_loss'].append(loss.item())
                self.stats['approx_kl'].append(approx_kl)
                self.stats['clip_fraction'].append(clip_fraction)
        
        # Return average statistics
        return {
            'policy_loss': np.mean(self.stats['policy_loss'][-10:]),
            'value_loss': np.mean(self.stats['value_loss'][-10:]),
            'entropy': np.mean(self.stats['entropy'][-10:]),
            'total_loss': np.mean(self.stats['total_loss'][-10:]),
            'approx_kl': np.mean(self.stats['approx_kl'][-10:]),
            'clip_fraction': np.mean(self.stats['clip_fraction'][-10:])
        }
    
    def train(self, env, total_timesteps=1000000, log_interval=10):
        """
        Train the policy using PPO.
        
        Args:
            env: Gymnasium environment
            total_timesteps: Total number of timesteps to train
            log_interval: How often to log progress
        
        Returns:
            episode_rewards: List of episode rewards
        """
        episode_rewards = []
        timesteps = 0
        iteration = 0
        
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        
        while timesteps < total_timesteps:
            # Collect rollout
            rollout, total_reward = self.collect_rollout(env, num_steps=2048)
            timesteps += len(rollout['observations'])
            iteration += 1
            
            # Update policy
            stats = self.update(rollout)
            
            episode_rewards.append(total_reward)
            
            # Log progress
            if iteration % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                print(f"Iteration {iteration} | Timesteps {timesteps}/{total_timesteps}")
                print(f"  Reward: {total_reward:.2f} | Avg(10): {avg_reward:.2f}")
                print(f"  Policy Loss: {stats['policy_loss']:.4f}")
                print(f"  Value Loss: {stats['value_loss']:.4f}")
                print(f"  Entropy: {stats['entropy']:.4f}")
                print(f"  KL: {stats['approx_kl']:.4f}")
                print()
        
        return episode_rewards

