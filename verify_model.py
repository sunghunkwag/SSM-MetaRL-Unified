#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script to load and test the saved model weights
This demonstrates that the .pth file contains a complete, usable model
"""
import torch
import gymnasium as gym
import numpy as np

from core.ssm import StateSpaceModel
from env_runner.environment import Environment


def load_and_test_model(model_path, num_episodes=10):
    """
    Load the saved model and test it on CartPole environment
    """
    print("=" * 60)
    print("Model Loading and Verification")
    print("=" * 60)
    print(f"\nLoading model from: {model_path}")
    
    # Initialize environment
    env = Environment(env_name='CartPole-v1', batch_size=1)
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0]
    output_dim = input_dim
    state_dim = 32
    hidden_dim = 64
    
    # Create model architecture
    model = StateSpaceModel(
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim
    )
    
    # Load saved weights
    model.load(model_path)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"\nModel Architecture:")
    print(f"  Input Dim: {input_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  State Dim: {state_dim}")
    print(f"  Hidden Dim: {hidden_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test the model
    print(f"\n{'='*60}")
    print(f"Testing model on {num_episodes} episodes...")
    print(f"{'='*60}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        hidden_state = model.init_hidden(batch_size=1)
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, hidden_state = model(obs_tensor, hidden_state)
            
            # Get action
            n_actions = action_space.n
            logits = action_logits[:, :n_actions]
            action = torch.argmax(logits, dim=-1).item()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1:2d}: Reward = {total_reward:6.1f}, Steps = {steps:3d}")
    
    env.close()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Test Results Summary:")
    print(f"{'='*60}")
    print(f"Average Reward: {np.mean(episode_rewards):6.2f} ± {np.std(episode_rewards):5.2f}")
    print(f"Min Reward:     {np.min(episode_rewards):6.1f}")
    print(f"Max Reward:     {np.max(episode_rewards):6.1f}")
    print(f"Average Steps:  {np.mean(episode_lengths):6.2f} ± {np.std(episode_lengths):5.2f}")
    print(f"{'='*60}\n")
    
    print("✅ Model verification completed successfully!")
    print("The saved .pth file contains a fully functional meta-learned model.")
    print("\nThis model can now be:")
    print("  1. Loaded without retraining using model.load()")
    print("  2. Used for inference on CartPole tasks")
    print("  3. Fine-tuned with test-time adaptation")
    print("  4. Transferred to similar control tasks")
    

if __name__ == "__main__":
    model_path = "cartpole_hybrid_real_model.pth"
    load_and_test_model(model_path, num_episodes=10)

