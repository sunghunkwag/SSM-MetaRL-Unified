# -*- coding: utf-8 -*-
"""
Main training and adaptation script for SSM-MetaRL-Unified.

This unified version integrates:
- Standard adaptation (baseline)
- Hybrid adaptation with experience replay (experience-augmented)

The hybrid mode leverages an ExperienceBuffer to combine current task data
with past experiences for more robust test-time adaptation.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import gymnasium as gym

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import StandardAdapter, StandardAdaptationConfig
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment


def collect_data(
    env, 
    policy_model, 
    num_episodes=10, 
    max_steps_per_episode=100, 
    device='cpu',
    experience_buffer=None
):
    """
    Collects trajectory data from environment.
    
    Args:
        env: Environment instance
        policy_model: SSM model for action selection
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        device: Torch device
        experience_buffer: Optional ExperienceBuffer to populate during collection
        
    Returns:
        Dictionary containing observations, actions, rewards, and next_observations
    """
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    policy_model.eval()
    
    obs = env.reset()
    hidden_state = policy_model.init_hidden(batch_size=env.batch_size)
    
    total_steps = 0
    for ep in range(num_episodes):
        steps_in_ep = 0
        done = False
        
        while not done and steps_in_ep < max_steps_per_episode:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)
                
                if isinstance(env.action_space, gym.spaces.Discrete):
                    n_actions = env.action_space.n
                    probs = torch.softmax(action_logits[:, :n_actions], dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = action_logits.cpu().numpy().flatten()
            
            next_obs, reward, done, info = env.step(action)
            
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(done)
            
            # Add to experience buffer if provided
            if experience_buffer is not None:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
                experience_buffer.add(obs_t.unsqueeze(0), next_obs_t.unsqueeze(0))
            
            obs = next_obs
            hidden_state = next_hidden_state
            steps_in_ep += 1
            total_steps += 1
        
        # Reset at episode end
        obs = env.reset()
        hidden_state = policy_model.init_hidden(batch_size=env.batch_size)
            
    return {
        'observations': torch.tensor(np.array(all_obs), dtype=torch.float32).unsqueeze(0).to(device),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.long).unsqueeze(0).to(device),
        'rewards': torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device),
        'next_observations': torch.tensor(np.array(all_next_obs), dtype=torch.float32).unsqueeze(0).to(device)
    }


def train_meta(args, model, env, device, experience_buffer=None):
    """
    Meta-training with MetaMAML.
    
    Args:
        args: Command-line arguments
        model: SSM model
        env: Environment instance
        device: Torch device
        experience_buffer: Optional buffer to populate during training
    """
    print("Starting MetaMAML training...")
    meta_learner = MetaMAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr
    )
    
    for epoch in range(args.num_epochs):
        data = collect_data(
            env, model, 
            num_episodes=args.episodes_per_task, 
            max_steps_per_episode=100, 
            device=device,
            experience_buffer=experience_buffer
        )
        
        obs_seq = data['observations']
        next_obs_seq = data['next_observations']
        
        total_len = obs_seq.shape[1]
        if total_len < 2:
            print("Warning: Collected data is too short, skipping epoch.")
            continue
            
        split_idx = total_len // 2
        
        x_support = obs_seq[:, :split_idx]
        y_support = next_obs_seq[:, :split_idx]
        x_query = obs_seq[:, split_idx:]
        y_query = next_obs_seq[:, split_idx:]
        
        tasks = [(x_support, y_support, x_query, y_query)]
        initial_hidden = model.init_hidden(batch_size=1)
        
        loss = meta_learner.meta_update(
            tasks, 
            initial_hidden_state=initial_hidden,
            loss_fn=nn.MSELoss()
        )
        
        if epoch % 10 == 0:
            buffer_info = ""
            if experience_buffer is not None:
                buffer_info = f", Buffer size: {len(experience_buffer)}"
            print(f"Epoch {epoch}, Meta Loss: {loss:.4f}{buffer_info}")
    
    print("MetaMAML training completed.")


def test_time_adapt(args, model, env, device, adaptation_mode='standard', experience_buffer=None):
    """
    Test-time adaptation with selectable adaptation strategy.
    
    Args:
        args: Command-line arguments
        model: SSM model
        env: Environment instance
        device: Torch device
        adaptation_mode: 'standard' or 'hybrid'
        experience_buffer: Required for hybrid mode
    """
    print(f"Starting test-time adaptation (mode: {adaptation_mode})...")
    
    # Create appropriate adapter based on mode
    if adaptation_mode == 'hybrid':
        if experience_buffer is None:
            raise ValueError("ExperienceBuffer is required for hybrid adaptation mode")
        
        config = HybridAdaptationConfig(
            learning_rate=args.adapt_lr,
            num_steps=5,
            experience_batch_size=args.experience_batch_size,
            experience_weight=args.experience_weight
        )
        adapter = HybridAdapter(
            model=model, 
            config=config, 
            experience_buffer=experience_buffer,
            device=device
        )
    else:  # standard mode
        config = StandardAdaptationConfig(
            learning_rate=args.adapt_lr,
            num_steps=5
        )
        adapter = StandardAdapter(model=model, config=config, device=device)
    
    # Initialize environment
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    
    for step in range(args.num_adapt_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        current_hidden_state_for_adapt = hidden_state
        
        # Get action and next state
        with torch.no_grad():
            output, hidden_state = model(obs_tensor, current_hidden_state_for_adapt)
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Perform adaptation update
        if adaptation_mode == 'hybrid':
            loss_val, steps_taken = adapter.update_step(
                x_current=obs_tensor,
                y_current=next_obs_tensor,
                hidden_state_current=current_hidden_state_for_adapt
            )
        else:  # standard mode
            loss_val, steps_taken = adapter.update_step(
                x=obs_tensor,
                y=next_obs_tensor,
                hidden_state=current_hidden_state_for_adapt
            )
        
        obs = next_obs
        
        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)
        
        if step % 10 == 0:
            print(f"Adaptation step {step}, Loss: {loss_val:.4f}, Steps taken: {steps_taken}")
    
    print("Adaptation completed.")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="SSM-MetaRL-Unified: Training and Adaptation with Experience Replay"
    )
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='CartPole-v1', 
                       help='Gymnasium environment name')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Environment batch size (currently only supports 1)')
    
    # Model architecture
    parser.add_argument('--state_dim', type=int, default=32, 
                       help='SSM state dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                       help='SSM hidden layer dimension')
    
    # Meta-training settings
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='Number of meta-training epochs')
    parser.add_argument('--episodes_per_task', type=int, default=5, 
                       help='Episodes collected per meta-task')
    parser.add_argument('--inner_lr', type=float, default=0.01, 
                       help='Inner learning rate for MetaMAML')
    parser.add_argument('--outer_lr', type=float, default=0.001, 
                       help='Outer learning rate for MetaMAML')
    
    # Adaptation settings
    parser.add_argument('--adapt_lr', type=float, default=0.01, 
                       help='Learning rate for test-time adaptation')
    parser.add_argument('--num_adapt_steps', type=int, default=50, 
                       help='Total number of adaptation steps during test')
    parser.add_argument('--adaptation_mode', type=str, default='standard', 
                       choices=['standard', 'hybrid'],
                       help='Adaptation mode: standard or hybrid (with experience replay)')
    
    # Experience replay settings (for hybrid mode)
    parser.add_argument('--buffer_size', type=int, default=10000, 
                       help='Maximum size of experience buffer')
    parser.add_argument('--experience_batch_size', type=int, default=32, 
                       help='Batch size for sampling from experience buffer')
    parser.add_argument('--experience_weight', type=float, default=0.1, 
                       help='Weight for experience loss in hybrid mode')
    
    args = parser.parse_args()
    
    if args.batch_size != 1:
        print("Warning: This example currently assumes batch_size=1 for simplicity.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize environment
    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    output_dim = input_dim
    
    args.input_dim = input_dim
    args.output_dim = output_dim
    
    # Initialize model
    model = StateSpaceModel(
        state_dim=args.state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Initialize experience buffer if using hybrid mode
    experience_buffer = None
    if args.adaptation_mode == 'hybrid':
        experience_buffer = ExperienceBuffer(
            max_size=args.buffer_size,
            device=str(device)
        )
        print(f"Initialized ExperienceBuffer with max_size={args.buffer_size}")
    
    print(f"\n=== SSM-MetaRL-Unified ===")
    print(f"Environment: {args.env_name}")
    print(f"Device: {device}")
    print(f"Input/Output Dim: {input_dim}/{output_dim}")
    print(f"State/Hidden Dim: {args.state_dim}/{args.hidden_dim}")
    print(f"Adaptation Mode: {args.adaptation_mode}")
    if args.adaptation_mode == 'hybrid':
        print(f"Experience Buffer Size: {args.buffer_size}")
        print(f"Experience Batch Size: {args.experience_batch_size}")
        print(f"Experience Weight: {args.experience_weight}")
    print("==================================\n")
    
    # Meta-train with MetaMAML
    train_meta(args, model, env, device, experience_buffer=experience_buffer)
    
    # Test-time adaptation
    test_time_adapt(
        args, model, env, device, 
        adaptation_mode=args.adaptation_mode,
        experience_buffer=experience_buffer
    )
    
    print("\n=== Execution completed successfully ===")
    print(f"All components working in {args.adaptation_mode} mode.")

if __name__ == "__main__":
    main()

