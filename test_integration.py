"""
Integration test script for SSM-MetaRL-Unified

Tests both standard and hybrid adaptation modes on CartPole environment.
"""
import torch
import torch.nn as nn
import numpy as np
from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import StandardAdapter, StandardAdaptationConfig
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment
import gymnasium as gym


def test_standard_mode():
    """Test standard adaptation mode."""
    print("\n" + "="*70)
    print("TEST 1: Standard Adaptation Mode")
    print("="*70)
    
    device = torch.device('cpu')
    env = Environment(env_name='CartPole-v1', batch_size=1)
    
    input_dim = env.observation_space.shape[0]
    output_dim = input_dim
    
    model = StateSpaceModel(
        state_dim=32,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64
    ).to(device)
    
    # Quick meta-training
    print("Meta-training...")
    meta_learner = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)
    
    for epoch in range(2):
        obs = env.reset()
        hidden_state = model.init_hidden(batch_size=1)
        
        observations = []
        next_observations = []
        
        for step in range(50):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output, hidden_state = model(obs_tensor, hidden_state)
                action = env.action_space.sample()
            
            next_obs, reward, done, info = env.step(action)
            
            observations.append(obs)
            next_observations.append(next_obs)
            
            obs = next_obs
            if done:
                obs = env.reset()
                hidden_state = model.init_hidden(batch_size=1)
        
        obs_seq = torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(0).to(device)
        next_obs_seq = torch.tensor(np.array(next_observations), dtype=torch.float32).unsqueeze(0).to(device)
        
        split_idx = len(observations) // 2
        tasks = [(obs_seq[:, :split_idx], next_obs_seq[:, :split_idx],
                 obs_seq[:, split_idx:], next_obs_seq[:, split_idx:])]
        
        initial_hidden = model.init_hidden(batch_size=1)
        loss = meta_learner.meta_update(tasks, initial_hidden_state=initial_hidden, loss_fn=nn.MSELoss())
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")
    
    # Test adaptation
    print("Testing adaptation...")
    config = StandardAdaptationConfig(learning_rate=0.01, num_steps=5)
    adapter = StandardAdapter(model=model, config=config, device=device)
    
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    
    for step in range(5):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, next_hidden_state = model(obs_tensor, hidden_state)
            action = env.action_space.sample()
        
        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        loss_val, steps_taken = adapter.update_step(
            x=obs_tensor,
            y=next_obs_tensor,
            hidden_state=hidden_state
        )
        
        print(f"  Step {step}: Loss = {loss_val:.4f}")
        
        obs = next_obs
        hidden_state = next_hidden_state
        
        if done:
            break
    
    env.close()
    print("✓ Standard mode test PASSED")
    return True


def test_hybrid_mode():
    """Test hybrid adaptation mode with experience replay."""
    print("\n" + "="*70)
    print("TEST 2: Hybrid Adaptation Mode (with Experience Replay)")
    print("="*70)
    
    device = torch.device('cpu')
    env = Environment(env_name='CartPole-v1', batch_size=1)
    
    input_dim = env.observation_space.shape[0]
    output_dim = input_dim
    
    model = StateSpaceModel(
        state_dim=32,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64
    ).to(device)
    
    # Initialize experience buffer
    experience_buffer = ExperienceBuffer(max_size=1000, device=str(device))
    print(f"Initialized ExperienceBuffer (max_size=1000)")
    
    # Quick meta-training with buffer population
    print("Meta-training with buffer population...")
    meta_learner = MetaMAML(model=model, inner_lr=0.01, outer_lr=0.001)
    
    for epoch in range(2):
        obs = env.reset()
        hidden_state = model.init_hidden(batch_size=1)
        
        observations = []
        next_observations = []
        
        for step in range(50):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output, hidden_state = model(obs_tensor, hidden_state)
                action = env.action_space.sample()
            
            next_obs, reward, done, info = env.step(action)
            
            observations.append(obs)
            next_observations.append(next_obs)
            
            # Add to experience buffer
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
            experience_buffer.add(obs_t.unsqueeze(0), next_obs_t.unsqueeze(0))
            
            obs = next_obs
            if done:
                obs = env.reset()
                hidden_state = model.init_hidden(batch_size=1)
        
        obs_seq = torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(0).to(device)
        next_obs_seq = torch.tensor(np.array(next_observations), dtype=torch.float32).unsqueeze(0).to(device)
        
        split_idx = len(observations) // 2
        tasks = [(obs_seq[:, :split_idx], next_obs_seq[:, :split_idx],
                 obs_seq[:, split_idx:], next_obs_seq[:, split_idx:])]
        
        initial_hidden = model.init_hidden(batch_size=1)
        loss = meta_learner.meta_update(tasks, initial_hidden_state=initial_hidden, loss_fn=nn.MSELoss())
        print(f"  Epoch {epoch}: Loss = {loss:.4f}, Buffer size = {len(experience_buffer)}")
    
    # Test hybrid adaptation
    print("Testing hybrid adaptation...")
    config = HybridAdaptationConfig(
        learning_rate=0.01,
        num_steps=5,
        experience_batch_size=16,
        experience_weight=0.1
    )
    adapter = HybridAdapter(
        model=model,
        config=config,
        experience_buffer=experience_buffer,
        device=device
    )
    
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    
    for step in range(5):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, next_hidden_state = model(obs_tensor, hidden_state)
            action = env.action_space.sample()
        
        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        loss_val, steps_taken = adapter.update_step(
            x_current=obs_tensor,
            y_current=next_obs_tensor,
            hidden_state_current=hidden_state
        )
        
        print(f"  Step {step}: Loss = {loss_val:.4f} (using experience replay)")
        
        obs = next_obs
        hidden_state = next_hidden_state
        
        if done:
            break
    
    env.close()
    print("✓ Hybrid mode test PASSED")
    return True


def test_experience_buffer():
    """Test ExperienceBuffer functionality."""
    print("\n" + "="*70)
    print("TEST 3: ExperienceBuffer Functionality")
    print("="*70)
    
    device = torch.device('cpu')
    buffer = ExperienceBuffer(max_size=100, device=str(device))
    
    print("Adding experiences...")
    for i in range(50):
        obs = torch.randn(5, 4)
        next_obs = torch.randn(5, 4)
        buffer.add(obs, next_obs)
    
    print(f"  Buffer size: {len(buffer)}/100")
    
    print("Sampling batch...")
    batch = buffer.get_batch(batch_size=10)
    
    if batch is not None:
        obs_batch, next_obs_batch = batch
        print(f"  Sampled batch shape: {obs_batch.shape}, {next_obs_batch.shape}")
        print("✓ ExperienceBuffer test PASSED")
        return True
    else:
        print("✗ ExperienceBuffer test FAILED")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "#"*70)
    print("# SSM-MetaRL-Unified Integration Tests")
    print("#"*70)
    
    results = []
    
    try:
        results.append(("ExperienceBuffer", test_experience_buffer()))
    except Exception as e:
        print(f"✗ ExperienceBuffer test FAILED: {e}")
        results.append(("ExperienceBuffer", False))
    
    try:
        results.append(("Standard Mode", test_standard_mode()))
    except Exception as e:
        print(f"✗ Standard mode test FAILED: {e}")
        results.append(("Standard Mode", False))
    
    try:
        results.append(("Hybrid Mode", test_hybrid_mode()))
    except Exception as e:
        print(f"✗ Hybrid mode test FAILED: {e}")
        results.append(("Hybrid Mode", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

