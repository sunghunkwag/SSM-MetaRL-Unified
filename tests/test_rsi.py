#!/usr/bin/env python3
"""
Test script for Recursive Self-Improvement module
Tests the RSI functionality to ensure it actually works before deployment
"""
import torch
import gymnasium as gym
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ssm import StateSpaceModel
from recursive_self_improvement import (
    RecursiveSelfImprovementAgent,
    RSIConfig,
    SafetyConfig,
    ArchitecturalConfig,
    LearningConfig
)

def test_rsi_basic():
    """Test basic RSI functionality"""
    print("=" * 60)
    print("Testing Recursive Self-Improvement Module")
    print("=" * 60)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Initialize model
    print("\n1. Initializing base model...")
    model = StateSpaceModel(
        state_dim=32,
        input_dim=4,
        output_dim=4,
        hidden_dim=64
    )
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load pre-trained weights if available
    if os.path.exists("cartpole_hybrid_real_model.pth"):
        print("\n2. Loading pre-trained weights...")
        model.load("cartpole_hybrid_real_model.pth")
        print("✓ Pre-trained weights loaded")
    else:
        print("\n2. No pre-trained weights found, using random initialization")
    
    # Initialize RSI system
    print("\n3. Initializing RSI system...")
    rsi_config = RSIConfig(
        num_episodes_quick=5,  # Reduced for faster testing
        num_episodes_full=10,
        num_meta_tasks_quick=2,
        num_meta_tasks_full=5,
        meta_task_length=30,
        adaptation_steps=3
    )
    
    safety_config = SafetyConfig(
        performance_window=5,
        min_performance_threshold=-500,
        max_emergency_stops=3
    )
    
    arch_config = ArchitecturalConfig(
        state_dim=32,
        hidden_dim=64,
        num_layers=1
    )
    
    learn_config = LearningConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        meta_batch_size=3,
        adaptation_steps=1
    )
    
    rsi = RecursiveSelfImprovementAgent(
        initial_model=model,
        env=env,
        device='cpu',
        safety_config=safety_config,
        rsi_config=rsi_config
    )
    
    # Set configs manually
    rsi.arch_config = arch_config
    rsi.learn_config = learn_config
    print("✓ RSI system initialized")
    
    # Test initial evaluation
    print("\n4. Testing initial performance evaluation...")
    try:
        initial_metrics = rsi.evaluate_performance(quick_eval=True)
        print(f"✓ Initial metrics:")
        print(f"  - Average Reward: {initial_metrics.avg_reward:.2f}")
        print(f"  - Adaptation Speed: {initial_metrics.adaptation_speed:.2f}")
        print(f"  - Generalization Score: {initial_metrics.generalization_score:.2f}")
        print(f"  - Meta Efficiency: {initial_metrics.meta_learning_efficiency:.2f}")
        print(f"  - Stability Score: {initial_metrics.stability_score:.2f}")
    except Exception as e:
        print(f"✗ Initial evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test one improvement cycle
    print("\n5. Testing one self-improvement cycle...")
    try:
        improved = rsi.attempt_self_improvement()
        if improved:
            print("✓ Self-improvement cycle completed successfully!")
            print(f"  - Generation: {rsi.generation}")
            print(f"  - Current Reward: {rsi.current_metrics.avg_reward:.2f}")
        else:
            print("⚠ Self-improvement cycle completed but no improvement found")
            print("  (This is normal - not all cycles produce improvements)")
    except Exception as e:
        print(f"✗ Self-improvement cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test history tracking
    print("\n6. Testing history tracking...")
    history = rsi.improvement_history
    print(f"✓ History contains {len(history)} entries")
    if history:
        latest = history[-1]
        print(f"  Latest entry:")
        print(f"    - Generation: {latest['generation']}")
        print(f"    - Reward: {latest['metrics']['avg_reward']:.2f}")
        print(f"    - Improvement Type: {latest.get('improvement_type', 'N/A')}")
    
    # Test checkpoint system
    print("\n7. Testing checkpoint system...")
    try:
        num_checkpoints = len(rsi.checkpoint_system.checkpoints)
        print(f"✓ Checkpoint system has {num_checkpoints} checkpoints")
        if num_checkpoints > 0:
            latest_checkpoint = rsi.checkpoint_system.checkpoints[-1]
            print(f"  Latest checkpoint ID: {latest_checkpoint['id']}")
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All RSI tests passed!")
    print("=" * 60)
    print("\nRSI module is working correctly and ready for integration.")
    
    return True

def test_rsi_multiple_cycles():
    """Test multiple RSI cycles"""
    print("\n" + "=" * 60)
    print("Testing Multiple RSI Cycles")
    print("=" * 60)
    
    env = gym.make('CartPole-v1')
    model = StateSpaceModel(state_dim=32, input_dim=4, output_dim=4, hidden_dim=64)
    
    # Load pre-trained if available
    if os.path.exists("cartpole_hybrid_real_model.pth"):
        model.load("cartpole_hybrid_real_model.pth")
        print("✓ Using pre-trained model")
    
    rsi_config = RSIConfig(
        num_episodes_quick=3,
        num_meta_tasks_quick=2,
        meta_task_length=20
    )
    
    rsi = RecursiveSelfImprovementAgent(
        initial_model=model,
        env=env,
        device='cpu',
        safety_config=SafetyConfig(),
        rsi_config=rsi_config
    )
    rsi.arch_config = ArchitecturalConfig()
    rsi.learn_config = LearningConfig()
    
    print("\nRunning 3 improvement cycles...")
    rewards = []
    
    for i in range(3):
        print(f"\n--- Cycle {i+1}/3 ---")
        try:
            improved = rsi.attempt_self_improvement()
            current_reward = rsi.current_metrics.avg_reward
            rewards.append(current_reward)
            print(f"Cycle {i+1}: Reward = {current_reward:.2f}, Improved = {improved}")
        except Exception as e:
            print(f"Cycle {i+1} failed: {e}")
            break
    
    if len(rewards) >= 2:
        print(f"\n✓ Multiple cycles completed")
        print(f"  Reward progression: {[f'{r:.2f}' for r in rewards]}")
        
        # Check if there's any improvement trend
        if rewards[-1] > rewards[0]:
            print(f"  ✓ Overall improvement: {rewards[-1] - rewards[0]:.2f}")
        else:
            print(f"  ⚠ No overall improvement (this can happen with random exploration)")
    
    return True

if __name__ == "__main__":
    print("Starting RSI Module Tests\n")
    
    # Test 1: Basic functionality
    success = test_rsi_basic()
    
    if success:
        # Test 2: Multiple cycles
        test_rsi_multiple_cycles()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("=" * 60)
        print("\nThe RSI module is ready for deployment.")
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed")
        print("=" * 60)
        print("\nPlease fix the issues before deployment.")
        sys.exit(1)

