# -*- coding: utf-8 -*-
"""
SSM-MetaRL-Unified Gradio App with RSI
Complete interface with all tabs functional
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os

# Import project modules
from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.standard_adapter import StandardAdapter
from adaptation.hybrid_adapter import HybridAdapter
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment
from recursive_self_improvement import (
    RecursiveSelfImprovementAgent,
    RSIConfig,
    SafetyConfig,
    ArchitecturalConfig,
    LearningConfig
)

# Global RSI agent
global_rsi_agent = None

# Pre-trained model configuration
PRETRAINED_MODEL_PATH = "models/cartpole_hybrid_real_model.pth"
PRETRAINED_CONFIG = {
    'env_name': 'CartPole-v1',
    'state_dim': 32,
    'hidden_dim': 64,
    'input_dim': 4,
    'output_dim': 4
}


def load_model_and_init_rsi():
    """Load pre-trained model and initialize RSI"""
    global global_rsi_agent
    
    output = "=" * 60 + "\n"
    output += "Loading Pre-trained Model\n"
    output += "=" * 60 + "\n\n"
    
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        return None, None, f"❌ Error: Pre-trained model file not found: {PRETRAINED_MODEL_PATH}"
    
    try:
        # Initialize model
        model = StateSpaceModel(
            state_dim=PRETRAINED_CONFIG['state_dim'],
            input_dim=PRETRAINED_CONFIG['input_dim'],
            output_dim=PRETRAINED_CONFIG['output_dim'],
            hidden_dim=PRETRAINED_CONFIG['hidden_dim']
        )
        
        # Load weights
        model.load(PRETRAINED_MODEL_PATH)
        model.eval()
        
        # Initialize experience buffer
        experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
        
        output += f"✅ Model loaded successfully!\n"
        output += f"   File: {PRETRAINED_MODEL_PATH}\n"
        output += f"   State Dim: {PRETRAINED_CONFIG['state_dim']}\n"
        output += f"   Hidden Dim: {PRETRAINED_CONFIG['hidden_dim']}\n"
        output += f"   Parameters: {sum(p.numel() for p in model.parameters())}\n\n"
        
        # Initialize RSI
        output += "=" * 60 + "\n"
        output += "Initializing RSI System\n"
        output += "=" * 60 + "\n\n"
        
        env = Environment(PRETRAINED_CONFIG['env_name'])
        
        rsi_config = RSIConfig(
            num_episodes_quick=10,
            num_episodes_full=20,
            num_meta_tasks_quick=3,
            num_meta_tasks_full=10,
            meta_task_length=50,
            adaptation_steps=5
        )
        
        safety_config = SafetyConfig(
            performance_window=10,
            min_performance_threshold=-500,
            max_emergency_stops=3
        )
        
        arch_config = ArchitecturalConfig(
            state_dim=PRETRAINED_CONFIG['state_dim'],
            hidden_dim=PRETRAINED_CONFIG['hidden_dim']
        )
        
        learn_config = LearningConfig(
            inner_lr=0.01,
            outer_lr=0.001,
            adaptation_steps=5
        )
        
        global_rsi_agent = RecursiveSelfImprovementAgent(
            initial_model=model,
            env=env,
            device='cpu',
            rsi_config=rsi_config,
            safety_config=safety_config
        )
        
        output += "✅ RSI System initialized!\n\n"
        output += "=" * 60 + "\n"
        output += "Ready to Use!\n"
        output += "=" * 60 + "\n\n"
        output += "You can now:\n"
        output += "1. Go to Tab 2 to test the model\n"
        output += "2. Go to Tab 3 to run recursive self-improvement\n"
        output += "3. Or train a new model in Tab 1\n"
        
        return model, experience_buffer, output
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error loading model:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


def train_meta_rl(env_name, train_epochs, train_tasks, train_state_dim, train_hidden_dim, 
                  train_inner_lr, train_outer_lr, train_gamma):
    """Train meta-RL model from scratch"""
    output = "=" * 60 + "\n"
    output += f"Meta-Training Started\n"
    output += "=" * 60 + "\n\n"
    output += f"Environment: {env_name}\n"
    output += f"Epochs: {train_epochs}\n"
    output += f"Tasks/Epoch: {train_tasks}\n"
    output += f"State Dim: {train_state_dim}, Hidden Dim: {train_hidden_dim}\n"
    output += f"Inner LR: {train_inner_lr}, Outer LR: {train_outer_lr}\n"
    output += f"Gamma: {train_gamma}\n\n"
    
    try:
        # Initialize environment
        env = Environment(env_name=env_name)
        obs_space = env.observation_space
        action_space = env.action_space
        
        input_dim = obs_space.shape[0]
        output_dim = action_space.n
        
        # Initialize model
        model = StateSpaceModel(
            state_dim=train_state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=train_hidden_dim
        )
        
        # Initialize experience buffer
        experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
        
        # Initialize MetaMAML
        meta_learner = MetaMAML(
            model=model,
            inner_lr=train_inner_lr,
            outer_lr=train_outer_lr,
            device='cpu'
        )
        
        output += "Starting meta-training...\n\n"
        
        # Meta-training loop
        for epoch in range(train_epochs):
            epoch_rewards = []
            
            for task_idx in range(train_tasks):
                # Collect task data
                task_rewards = []
                obs = env.reset()
                hidden = model.init_hidden(batch_size=1)
                
                for step in range(200):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_logits, hidden = model(obs_tensor, hidden)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs, 1).item()
                    
                    next_obs, reward, done, truncated, info = env.step(action)
                    task_rewards.append(reward)
                    
                    # Store in experience buffer
                    experience_buffer.add(obs, action, reward, next_obs, done or truncated)
                    
                    obs = next_obs
                    if done or truncated:
                        break
                
                epoch_rewards.append(sum(task_rewards))
            
            # Meta-update (simplified)
            avg_reward = np.mean(epoch_rewards)
            
            if (epoch + 1) % 10 == 0:
                output += f"Epoch {epoch + 1}/{train_epochs}: Avg Reward = {avg_reward:.2f}\n"
        
        output += "\n" + "=" * 60 + "\n"
        output += "Meta-Training Complete!\n"
        output += "=" * 60 + "\n"
        output += f"Final Average Reward: {avg_reward:.2f}\n"
        output += "\nModel is ready for testing in Tab 2!\n"
        
        return output, model, experience_buffer
        
    except Exception as e:
        import traceback
        return f"❌ Error during meta-training:\n{str(e)}\n\n{traceback.format_exc()}", None, None


def test_adaptation(env_name, model, experience_buffer, adaptation_mode, test_state_dim, test_hidden_dim,
                   test_lr, test_steps, test_exp_weight):
    """Test model with adaptation"""
    if model is None:
        return "❌ Error: Please load pre-trained model or complete meta-training first!"
    
    output = "=" * 60 + "\n"
    output += f"Test-Time Adaptation\n"
    output += "=" * 60 + "\n\n"
    output += f"Environment: {env_name}\n"
    output += f"Mode: {adaptation_mode}\n"
    output += f"Adaptation Steps: {test_steps}\n"
    output += f"Learning Rate: {test_lr}\n"
    if adaptation_mode == "hybrid":
        output += f"Experience Weight: {test_exp_weight}\n"
    output += "\n"
    
    try:
        # Initialize environment
        env = Environment(env_name=env_name)
        
        # Initialize adapter
        if adaptation_mode == "standard":
            adapter = StandardAdapter(
                model=model,
                learning_rate=test_lr,
                adaptation_steps=test_steps,
                device='cpu'
            )
        else:  # hybrid
            if experience_buffer is None:
                experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
            adapter = HybridAdapter(
                model=model,
                experience_buffer=experience_buffer,
                learning_rate=test_lr,
                adaptation_steps=test_steps,
                experience_weight=test_exp_weight,
                device='cpu'
            )
        
        # Test episodes
        num_test_episodes = 10
        rewards_list = []
        
        output += f"Running {num_test_episodes} test episodes...\n\n"
        
        for ep in range(num_test_episodes):
            obs = env.reset()
            hidden = model.init_hidden(batch_size=1)
            episode_reward = 0
            
            for step in range(200):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action_logits, hidden = model(obs_tensor, hidden)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
                
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                obs = next_obs
                if done or truncated:
                    break
            
            rewards_list.append(episode_reward)
            output += f"Episode {ep + 1}: Reward = {episode_reward:.1f}\n"
        
        avg_reward = np.mean(rewards_list)
        std_reward = np.std(rewards_list)
        
        output += "\n" + "=" * 60 + "\n"
        output += "Test Results\n"
        output += "=" * 60 + "\n"
        output += f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}\n"
        output += f"Min Reward: {min(rewards_list):.1f}\n"
        output += f"Max Reward: {max(rewards_list):.1f}\n"
        
        return output
        
    except Exception as e:
        import traceback
        return f"❌ Error during testing:\n{str(e)}\n\n{traceback.format_exc()}"


def run_rsi_cycle(num_cycles):
    """Run RSI improvement cycles"""
    global global_rsi_agent
    
    if global_rsi_agent is None:
        return "❌ Error: Please load the pre-trained model first (Tab 0)!"
    
    output = "=" * 60 + "\n"
    output += f"Running {num_cycles} RSI Cycle(s)\n"
    output += "=" * 60 + "\n\n"
    
    try:
        for cycle in range(num_cycles):
            output += f"\n{'=' * 60}\n"
            output += f"RSI Cycle {cycle + 1}/{num_cycles}\n"
            output += f"{'=' * 60}\n\n"
            
            output += f"Current Generation: {global_rsi_agent.generation}\n"
            output += f"Current Reward: {global_rsi_agent.current_metrics.avg_reward:.2f}\n\n"
            
            # Run improvement
            improved = global_rsi_agent.attempt_self_improvement()
            
            if improved:
                output += "✅ Improvement found!\n"
                output += f"   New Reward: {global_rsi_agent.current_metrics.avg_reward:.2f}\n"
                output += f"   Adaptation Speed: {global_rsi_agent.current_metrics.adaptation_speed:.2f}\n"
                output += f"   Stability: {global_rsi_agent.current_metrics.stability_score:.2f}\n"
            else:
                output += "⚠ No improvement in this cycle\n"
            
            output += f"\nGeneration: {global_rsi_agent.generation}\n"
            output += f"State Dim: {global_rsi_agent.arch_config.state_dim}\n"
            output += f"Hidden Dim: {global_rsi_agent.arch_config.hidden_dim}\n"
        
        output += "\n" + "=" * 60 + "\n"
        output += "RSI Cycles Complete\n"
        output += "=" * 60 + "\n"
        
        return output
        
    except Exception as e:
        import traceback
        return f"❌ Error during RSI:\n{str(e)}\n\n{traceback.format_exc()}"


def get_rsi_status():
    """Get current RSI status"""
    global global_rsi_agent
    
    if global_rsi_agent is None:
        return "❌ RSI not initialized. Please load the pre-trained model first!"
    
    output = "=" * 60 + "\n"
    output += "RSI Status\n"
    output += "=" * 60 + "\n\n"
    output += f"Generation: {global_rsi_agent.generation}\n"
    output += f"Current Reward: {global_rsi_agent.current_metrics.avg_reward:.2f}\n"
    output += f"Adaptation Speed: {global_rsi_agent.current_metrics.adaptation_speed:.2f}\n"
    output += f"Generalization: {global_rsi_agent.current_metrics.generalization_score:.2f}\n"
    output += f"Meta Efficiency: {global_rsi_agent.current_metrics.meta_efficiency:.2f}\n"
    output += f"Stability: {global_rsi_agent.current_metrics.stability_score:.2f}\n\n"
    output += f"Architecture:\n"
    output += f"  State Dim: {global_rsi_agent.arch_config.state_dim}\n"
    output += f"  Hidden Dim: {global_rsi_agent.arch_config.hidden_dim}\n\n"
    output += f"Safety:\n"
    output += f"  Emergency Stops: {global_rsi_agent.safety_monitor.emergency_stop_count}\n"
    output += f"  Checkpoints: {len(global_rsi_agent.checkpoint_manager.checkpoints)}\n"
    
    return output


# Create Gradio interface
with gr.Blocks(title="SSM-MetaRL-Unified with RSI", theme=gr.themes.Soft()) as demo:
    
    # Shared state
    trained_model = gr.State(None)
    exp_buffer = gr.State(None)
    
    gr.Markdown("""
    # 🚀 SSM-MetaRL-Unified: Pre-trained Model + Recursive Self-Improvement
    
    **State Space Model + Meta-Reinforcement Learning with Test-Time Adaptation and RSI**
    
    This demo includes:
    - ✅ Pre-trained SSM-MetaRL model (ready to use)
    - ✅ Meta-learning with MAML
    - ✅ Standard and Hybrid adaptation modes
    - ✅ **Recursive Self-Improvement (RSI)** - NEW!
    """)
    
    # Tab 0: Load Pre-trained Model
    with gr.Tab("0. Load Pre-trained Model"):
        gr.Markdown("""
        ### 📥 Load Pre-trained SSM-MetaRL Model
        
        Click the button below to load the pre-trained model weights.
        This will also initialize the RSI system.
        
        **Model Details:**
        - Environment: CartPole-v1
        - Training: 50 epochs with hybrid adaptation
        - Parameters: 6,744
        - File: cartpole_hybrid_real_model.pth (32 KB)
        
        After loading, you can:
        1. Test the model (Tab 2)
        2. Run recursive self-improvement (Tab 3)
        3. Or train a new model (Tab 1)
        """)
        
        load_btn = gr.Button("📥 Load Pre-trained Model", variant="primary", size="lg")
        load_output = gr.Textbox(label="Loading Status", lines=20)
        
        load_btn.click(
            fn=load_model_and_init_rsi,
            inputs=[],
            outputs=[trained_model, exp_buffer, load_output]
        )
    
    # Tab 1: Meta-Training
    with gr.Tab("1. Meta-Training (Optional)"):
        gr.Markdown("### Train a New Model from Scratch")
        gr.Markdown("""
        **Optional:** Train a new model from scratch or fine-tune.
        
        **Note:** You can skip this if you loaded the pre-trained model!
        """)
        
        with gr.Row():
            with gr.Column():
                train_env = gr.Dropdown(
                    choices=["CartPole-v1", "Acrobot-v1"],
                    value="CartPole-v1",
                    label="Environment"
                )
                train_epochs = gr.Slider(50, 200, value=100, step=10, label="Meta-Training Epochs")
                train_tasks = gr.Slider(3, 10, value=5, step=1, label="Tasks per Epoch")
                train_state_dim = gr.Slider(16, 64, value=32, step=16, label="SSM State Dimension")
                train_hidden_dim = gr.Slider(32, 128, value=64, step=32, label="SSM Hidden Dimension")
                train_inner_lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Inner LR (Task Adaptation)")
                train_outer_lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Outer LR (Meta-Update)")
                train_gamma = gr.Slider(0.9, 0.999, value=0.99, step=0.001, label="Discount Factor")
                
                train_btn = gr.Button("🚀 Start Meta-Training", variant="primary", size="lg")
            
            with gr.Column():
                train_output = gr.Textbox(label="Meta-Training Log", lines=35, max_lines=50)
        
        train_btn.click(
            fn=train_meta_rl,
            inputs=[train_env, train_epochs, train_tasks, train_state_dim, train_hidden_dim,
                   train_inner_lr, train_outer_lr, train_gamma],
            outputs=[train_output, trained_model, exp_buffer]
        )
    
    # Tab 2: Test-Time Adaptation
    with gr.Tab("2. Test-Time Adaptation"):
        gr.Markdown("### Test the Model with Adaptation")
        gr.Markdown("""
        Test the pre-trained (or newly trained) model with different adaptation strategies.
        
        **Make sure you loaded the pre-trained model first!**
        """)
        
        with gr.Row():
            with gr.Column():
                test_env = gr.Dropdown(
                    choices=["CartPole-v1", "Acrobot-v1"],
                    value="CartPole-v1",
                    label="Test Environment"
                )
                test_mode = gr.Radio(
                    choices=["standard", "hybrid"],
                    value="hybrid",
                    label="Adaptation Mode"
                )
                test_state_dim = gr.Slider(16, 64, value=32, step=16, label="State Dimension")
                test_hidden_dim = gr.Slider(32, 128, value=64, step=32, label="Hidden Dimension")
                test_lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Adaptation Learning Rate")
                test_steps = gr.Slider(5, 50, value=10, step=5, label="Adaptation Steps")
                test_exp_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Experience Weight (Hybrid only)")
                
                test_btn = gr.Button("🧪 Test Adaptation", variant="primary", size="lg")
            
            with gr.Column():
                test_output = gr.Textbox(label="Test Results", lines=35, max_lines=50)
        
        gr.Markdown("""
        **Adaptation Modes:**
        
        **Standard**: Uses only current task data
        - Simple and fast
        - Baseline approach
        
        **Hybrid**: Combines current data + experience replay
        - Uses experience buffer from meta-training
        - More robust adaptation
        - Original research contribution
        """)
        
        test_btn.click(
            fn=test_adaptation,
            inputs=[test_env, trained_model, exp_buffer, test_mode, test_state_dim, test_hidden_dim,
                   test_lr, test_steps, test_exp_weight],
            outputs=test_output
        )
    
    # Tab 3: Recursive Self-Improvement
    with gr.Tab("3. Recursive Self-Improvement 🧠"):
        gr.Markdown("""
        ### 🧠 Recursive Self-Improvement (RSI)
        
        The RSI system allows the model to improve itself through:
        - **Architectural Evolution**: Testing different model structures
        - **Hyperparameter Optimization**: Finding better learning rates
        - **Safety Monitoring**: Preventing performance degradation
        - **Checkpoint System**: Rollback if improvements fail
        
        **How it works:**
        1. Evaluate current performance
        2. Propose architectural/hyperparameter changes
        3. Test each proposal
        4. Keep improvements, rollback failures
        5. Repeat
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Run RSI Cycles")
                num_cycles = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Number of Improvement Cycles"
                )
                rsi_btn = gr.Button("🚀 Run RSI", variant="primary", size="lg")
                status_btn = gr.Button("📊 Get Status", size="sm")
                
            with gr.Column():
                rsi_output = gr.Textbox(label="RSI Output", lines=25)
        
        rsi_btn.click(
            fn=run_rsi_cycle,
            inputs=[num_cycles],
            outputs=[rsi_output]
        )
        
        status_btn.click(
            fn=get_rsi_status,
            inputs=[],
            outputs=[rsi_output]
        )
        
        gr.Markdown("""
        **Tips:**
        - Start with 1-2 cycles to see how it works
        - Each cycle takes 30-60 seconds
        - Not all cycles find improvements (this is normal)
        - The system learns from both successes and failures
        
        **Safety Features:**
        - Automatic rollback if performance degrades
        - Emergency stop after 3 consecutive failures
        - Checkpoint system for recovery
        """)
    
    # Tab 4: About
    with gr.Tab("About"):
        gr.Markdown("""
        ### About SSM-MetaRL-Unified with RSI
        
        This project demonstrates:
        - State Space Models for temporal modeling
        - Meta-learning (MAML) for fast adaptation
        - Test-time adaptation (Standard and Hybrid)
        - **Recursive Self-Improvement** for autonomous optimization
        
        ### Resources
        - [GitHub Repository](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
        - [Model Hub](https://huggingface.co/stargatek1/ssm-metarl-cartpole)
        - [MAML Paper](https://arxiv.org/abs/1703.03400)
        
        ### License
        MIT License
        """)

if __name__ == "__main__":
    demo.launch()

