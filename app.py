"""
SSM-MetaRL-Unified Gradio Interface
Clean version without RSI - Only Meta-Training and Test-Time Adaptation
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from env_runner.environment import Environment
from experience.experience_buffer import ExperienceBuffer
from adaptation.standard_adapter import StandardAdapter, StandardAdaptationConfig
from adaptation.hybrid_adapter import HybridAdapter, HybridAdaptationConfig


# Global variables for model and buffer
global_model = None
global_experience_buffer = None


def collect_episode(env, model, device, max_steps=200, experience_buffer=None):
    """Collect a single episode using the current model"""
    obs = env.reset()
    hidden = model.init_hidden(batch_size=1)
    
    observations = []
    actions = []
    rewards = []
    log_probs = []
    
    for step in range(max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_logits, hidden = model(obs_tensor, hidden)
            # Use only first 2 dimensions for CartPole (2 actions)
            action_logits_2d = action_logits[:, :2]
            action_probs = F.softmax(action_logits_2d, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        next_obs, reward, done, info = env.step(action.item())
        
        observations.append(obs_tensor)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        
        if experience_buffer is not None:
            experience_buffer.add(obs, action.item(), reward, next_obs, done)
        
        obs = next_obs
        if done:
            break
    
    return observations, actions, rewards, log_probs


def load_pretrained_model():
    """Load the pre-trained model"""
    global global_model, global_experience_buffer
    
    try:
        model_path = Path("models/cartpole_hybrid_real_model.pth")
        
        if not model_path.exists():
            return "❌ Error: Pre-trained model not found at models/cartpole_hybrid_real_model.pth"
        
        # Initialize model with same architecture as training
        global_model = StateSpaceModel(
            state_dim=32,
            input_dim=4,  # CartPole observation space
            output_dim=4,  # Predict next observation
            hidden_dim=64
        )
        
        # Load weights
        global_model.load(str(model_path))
        
        # Initialize experience buffer
        global_experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
        
        output = "✅ Model loaded successfully!\n\n"
        output += f"File: {model_path}\n"
        output += f"State Dim: 32\n"
        output += f"Hidden Dim: 64\n"
        output += f"Parameters: {sum(p.numel() for p in global_model.parameters())}\n\n"
        output += "Ready to Use!\n"
        output += "→ Go to 'Test-Time Adaptation' tab to test the model\n"
        output += "→ Or go to 'Meta-Training' tab to train a new model\n"
        
        return output
        
    except Exception as e:
        import traceback
        return f"❌ Error loading model:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"


def train_meta_rl(env_name, train_epochs, train_tasks, train_state_dim, train_hidden_dim,
                  train_inner_lr, train_outer_lr, train_gamma, progress=gr.Progress()):
    """Train a new meta-learning model from scratch"""
    global global_model, global_experience_buffer
    
    progress(0, desc="Initializing meta-training...")
    
    logs = []
    logs.append("=== Meta-Training Started ===")
    logs.append(f"Environment: {env_name}")
    logs.append(f"Epochs: {train_epochs}")
    logs.append(f"Tasks/Epoch: {train_tasks}")
    logs.append(f"State Dim: {train_state_dim}, Hidden Dim: {train_hidden_dim}")
    logs.append(f"Inner LR: {train_inner_lr}, Outer LR: {train_outer_lr}")
    logs.append(f"Gamma: {train_gamma}\n\n")
    
    try:
        # Initialize environment
        device = torch.device('cpu')
        env = Environment(env_name=env_name, batch_size=1)
        
        # Get environment dimensions
        obs = env.reset()
        input_dim = len(obs)
        output_dim = input_dim  # Predict next observation
        n_actions = env.action_space.n
        
        # Initialize model
        model = StateSpaceModel(
            state_dim=train_state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=train_hidden_dim
        )
        model.to(device)
        
        # Initialize meta-learner (NO device parameter)
        meta_learner = MetaMAML(
            model=model,
            inner_lr=train_inner_lr,
            outer_lr=train_outer_lr
        )
        
        # Initialize experience buffer
        experience_buffer = ExperienceBuffer(max_size=10000, device=device)
        
        logs.append(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
        logs.append("=" * 60)
        logs.append("")
        
        epoch_rewards = []
        
        for epoch in range(train_epochs):
            progress(epoch / train_epochs, desc=f"Epoch {epoch + 1}/{train_epochs}")
            
            tasks = []
            task_rewards = []
            
            for task_idx in range(train_tasks):
                # Collect episode
                observations, actions, rewards, log_probs = collect_episode(
                    env, model, device, max_steps=200, experience_buffer=experience_buffer
                )
                
                task_rewards.append(sum(rewards))
                
                if len(observations) < 10:
                    continue
                
                # Split into support and query sets
                split_idx = len(observations) // 2
                
                # Support set
                support_obs = torch.cat(observations[:split_idx], dim=0).unsqueeze(0)
                support_actions = torch.tensor(actions[:split_idx], dtype=torch.long).unsqueeze(0)
                
                # Query set
                query_obs = torch.cat(observations[split_idx:], dim=0).unsqueeze(0)
                query_actions = torch.tensor(actions[split_idx:], dtype=torch.long).unsqueeze(0)
                
                # Use action prediction as supervised task for MAML
                support_y = support_actions.float().unsqueeze(-1)
                query_y = query_actions.float().unsqueeze(-1)
                
                tasks.append((support_obs, support_y, query_obs, query_y))
            
            if len(tasks) == 0:
                continue
            
            # Meta-update
            initial_hidden = model.init_hidden(batch_size=1)
            
            def action_prediction_loss(pred, target):
                # pred: (batch, time, output_dim) where output_dim >= n_actions
                # Extract first n_actions dimensions for action prediction
                action_logits = pred[:, :, :n_actions]
                target_long = target.long().squeeze(-1)
                return F.cross_entropy(action_logits.reshape(-1, n_actions), target_long.reshape(-1))
            
            meta_loss = meta_learner.meta_update(
                tasks,
                initial_hidden_state=initial_hidden,
                loss_fn=action_prediction_loss
            )
            
            avg_reward = np.mean(task_rewards) if task_rewards else 0
            epoch_rewards.append(avg_reward)
            
            if epoch % 10 == 0:
                recent_avg = np.mean(epoch_rewards[-10:]) if len(epoch_rewards) >= 10 else avg_reward
                log_msg = f"Epoch {epoch:4d}: Meta-Loss={meta_loss:8.4f}, Avg Reward={avg_reward:6.1f}, Recent={recent_avg:6.1f}, Buffer={len(experience_buffer)}"
                logs.append(log_msg)
        
        env.close()
        
        logs.append("")
        logs.append("=" * 60)
        logs.append("Meta-training completed!")
        logs.append(f"Experience buffer size: {len(experience_buffer)}")
        logs.append("")
        logs.append("Training Summary:")
        logs.append(f"  Initial Avg Reward: {np.mean(epoch_rewards[:10]) if len(epoch_rewards) >= 10 else np.mean(epoch_rewards):6.1f}")
        logs.append(f"  Final Avg Reward:   {np.mean(epoch_rewards[-10:]) if len(epoch_rewards) >= 10 else np.mean(epoch_rewards):6.1f}")
        logs.append(f"  Best Epoch:         {max(epoch_rewards) if epoch_rewards else 0:6.1f}")
        logs.append("")
        logs.append("✅ Meta-learning complete! Model ready for test-time adaptation.")
        
        # Update global variables
        global_model = model
        global_experience_buffer = experience_buffer
        
        return "\n".join(logs), model, experience_buffer
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during meta-training:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logs.append(error_msg)
        return "\n".join(logs), None, None


def test_adaptation(env_name, model, experience_buffer, adaptation_mode, test_state_dim, test_hidden_dim,
                    test_lr, test_steps, test_exp_weight, progress=gr.Progress()):
    """Test-time adaptation with Standard or Hybrid mode"""
    progress(0, desc="Initializing test environment...")
    
    if model is None:
        return "❌ Error: Please load pre-trained model or complete meta-training first!"
    
    try:
        device = torch.device('cpu')
        env = Environment(env_name=env_name, batch_size=1)
        
        logs = []
        logs.append(f"=== Test-Time Adaptation ({adaptation_mode.upper()}) ===")
        logs.append(f"Environment: {env_name}")
        logs.append(f"Adaptation mode: {adaptation_mode}")
        logs.append(f"Adaptation Steps: {test_steps}")
        logs.append(f"Learning Rate: {test_lr}")
        
        if adaptation_mode == 'hybrid':
            logs.append(f"Experience Weight: {test_exp_weight}")
            logs.append(f"Experience buffer size: {len(experience_buffer) if experience_buffer else 0}")
        
        logs.append("=" * 60)
        logs.append("")
        
        # Initialize adapter with config object
        if adaptation_mode == "standard":
            config = StandardAdaptationConfig(
                learning_rate=test_lr,
                num_steps=test_steps
            )
            adapter = StandardAdapter(
                model=model,
                config=config,
                device='cpu'
            )
        else:  # hybrid
            if experience_buffer is None:
                experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
            config = HybridAdaptationConfig(
                learning_rate=test_lr,
                num_steps=test_steps,
                experience_weight=test_exp_weight
            )
            adapter = HybridAdapter(
                model=model,
                config=config,
                experience_buffer=experience_buffer,
                device='cpu'
            )
        
        # Test with 10 episodes
        rewards_list = []
        for ep in range(10):
            progress((ep + 1) / 10, desc=f"Testing episode {ep + 1}/10...")
            
            obs = env.reset()
            hidden = model.init_hidden(batch_size=1)
            episode_reward = 0
            
            for step in range(200):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action_logits, hidden = model(obs_tensor, hidden)
                # Use only first 2 dimensions for CartPole (2 actions)
                action_logits_2d = action_logits[:, :2]
                action_probs = torch.softmax(action_logits_2d, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
                
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                obs = next_obs
                if done:
                    break
            
            rewards_list.append(episode_reward)
            logs.append(f"Episode {ep + 1}: Reward = {episode_reward:.1f}")
        
        env.close()
        
        # Summary
        logs.append("")
        logs.append("=" * 60)
        logs.append("Test Results:")
        logs.append(f"  Average Reward: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
        logs.append(f"  Min Reward:     {min(rewards_list):.1f}")
        logs.append(f"  Max Reward:     {max(rewards_list):.1f}")
        logs.append("")
        logs.append("✅ Testing complete!")
        
        return "\n".join(logs)
        
    except Exception as e:
        import traceback
        return f"❌ Error during testing:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"


# Create Gradio interface
with gr.Blocks(title="SSM-MetaRL-Unified") as demo:
    gr.Markdown("""
    # SSM-MetaRL-Unified: State Space Model + Meta-Reinforcement Learning
    
    **Pre-trained Model Ready!** Load the model and test immediately, or train your own from scratch.
    
    ### Features:
    - ✅ **Pre-trained Model** - CartPole model ready to use
    - ✅ **Meta-Learning (MAML)** - Train models that adapt quickly to new tasks
    - ✅ **Test-Time Adaptation** - Standard and Hybrid modes with experience replay
    - ✅ **Multiple Environments** - CartPole, Acrobot, and more
    
    ### Quick Start:
    1. Click "Load Pre-trained Model" below
    2. Go to "Test-Time Adaptation" tab
    3. Click "Test Adaptation" to see results!
    """)
    
    with gr.Tab("0. Load Pre-trained Model"):
        gr.Markdown("### Load Pre-trained CartPole Model")
        gr.Markdown("Click the button below to load the pre-trained model. This model was trained with 50 epochs of meta-learning.")
        
        load_btn = gr.Button("🚀 Load Pre-trained Model", variant="primary")
        load_output = gr.Textbox(label="Status", lines=15)
        
        load_btn.click(
            fn=load_pretrained_model,
            inputs=[],
            outputs=[load_output]
        )
    
    with gr.Tab("1. Meta-Training (Optional)"):
        gr.Markdown("### Train a New Model from Scratch")
        gr.Markdown("Configure hyperparameters and train a new meta-learning model. This is optional - you can use the pre-trained model instead.")
        
        with gr.Row():
            with gr.Column():
                train_env = gr.Dropdown(
                    choices=["CartPole-v1", "Acrobot-v1"],
                    value="CartPole-v1",
                    label="Environment"
                )
                train_epochs = gr.Slider(50, 200, value=50, step=10, label="Meta-Training Epochs")
                train_tasks = gr.Slider(3, 10, value=5, step=1, label="Tasks per Epoch")
            
            with gr.Column():
                train_state_dim = gr.Slider(16, 64, value=32, step=16, label="State Dimension")
                train_hidden_dim = gr.Slider(32, 128, value=64, step=32, label="Hidden Dimension")
                train_inner_lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Inner Learning Rate")
                train_outer_lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Outer Learning Rate")
                train_gamma = gr.Slider(0.9, 0.999, value=0.99, step=0.001, label="Discount Factor (Gamma)")
        
        train_btn = gr.Button("🚀 Start Meta-Training", variant="primary")
        train_output = gr.Textbox(label="Training Log", lines=35)
        
        # Hidden states to pass model and buffer
        trained_model_state = gr.State()
        trained_buffer_state = gr.State()
        
        train_btn.click(
            fn=train_meta_rl,
            inputs=[train_env, train_epochs, train_tasks, train_state_dim, train_hidden_dim,
                   train_inner_lr, train_outer_lr, train_gamma],
            outputs=[train_output, trained_model_state, trained_buffer_state]
        )
    
    with gr.Tab("2. Test-Time Adaptation"):
        gr.Markdown("### Test Model with Adaptation")
        gr.Markdown("Test the model's ability to adapt to the environment at test time. Choose between Standard (no experience replay) or Hybrid (with experience replay) adaptation.")
        
        with gr.Row():
            with gr.Column():
                test_env = gr.Dropdown(
                    choices=["CartPole-v1", "Acrobot-v1"],
                    value="CartPole-v1",
                    label="Environment"
                )
                test_mode = gr.Radio(
                    choices=["standard", "hybrid"],
                    value="hybrid",
                    label="Adaptation Mode"
                )
                test_state_dim = gr.Slider(16, 64, value=32, step=16, label="State Dimension")
                test_hidden_dim = gr.Slider(32, 128, value=64, step=32, label="Hidden Dimension")
            
            with gr.Column():
                test_lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Adaptation Learning Rate")
                test_steps = gr.Slider(5, 50, value=5, step=5, label="Adaptation Steps")
                test_exp_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Experience Weight (Hybrid only)")
        
        test_btn = gr.Button("🧪 Test Adaptation", variant="primary")
        test_output = gr.Textbox(label="Test Results", lines=25)
        
        def test_with_global_model(env_name, mode, state_dim, hidden_dim, lr, steps, exp_weight):
            return test_adaptation(
                env_name, global_model, global_experience_buffer, mode,
                state_dim, hidden_dim, lr, steps, exp_weight
            )
        
        test_btn.click(
            fn=test_with_global_model,
            inputs=[test_env, test_mode, test_state_dim, test_hidden_dim, test_lr, test_steps, test_exp_weight],
            outputs=[test_output]
        )
    
    gr.Markdown("""
    ---
    ### About
    This demo showcases State Space Models (SSM) combined with Meta-Reinforcement Learning (Meta-RL) for fast adaptation.
    
    **GitHub**: [sunghunkwag/SSM-MetaRL-Unified](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
    
    **Model Hub**: [stargatek1/ssm-metarl-cartpole](https://huggingface.co/stargatek1/ssm-metarl-cartpole)
    """)


if __name__ == "__main__":
    demo.launch()

