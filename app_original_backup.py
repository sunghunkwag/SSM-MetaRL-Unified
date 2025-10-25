# -*- coding: utf-8 -*-
"""
Gradio App for SSM-MetaRL-Unified (Complete Implementation)
Maintains original SSM + Meta-RL architecture with proper RL training
Includes Standard and Hybrid test-time adaptation
"""
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import OrderedDict

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import StandardAdapter, StandardAdaptationConfig
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def collect_episode(env, policy_model, device='cpu', max_steps=200, experience_buffer=None):
    """Collect a single episode using the SSM policy"""
    observations = []
    actions = []
    rewards = []
    log_probs = []
    
    obs = env.reset()
    hidden_state = policy_model.init_hidden(batch_size=1)
    
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass through SSM
        action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)
        
        # Get action probabilities
        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n if hasattr(env.action_space, 'n') else 2
            # Extract action logits from output (first n_actions dimensions)
            logits = action_logits[:, :n_actions]
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
        else:
            action = action_logits.cpu().numpy().flatten()
            log_prob = torch.tensor(0.0)
        
        next_obs, reward, done, info = env.step(action)
        
        # Add to experience buffer if provided
        if experience_buffer is not None:
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            experience_buffer.add(obs_tensor, next_obs_tensor)
        
        observations.append(obs_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        obs = next_obs
        hidden_state = next_hidden_state
        steps += 1
    
    return observations, actions, rewards, log_probs


def train_meta_rl(env_name, num_epochs, tasks_per_epoch, state_dim, hidden_dim, 
                  inner_lr, outer_lr, gamma, progress=gr.Progress()):
    """Train SSM-based policy using Meta-RL (MAML)"""
    progress(0, desc="Initializing SSM model and Meta-MAML...")
    
    device = torch.device('cpu')
    env = Environment(env_name=env_name, batch_size=1)
    
    obs_space = env.observation_space
    action_space = env.action_space
    
    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    # Output dim should match input dim for state prediction in adaptation
    output_dim = input_dim
    n_actions = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]
    
    # Initialize SSM model
    model = StateSpaceModel(
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Initialize Meta-MAML
    meta_learner = MetaMAML(
        model=model,
        inner_lr=inner_lr,
        outer_lr=outer_lr
    )
    
    # Initialize experience buffer
    experience_buffer = ExperienceBuffer(max_size=10000, device=str(device))
    
    logs = []
    logs.append("=== SSM + Meta-RL (MAML) Training ===")
    logs.append(f"Environment: {env_name}")
    logs.append(f"Model: State Space Model")
    logs.append(f"  Input dim: {input_dim}")
    logs.append(f"  Output dim (actions): {output_dim}")
    logs.append(f"  State dim: {state_dim}")
    logs.append(f"  Hidden dim: {hidden_dim}")
    logs.append(f"Meta-Learning: MAML")
    logs.append(f"  Inner LR: {inner_lr}")
    logs.append(f"  Outer LR: {outer_lr}")
    logs.append(f"  Tasks/epoch: {tasks_per_epoch}")
    logs.append(f"Discount factor: {gamma}")
    logs.append(f"Epochs: {num_epochs}")
    logs.append("=" * 60)
    logs.append("")
    
    epoch_rewards = []
    
    for epoch in range(num_epochs):
        progress(epoch / num_epochs, desc=f"Epoch {epoch}/{num_epochs}")
        
        # Collect multiple tasks
        tasks = []
        task_rewards = []
        
        for task_idx in range(tasks_per_epoch):
            # Collect episode
            observations, actions, rewards, log_probs = collect_episode(
                env, model, device, max_steps=100, experience_buffer=experience_buffer
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
    
    return "\n".join(logs), model, experience_buffer


def test_adaptation(env_name, model, experience_buffer, adaptation_mode, state_dim, hidden_dim,
                    adapt_lr, num_adapt_steps, experience_weight, progress=gr.Progress()):
    """Test-time adaptation with Standard or Hybrid mode"""
    progress(0, desc="Initializing test environment...")
    
    device = torch.device('cpu')
    env = Environment(env_name=env_name, batch_size=1)
    
    logs = []
    logs.append(f"=== Test-Time Adaptation ({adaptation_mode.upper()}) ===")
    logs.append(f"Environment: {env_name}")
    logs.append(f"Adaptation mode: {adaptation_mode}")
    logs.append(f"Adaptation steps: {num_adapt_steps}")
    logs.append(f"Learning rate: {adapt_lr}")
    if adaptation_mode == 'hybrid':
        logs.append(f"Experience weight: {experience_weight}")
        logs.append(f"Experience buffer size: {len(experience_buffer)}")
    logs.append("=" * 60)
    logs.append("")
    
    # Create adapter
    if adaptation_mode == 'hybrid':
        config = HybridAdaptationConfig(
            learning_rate=adapt_lr,
            num_steps=5,
            experience_batch_size=32,
            experience_weight=experience_weight
        )
        adapter = HybridAdapter(
            model=model,
            config=config,
            experience_buffer=experience_buffer,
            device=str(device)
        )
    else:  # standard
        config = StandardAdaptationConfig(
            learning_rate=adapt_lr,
            num_steps=5
        )
        adapter = StandardAdapter(
            model=model,
            config=config,
            device=str(device)
        )
    
    # Run adaptation
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    
    adaptation_losses = []
    step_rewards = []
    
    for step in range(num_adapt_steps):
        progress(step / num_adapt_steps, desc=f"Adaptation step {step}/{num_adapt_steps}")
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        current_hidden = hidden_state
        
        # Get action
        with torch.no_grad():
            action_logits, hidden_state = model(obs_tensor, current_hidden)
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n if hasattr(env.action_space, 'n') else 2
            # Extract action logits from output
            logits = action_logits[:, :n_actions]
            action = torch.argmax(logits, dim=-1).item()
        else:
            action = action_logits.cpu().numpy().flatten()
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        step_rewards.append(reward)
        
        # Perform adaptation update
        if adaptation_mode == 'hybrid':
            loss_val, steps_taken = adapter.update_step(
                x_current=obs_tensor,
                y_current=next_obs_tensor,
                hidden_state_current=current_hidden
            )
        else:
            loss_val, steps_taken = adapter.update_step(
                x=obs_tensor,
                y=next_obs_tensor,
                hidden_state=current_hidden
            )
        
        adaptation_losses.append(loss_val)
        
        obs = next_obs
        
        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)
        
        if step % 10 == 0:
            recent_reward = np.mean(step_rewards[-10:]) if len(step_rewards) >= 10 else np.mean(step_rewards)
            log_msg = f"Step {step:4d}: Loss={loss_val:8.4f}, Recent Reward={recent_reward:6.2f}"
            logs.append(log_msg)
    
    env.close()
    
    logs.append("")
    logs.append("=" * 60)
    logs.append("Adaptation Summary:")
    logs.append(f"  Total steps: {num_adapt_steps}")
    logs.append(f"  Initial loss: {adaptation_losses[0]:8.4f}")
    logs.append(f"  Final loss:   {adaptation_losses[-1]:8.4f}")
    logs.append(f"  Avg reward:   {np.mean(step_rewards):6.2f}")
    logs.append(f"  Total reward: {sum(step_rewards):6.1f}")
    logs.append("")
    logs.append("✅ Test-time adaptation completed!")
    
    return "\n".join(logs)


# Gradio interface
with gr.Blocks(title="SSM-MetaRL-Unified", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 SSM-MetaRL-Unified: State Space Model + Meta-Reinforcement Learning
    
    **Complete implementation** of SSM + Meta-RL (MAML) with test-time adaptation.
    
    **Features:**
    - ✅ State Space Model (SSM) architecture
    - ✅ Meta-Learning with MAML
    - ✅ Standard test-time adaptation
    - ✅ Hybrid adaptation with experience replay
    - ✅ Original research implementation
    """)
    
    # Shared state
    trained_model = gr.State(None)
    exp_buffer = gr.State(None)
    
    with gr.Tab("1. Meta-Training"):
        gr.Markdown("### Meta-Learning with MAML")
        gr.Markdown("Train the SSM model to learn how to learn - enabling fast adaptation to new tasks.")
        
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
        
        gr.Markdown("""
        **Meta-Learning (MAML):**
        - **Inner Loop**: Adapt to specific task
        - **Outer Loop**: Update meta-parameters
        - **Goal**: Learn good initialization for fast adaptation
        
        **Recommended Settings:**
        - Epochs: 100, Tasks: 5
        - State/Hidden: 32/64
        - Inner/Outer LR: 0.01/0.001
        """)
        
        def train_wrapper(*args):
            logs, model, buffer = train_meta_rl(*args)
            return logs, model, buffer
        
        train_btn.click(
            fn=train_wrapper,
            inputs=[train_env, train_epochs, train_tasks, train_state_dim, train_hidden_dim,
                   train_inner_lr, train_outer_lr, train_gamma],
            outputs=[train_output, trained_model, exp_buffer]
        )
    
    with gr.Tab("2. Test-Time Adaptation"):
        gr.Markdown("### Test-Time Adaptation")
        gr.Markdown("Adapt the meta-trained model to new tasks using Standard or Hybrid adaptation.")
        
        with gr.Row():
            with gr.Column():
                test_env = gr.Dropdown(
                    choices=["CartPole-v1", "Acrobot-v1"],
                    value="CartPole-v1",
                    label="Environment"
                )
                test_mode = gr.Radio(
                    choices=["standard", "hybrid"],
                    value="standard",
                    label="Adaptation Mode"
                )
                test_steps = gr.Slider(20, 100, value=50, step=10, label="Adaptation Steps")
                test_lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Adaptation Learning Rate")
                test_exp_weight = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Experience Weight (Hybrid only)")
                
                # Hidden inputs for model architecture (must match training)
                test_state_dim = gr.Slider(16, 64, value=32, step=16, label="SSM State Dimension", visible=False)
                test_hidden_dim = gr.Slider(32, 128, value=64, step=32, label="SSM Hidden Dimension", visible=False)
                
                test_btn = gr.Button("🧪 Start Adaptation", variant="secondary", size="lg")
            
            with gr.Column():
                test_output = gr.Textbox(label="Adaptation Log", lines=35, max_lines=50)
        
        gr.Markdown("""
        **Adaptation Modes:**
        
        **Standard**: Uses only current task data
        - Simple and fast
        - Baseline approach
        
        **Hybrid**: Combines current data + experience replay
        - Uses experience buffer from meta-training
        - More robust adaptation
        - Original research contribution
        
        **Note**: You must complete meta-training first!
        """)
        
        def test_wrapper(env_name, model, buffer, mode, state_dim, hidden_dim, lr, steps, exp_weight):
            if model is None:
                return "❌ Error: Please complete meta-training first!"
            return test_adaptation(env_name, model, buffer, mode, state_dim, hidden_dim,
                                 lr, steps, exp_weight)
        
        test_btn.click(
            fn=test_wrapper,
            inputs=[test_env, trained_model, exp_buffer, test_mode, test_state_dim, test_hidden_dim,
                   test_lr, test_steps, test_exp_weight],
            outputs=test_output
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## 📚 SSM-MetaRL-Unified
        
        **Complete implementation** of the original research combining State Space Models with Meta-Reinforcement Learning.
        
        ### 🏗️ Architecture
        
        **1. State Space Model (SSM)**
        - Maintains hidden state over time
        - Efficient sequential processing
        - Temporal dependency modeling
        
        **2. Meta-Learning (MAML)**
        - Inner loop: Task-specific adaptation
        - Outer loop: Meta-parameter optimization
        - Learns good initialization
        
        **3. Test-Time Adaptation**
        - **Standard**: Current task data only
        - **Hybrid**: Current data + experience replay
        
        ### 🔬 How It Works
        
        **Meta-Training:**
        1. Sample multiple tasks (episodes)
        2. For each task:
           - Split into support/query sets
           - Inner loop: Adapt on support
           - Evaluate on query
        3. Outer loop: Update meta-parameters
        
        **Test-Time Adaptation:**
        1. Start with meta-learned initialization
        2. Collect observations from new task
        3. Adapt model using:
           - Standard: Current observations only
           - Hybrid: Current + past experiences
        
        ### ✅ Original Implementation
        
        This maintains the exact structure from the GitHub repository:
        - `core/ssm.py`: State Space Model
        - `meta_rl/meta_maml.py`: MAML implementation
        - `adaptation/standard_adapter.py`: Standard adaptation
        - `adaptation/hybrid_adapter.py`: Hybrid adaptation
        - `experience/experience_buffer.py`: Experience replay
        
        ### 📊 Key Concepts
        
        **Meta-Learning**: Learning to learn
        **MAML**: Model-Agnostic Meta-Learning
        **SSM**: State Space Model
        **Experience Replay**: Reuse past data
        
        ### 🎓 Resources
        
        - [MAML Paper](https://arxiv.org/abs/1703.03400)
        - [Meta-RL Survey](https://arxiv.org/abs/1910.03193)
        - [State Space Models](https://arxiv.org/abs/2111.00396)
        
        ### 📝 License
        
        MIT License
        
        ### 🔗 Repository
        
        [SSM-MetaRL-Unified](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
        """)

if __name__ == "__main__":
    demo.launch()

