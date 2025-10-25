"""
Gradio App for SSM-MetaRL-Unified with Recursive Self-Improvement
Includes pre-trained model loading and RSI functionality
"""
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import OrderedDict
import os

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation import StandardAdapter, StandardAdaptationConfig
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment

# Import RSI modules
from recursive_self_improvement import (
    RecursiveSelfImprovementAgent,
    RSIConfig,
    SafetyConfig,
    ArchitecturalConfig,
    LearningConfig
)

# Pre-trained model configuration
PRETRAINED_MODEL_PATH = "cartpole_hybrid_real_model.pth"
PRETRAINED_CONFIG = {
    'env_name': 'CartPole-v1',
    'state_dim': 32,
    'hidden_dim': 64,
    'input_dim': 4,
    'output_dim': 4
}

# Global RSI agent
rsi_agent = None
def load_pretrained_model():
    """
    Load the pre-trained SSM-MetaRL model
    Returns: (model, experience_buffer, logs)
    """
    logs = []
    logs.append("=" * 60)
    logs.append("Loading Pre-trained SSM-MetaRL Model")
    logs.append("=" * 60)
    
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logs.append(f"❌ Pre-trained model not found: {PRETRAINED_MODEL_PATH}")
        logs.append("Please run meta-training first or ensure the model file exists.")
        return None, None, "\n".join(logs)
    
    try:
        # Initialize model architecture
        model = StateSpaceModel(
            state_dim=PRETRAINED_CONFIG['state_dim'],
            input_dim=PRETRAINED_CONFIG['input_dim'],
            output_dim=PRETRAINED_CONFIG['output_dim'],
            hidden_dim=PRETRAINED_CONFIG['hidden_dim']
        )
        
        # Load pre-trained weights
        model.load(PRETRAINED_MODEL_PATH)
        model.eval()
        
        # Initialize experience buffer
        experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        logs.append(f"✅ Successfully loaded pre-trained model!")
        logs.append(f"\nModel Configuration:")
        logs.append(f"  Environment: {PRETRAINED_CONFIG['env_name']}")
        logs.append(f"  State Dimension: {PRETRAINED_CONFIG['state_dim']}")
        logs.append(f"  Hidden Dimension: {PRETRAINED_CONFIG['hidden_dim']}")
        logs.append(f"  Input Dimension: {PRETRAINED_CONFIG['input_dim']}")
        logs.append(f"  Output Dimension: {PRETRAINED_CONFIG['output_dim']}")
        logs.append(f"  Total Parameters: {total_params:,}")
        logs.append(f"\nModel File: {PRETRAINED_MODEL_PATH}")
        logs.append(f"File Size: {os.path.getsize(PRETRAINED_MODEL_PATH):,} bytes")
        logs.append("\n" + "=" * 60)
        logs.append("✅ Pre-trained model ready for testing!")
        logs.append("=" * 60)
        logs.append("\nYou can now:")
        logs.append("  1. Go to 'Test-Time Adaptation' tab")
        logs.append("  2. Select adaptation mode (Standard or Hybrid)")
        logs.append("  3. Test the pre-trained model immediately")
        logs.append("\nNo meta-training required!")
        
        return model, experience_buffer, "\n".join(logs)
        
    except Exception as e:
        logs.append(f"❌ Error loading model: {str(e)}")
        return None, None, "\n".join(logs)


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
    
    if model is None:
        return "❌ Error: Please load pre-trained model or complete meta-training first!"
    
    device = torch.device('cpu')
    env = Environment(env_name=env_name, batch_size=1)
    
    logs = []
    logs.append(f"=== Test-Time Adaptation ({adaptation_mode.upper()}) ===")
    logs.append(f"Environment: {env_name}")
    logs.append(f"Adaptation mode: {adaptation_mode}")
    logs.append(f"Adaptation steps: {num_adapt_steps}")
    
    if adaptation_mode == 'hybrid':
        logs.append(f"Experience weight: {experience_weight}")
        logs.append(f"Experience buffer size: {len(experience_buffer) if experience_buffer else 0}")
    
    logs.append("=" * 60)
    logs.append("")
    
    # Collect initial episodes (before adaptation)
    if adaptation_mode == 'hybrid':
        progress(0.1, desc="Collecting episodes for hybrid adaptation...")
        for _ in range(3):
            collect_episode(env, model, device, max_steps=200, experience_buffer=experience_buffer)
    
    # Test episodes
    num_test_episodes = 10
    test_rewards = []
    
    progress(0.3, desc=f"Running {num_test_episodes} test episodes...")
    
    for ep in range(num_test_episodes):
        progress(0.3 + 0.6 * ep / num_test_episodes, desc=f"Episode {ep+1}/{num_test_episodes}")
        
        # Collect episode
        observations, actions, rewards, log_probs = collect_episode(
            env, model, device, max_steps=500, experience_buffer=experience_buffer
        )
        
        total_reward = sum(rewards)
        test_rewards.append(total_reward)
        
        if ep < 5:  # Log first 5 episodes
            logs.append(f"Episode {ep+1:2d}: Reward = {total_reward:6.1f}, Steps = {len(rewards):3d}")
    
    env.close()
    
    logs.append("")
    logs.append("=" * 60)
    logs.append("Test Results:")
    logs.append(f"  Average Reward: {np.mean(test_rewards):6.2f} ± {np.std(test_rewards):5.2f}")
    logs.append(f"  Min Reward:     {np.min(test_rewards):6.1f}")
    logs.append(f"  Max Reward:     {np.max(test_rewards):6.1f}")
    logs.append("=" * 60)
    logs.append("")
    logs.append("✅ Test-time adaptation completed!")
    
    return "\n".join(logs)


def initialize_rsi(model, experience_buffer):
    """Initialize RSI agent with loaded model"""
    global rsi_agent
    
    logs = []
    logs.append("=" * 60)
    logs.append("Initializing Recursive Self-Improvement System")
    logs.append("=" * 60)
    
    try:
        # Create environment
        env = gym.make('CartPole-v1')
        
        # Configure RSI
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
            state_dim=model.state_dim,
            hidden_dim=model.hidden_dim,
            num_layers=1
        )
        
        learn_config = LearningConfig(
            inner_lr=0.01,
            outer_lr=0.001,
            meta_batch_size=5,
            adaptation_steps=1
        )
        
        # Initialize RSI agent
        rsi_agent = RecursiveSelfImprovementAgent(
            initial_model=model,
            env=env,
            device='cpu',
            safety_config=safety_config,
            rsi_config=rsi_config
        )
        
        # Set configs
        rsi_agent.arch_config = arch_config
        rsi_agent.learn_config = learn_config
        
        logs.append("✅ RSI system initialized successfully!")
        logs.append(f"\nConfiguration:")
        logs.append(f"  - State Dimension: {arch_config.state_dim}")
        logs.append(f"  - Hidden Dimension: {arch_config.hidden_dim}")
        logs.append(f"  - Inner Learning Rate: {learn_config.inner_lr}")
        logs.append(f"  - Outer Learning Rate: {learn_config.outer_lr}")
        logs.append(f"\nSafety Settings:")
        logs.append(f"  - Performance Window: {safety_config.performance_window}")
        logs.append(f"  - Min Threshold: {safety_config.min_performance_threshold}")
        logs.append(f"  - Max Emergency Stops: {safety_config.max_emergency_stops}")
        logs.append("\n" + "=" * 60)
        logs.append("✅ Ready for recursive self-improvement!")
        logs.append("=" * 60)
        
        return "\n".join(logs)
        
    except Exception as e:
        logs.append(f"\n❌ Error initializing RSI: {str(e)}")
        import traceback
        logs.append(traceback.format_exc())
        return "\n".join(logs)

def run_rsi_cycle(num_cycles=1):
    """Run RSI improvement cycles"""
    global rsi_agent
    
    if rsi_agent is None:
        return "❌ RSI not initialized. Please load a model first and initialize RSI."
    
    logs = []
    logs.append("=" * 60)
    logs.append(f"Running {num_cycles} RSI Cycle(s)")
    logs.append("=" * 60)
    
    try:
        for cycle in range(num_cycles):
            logs.append(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")
            logs.append(f"Generation: {rsi_agent.generation}")
            logs.append(f"Current Reward: {rsi_agent.current_metrics.avg_reward:.2f}")
            
            # Run improvement cycle
            improved = rsi_agent.attempt_self_improvement()
            
            if improved:
                logs.append("✅ Improvement found!")
                logs.append(f"  New Reward: {rsi_agent.current_metrics.avg_reward:.2f}")
                logs.append(f"  Adaptation Speed: {rsi_agent.current_metrics.adaptation_speed:.2f}")
                logs.append(f"  Stability: {rsi_agent.current_metrics.stability_score:.2f}")
            else:
                logs.append("⚠ No improvement in this cycle")
                logs.append("  (This is normal - not all cycles find improvements)")
        
        # Summary
        logs.append("\n" + "=" * 60)
        logs.append("RSI Summary")
        logs.append("=" * 60)
        logs.append(f"Total Generations: {rsi_agent.generation}")
        logs.append(f"Current Performance:")
        logs.append(f"  - Reward: {rsi_agent.current_metrics.avg_reward:.2f}")
        logs.append(f"  - Adaptation Speed: {rsi_agent.current_metrics.adaptation_speed:.2f}")
        logs.append(f"  - Generalization: {rsi_agent.current_metrics.generalization_score:.2f}")
        logs.append(f"  - Meta Efficiency: {rsi_agent.current_metrics.meta_learning_efficiency:.2f}")
        logs.append(f"  - Stability: {rsi_agent.current_metrics.stability_score:.2f}")
        
        # Checkpoints
        num_checkpoints = len(rsi_agent.checkpoint_system.checkpoints)
        logs.append(f"\nCheckpoints: {num_checkpoints}")
        
        logs.append("\n" + "=" * 60)
        logs.append("✅ RSI cycles completed!")
        logs.append("=" * 60)
        
        return "\n".join(logs)
        
    except Exception as e:
        logs.append(f"\n❌ Error during RSI: {str(e)}")
        import traceback
        logs.append(traceback.format_exc())
        return "\n".join(logs)

def get_rsi_status():
    """Get current RSI status"""
    global rsi_agent
    
    if rsi_agent is None:
        return "RSI not initialized"
    
    status = []
    status.append("=" * 60)
    status.append("RSI Status")
    status.append("=" * 60)
    status.append(f"\nGeneration: {rsi_agent.generation}")
    status.append(f"\nCurrent Performance:")
    status.append(f"  - Average Reward: {rsi_agent.current_metrics.avg_reward:.2f}")
    status.append(f"  - Adaptation Speed: {rsi_agent.current_metrics.adaptation_speed:.2f}")
    status.append(f"  - Generalization Score: {rsi_agent.current_metrics.generalization_score:.2f}")
    status.append(f"  - Meta Efficiency: {rsi_agent.current_metrics.meta_learning_efficiency:.2f}")
    status.append(f"  - Stability Score: {rsi_agent.current_metrics.stability_score:.2f}")
    
    status.append(f"\nArchitecture:")
    status.append(f"  - State Dim: {rsi_agent.arch_config.state_dim}")
    status.append(f"  - Hidden Dim: {rsi_agent.arch_config.hidden_dim}")
    
    status.append(f"\nSafety:")
    status.append(f"  - Emergency Stops: {rsi_agent.safety_monitor.emergency_stops}")
    status.append(f"  - Checkpoints: {len(rsi_agent.checkpoint_system.checkpoints)}")
    
    
    return "\n".join(status)

# Load existing app functions
def load_model_and_init_rsi():
    """Load pre-trained model and initialize RSI"""
    model, buffer, logs = load_pretrained_model()
    
    if model is not None:
        rsi_logs = initialize_rsi(model, buffer)
        return model, buffer, logs + "\n\n" + rsi_logs
    else:
        return model, buffer, logs

# Create Gradio interface
with gr.Blocks(title="SSM-MetaRL-Unified with RSI", theme=gr.themes.Soft()) as demo:
    
    # Shared state
    model_state = gr.State(None)
    buffer_state = gr.State(None)
    
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
            outputs=[model_state, buffer_state, load_output]
        )
    
    # Tab 1: Meta-Training (keep existing)
    with gr.Tab("1. Meta-Training (Optional)"):
        gr.Markdown("### Train a new model from scratch (optional)")
        # ... (keep existing meta-training interface)
    
    # Tab 2: Test-Time Adaptation (keep existing)
    with gr.Tab("2. Test-Time Adaptation"):
        gr.Markdown("### Test the loaded model with adaptation")
        # ... (keep existing adaptation interface)
    
    # Tab 3: Recursive Self-Improvement (NEW)
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

