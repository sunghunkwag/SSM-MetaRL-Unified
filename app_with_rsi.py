"""
Gradio App for SSM-MetaRL-Unified with Recursive Self-Improvement
Includes pre-trained model loading and RSI functionality
"""
import gradio as gr
import torch
import gymnasium as gym
import os
import sys

# Import existing modules
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

# Import existing functions from original app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import (
    load_pretrained_model,
    train_meta_rl,
    test_adaptation
)

# Global RSI agent
rsi_agent = None

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
    
    status.append("\n" + "=" * 60)
    
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

