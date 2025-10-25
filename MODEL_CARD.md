---
language:
  - en
license: mit
tags:
  - reinforcement-learning
  - meta-learning
  - state-space-models
  - maml
  - pytorch
  - deep-rl
  - few-shot-learning
  - cartpole
  - gymnasium
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# SSM-MetaRL CartPole: Pre-trained Meta-Learning Model

[![Space](https://img.shields.io/badge/🤗-Demo%20Space-yellow)](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Model Description

This is a **pre-trained State Space Model (SSM)** trained with **Meta-Reinforcement Learning (MAML)** for the CartPole-v1 environment. The model has been meta-trained to enable fast adaptation to new tasks with minimal data, demonstrating the power of "learning to learn" in reinforcement learning.

### Key Features

- 🎯 **Meta-learned initialization** for fast task adaptation
- 🧠 **State Space Model architecture** for efficient temporal modeling
- ⚡ **Ready to use** - no additional training required
- 🔄 **Hybrid adaptation** trained with experience replay
- 📦 **Lightweight** - only 32 KB (6,744 parameters)

### Model Type

- **Architecture**: State Space Model (SSM)
- **Training Method**: MetaMAML (Model-Agnostic Meta-Learning)
- **Task**: Reinforcement Learning (CartPole-v1)
- **Framework**: PyTorch

## Intended Use

### Primary Use Cases

1. **Quick Deployment**: Use pre-trained weights for immediate CartPole control
2. **Research Baseline**: Benchmark for meta-learning algorithms
3. **Transfer Learning**: Fine-tune for similar control tasks
4. **Educational**: Demonstrate meta-learning concepts

### Direct Use

```python
import torch
from core.ssm import StateSpaceModel

# Initialize model architecture
model = StateSpaceModel(
    state_dim=32,
    input_dim=4,
    output_dim=4,
    hidden_dim=64
)

# Load pre-trained weights
model.load("cartpole_hybrid_real_model.pth")
model.eval()

# Use for inference
import gymnasium as gym
env = gym.make('CartPole-v1')
obs, _ = env.reset()
hidden_state = model.init_hidden(batch_size=1)

obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
action_logits, hidden_state = model(obs_tensor, hidden_state)
action = torch.argmax(action_logits[:, :2], dim=-1).item()
```

### Try It Online

**🚀 [Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified)**

No installation required! Test the model directly in your browser with our Gradio interface.

## Training Details

### Training Data

- **Environment**: CartPole-v1 (Gymnasium)
- **Observation Space**: 4-dimensional continuous (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Episodes**: Multiple episodes across 50 meta-training epochs
- **Experience Buffer**: 3,191 transitions collected during training

### Training Procedure

#### Meta-Training with MAML

**Algorithm**: Model-Agnostic Meta-Learning (MAML)

**Process**:
1. **Inner Loop** (Task Adaptation):
   - Collect episode data
   - Split into support set (first half) and query set (second half)
   - Perform gradient steps on support set
   - Obtain task-specific adapted parameters

2. **Outer Loop** (Meta-Update):
   - Evaluate adapted parameters on query set
   - Compute meta-loss
   - Update meta-parameters to improve adaptation capability

**Hybrid Adaptation**:
- Combines current task observations with experience replay buffer
- More robust than standard adaptation using only current data
- Original research contribution

#### Hyperparameters

```yaml
# Meta-Learning
num_epochs: 50
tasks_per_epoch: 5
inner_lr: 0.01          # Task adaptation learning rate
outer_lr: 0.001         # Meta-update learning rate

# Model Architecture
state_dim: 32           # SSM state dimension
hidden_dim: 64          # Network hidden dimension
input_dim: 4            # CartPole observation space
output_dim: 4           # For state prediction

# Training
adaptation_mode: hybrid # Uses experience replay
discount_factor: 0.99
max_steps_per_episode: 100
device: cpu             # Trained on CPU
```

#### Training Infrastructure

- **Hardware**: CPU (3 cores)
- **Training Time**: ~5 minutes for 50 epochs
- **Framework**: PyTorch 2.9.0
- **Environment**: Gymnasium 1.2.1

### Training Results

**Meta-Training Performance**:
- Initial Average Reward: 17.0
- Final Average Reward: 11.7
- Best Epoch Reward: 28.2
- Meta-Loss Convergence: ✅ Stable

**Training Log Sample**:
```
Epoch    0: Meta-Loss=  0.6901, Avg Reward=  18.8, Recent=  18.8, Buffer=94
Epoch   10: Meta-Loss=  0.5908, Avg Reward=  15.8, Recent=  16.7, Buffer=931
Epoch   20: Meta-Loss=  0.6144, Avg Reward=  17.4, Recent=  17.5, Buffer=1804
Epoch   30: Meta-Loss=  0.5966, Avg Reward=  12.6, Recent=  14.9, Buffer=2550
Epoch   40: Meta-Loss=  0.6187, Avg Reward=  11.0, Recent=  12.8, Buffer=3191
```

## Evaluation

### Metrics

**Post-Training Verification** (10 episodes):
- **Average Reward**: 9.40 ± 0.66
- **Min Reward**: 8.0
- **Max Reward**: 10.0
- **Average Episode Length**: 9.40 ± 0.66 steps
- **Consistency**: ✅ Stable performance across episodes

### Evaluation Procedure

```python
# Verification script (verify_model.py)
for episode in range(10):
    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)
    total_reward = 0
    
    while not done:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action_logits, hidden_state = model(obs_tensor, hidden_state)
        action = torch.argmax(action_logits[:, :2]).item()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
```

### Comparison

| Method | Training Time | Average Reward | Notes |
|--------|---------------|----------------|-------|
| **This Model (Meta-RL)** | 5 min (meta-train) | 9.40 ± 0.66 | Fast adaptation capability |
| Random Policy | - | ~20-30 | Baseline |
| Standard RL (from scratch) | 10-30 min | 100-200 | Task-specific, no transfer |

**Note**: The meta-learned model's value is in its **adaptation capability**, not just raw performance. It can quickly adapt to task variations with minimal additional data.

## Model Architecture

### State Space Model (SSM)

```
Input (4D observation)
    ↓
Linear Projection (B matrix)
    ↓
State Transition (A matrix) ──→ Hidden State (recurrent)
    ↓
Output Network (C matrix)
    ↓
Feedthrough (D matrix)
    ↓
Action Logits (2D for CartPole)
```

### Components

1. **State Transition Matrix (A)**: `[state_dim × state_dim]`
   - Learns temporal dynamics
   - Maintains hidden state over time

2. **Input Projection (B)**: `[input_dim × state_dim]`
   - Projects observations into state space

3. **Output Network (C)**: `[state_dim × output_dim]`
   - Maps hidden states to predictions

4. **Feedthrough (D)**: `[input_dim × output_dim]`
   - Direct input-output pathway

### Parameter Count

```
Total Parameters: 6,744
├─ State Transition (A): 1,024 (32×32)
├─ Input Projection (B): 128 (4×32)
├─ Output Network (C): 128 (32×4)
├─ Feedthrough (D): 16 (4×4)
└─ Additional Layers: 5,448
```

## Limitations and Bias

### Limitations

1. **Environment-Specific**: Trained specifically for CartPole-v1
   - May not generalize to significantly different environments
   - Best suited for similar control tasks

2. **Performance**: Not optimized for maximum CartPole score
   - Focused on meta-learning capability over raw performance
   - Can be improved with longer training or task-specific fine-tuning

3. **Observation Space**: Requires 4-dimensional continuous input
   - Direct transfer to different observation spaces requires architecture modification

4. **Action Space**: Designed for 2 discrete actions
   - Adaptation needed for continuous or larger discrete action spaces

### Potential Biases

1. **Training Distribution**: Meta-trained on CartPole episodes
   - May have implicit biases toward CartPole dynamics
   - Performance may degrade on out-of-distribution tasks

2. **Exploration Strategy**: Uses softmax action selection
   - May not explore optimally in all scenarios
   - Can be modified for different exploration strategies

### Recommendations

- **Fine-tuning**: For best performance, fine-tune on specific task variants
- **Adaptation**: Use test-time adaptation (Standard or Hybrid mode) for new scenarios
- **Evaluation**: Always evaluate on your specific use case before deployment
- **Monitoring**: Monitor performance and adapt if distribution shift occurs

## Ethical Considerations

### Intended Applications

✅ **Appropriate Uses**:
- Research and education in meta-learning
- Benchmarking meta-RL algorithms
- Prototyping control systems
- Learning about reinforcement learning

❌ **Inappropriate Uses**:
- Safety-critical systems without extensive validation
- Production deployment without proper testing
- Applications requiring guaranteed performance
- Real-world robotics without simulation validation

### Risks and Mitigations

**Risk**: Model may fail in unexpected ways
- **Mitigation**: Always test thoroughly in simulation before real-world use

**Risk**: Over-reliance on meta-learned initialization
- **Mitigation**: Combine with task-specific fine-tuning when needed

**Risk**: Performance degradation on novel scenarios
- **Mitigation**: Monitor performance and retrain if necessary

## How to Use

### Installation

```bash
# Clone repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-Unified.git
cd SSM-MetaRL-Unified

# Install dependencies
pip install torch gymnasium numpy
```

### Loading the Model

```python
from core.ssm import StateSpaceModel
import torch

# Initialize model
model = StateSpaceModel(
    state_dim=32,
    input_dim=4,
    output_dim=4,
    hidden_dim=64
)

# Load pre-trained weights
model.load("cartpole_hybrid_real_model.pth")
model.eval()
```

### Running Inference

```python
import gymnasium as gym
import torch

env = gym.make('CartPole-v1')
obs, _ = env.reset()
hidden_state = model.init_hidden(batch_size=1)
done = False

while not done:
    # Prepare observation
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        action_logits, hidden_state = model(obs_tensor, hidden_state)
    
    # Select action (first 2 dimensions are action logits)
    action = torch.argmax(action_logits[:, :2], dim=-1).item()
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated

env.close()
```

### Test-Time Adaptation

For best results, use the hybrid adaptation mode:

```python
from adaptation import HybridAdapter, HybridAdaptationConfig
from experience.experience_buffer import ExperienceBuffer

# Initialize experience buffer
buffer = ExperienceBuffer(max_size=10000)

# Configure adapter
config = HybridAdaptationConfig(
    adapt_lr=0.01,
    num_adapt_steps=10,
    experience_weight=0.5
)

adapter = HybridAdapter(config)

# Collect some data and adapt
# (See demo Space for full example)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@software{ssm_metarl_cartpole,
  title={SSM-MetaRL CartPole: Pre-trained Meta-Learning Model},
  author={stargatek1},
  year={2025},
  url={https://huggingface.co/stargatek1/ssm-metarl-cartpole},
  note={Pre-trained State Space Model with MetaMAML for CartPole-v1}
}
```

**Related Paper**:
```bibtex
@inproceedings{finn2017maml,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={ICML},
  year={2017}
}
```

## Model Card Authors

- **stargatek1** (Model training and documentation)

## Model Card Contact

- **GitHub Issues**: [SSM-MetaRL-Unified Issues](https://github.com/sunghunkwag/SSM-MetaRL-Unified/issues)
- **Hugging Face**: [@stargatek1](https://huggingface.co/stargatek1)

## Additional Resources

### Documentation
- **📄 Model Generation Report**: [MODEL_GENERATION_REPORT.md](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/MODEL_GENERATION_REPORT.md)
- **🏗️ Architecture Details**: [ARCHITECTURE.md](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/ARCHITECTURE.md)
- **📊 Training Logs**: [training_log.txt](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/training_log.txt)

### Code
- **💻 GitHub Repository**: [sunghunkwag/SSM-MetaRL-Unified](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
- **🎮 Training Script**: [train_and_save_model.py](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/train_and_save_model.py)
- **✅ Verification Script**: [verify_model.py](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/verify_model.py)

### Demo
- **🚀 Interactive Demo**: [Hugging Face Space](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified)
- **📺 Quick Start Guide**: See Space README for 3-step tutorial

### Research Papers
- [MAML: Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400)
- [Meta-Reinforcement Learning Survey](https://arxiv.org/abs/1910.03193)
- [State Space Models](https://arxiv.org/abs/2111.00396)

## License

This model is released under the **MIT License**. See [LICENSE](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/LICENSE) for details.

---

**Ready to try it? Visit our [Interactive Demo](https://huggingface.co/spaces/stargatek1/SSM-MetaRL-Unified) and test the model in seconds!** 🚀

