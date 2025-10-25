---
title: SSM-MetaRL-Unified
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: mit
tags:
  - reinforcement-learning
  - meta-learning
  - state-space-models
  - maml
  - pytorch
  - gymnasium
  - gradio
  - deep-reinforcement-learning
  - few-shot-learning
  - experience-replay
---

# 🚀 SSM-MetaRL-Unified: State Space Model + Meta-Reinforcement Learning

**Try it now! Pre-trained model ready for immediate testing - no training required!**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
[![Model](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/stargatek1/ssm-metarl-cartpole)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MAML-red)](https://arxiv.org/abs/1703.03400)

## 🎯 What is This?

This Space demonstrates a complete implementation of **State Space Models (SSM)** combined with **Meta-Reinforcement Learning (MAML)** for fast adaptation to new tasks. The key innovation is the **Hybrid Adaptation** mode that combines current task data with experience replay for more robust learning.

### Key Features

- ✅ **Pre-trained Model Available** - Load and test immediately without waiting for training!
- ✅ **State Space Model (SSM)** - Efficient temporal modeling with hidden states
- ✅ **Meta-Learning (MAML)** - Learn to learn for fast adaptation
- ✅ **Standard Adaptation** - Baseline using only current task data
- ✅ **Hybrid Adaptation** - Novel approach combining current data + experience replay (research contribution)
- ✅ **Interactive Demo** - Test different adaptation strategies in real-time

## 🚀 Quick Start (3 Simple Steps!)

### Step 1: Load Pre-trained Model
1. Click on the **"0. Load Pre-trained Model"** tab
2. Click the **"📥 Load Pre-trained Model"** button
3. Wait ~5 seconds for the model to load

### Step 2: Test the Model
1. Go to the **"2. Test-Time Adaptation"** tab
2. Select adaptation mode:
   - **Standard**: Uses only current task observations (baseline)
   - **Hybrid**: Uses current observations + past experiences (recommended)
3. Click **"🧪 Test Adaptation"**

### Step 3: View Results
- See the model's performance in the CartPole environment
- Compare different adaptation strategies
- Observe how meta-learning enables fast adaptation

**That's it! No training required - the pre-trained model is ready to use!**

## 🎓 What You'll Learn

### State Space Models (SSM)
Efficient sequential models that maintain hidden states over time, enabling temporal dependency modeling without the complexity of traditional RNNs.

### Meta-Learning (MAML)
Model-Agnostic Meta-Learning learns good parameter initializations that can quickly adapt to new tasks with minimal data - essentially "learning how to learn."

### Test-Time Adaptation
- **Standard Mode**: Adapts using only current task observations (simple baseline)
- **Hybrid Mode**: Combines current observations with experience replay buffer (more robust, original research contribution)

## 🏗️ Architecture

```
Input (CartPole observations)
    ↓
State Space Model (SSM)
    ├─ State Transition (A matrix)
    ├─ Input Projection (B matrix)
    ├─ Output Network (C matrix)
    └─ Feedthrough (D matrix)
    ↓
Action Selection
    ↓
Environment Interaction
```

### Meta-Learning Process

**Meta-Training (MAML):**
1. Sample multiple tasks (episodes)
2. Inner loop: Adapt on support set
3. Outer loop: Update meta-parameters based on query set
4. Result: Good initialization for fast adaptation

**Test-Time Adaptation:**
1. Start with meta-learned initialization
2. Collect observations from new task
3. Adapt using Standard or Hybrid mode
4. Evaluate performance

## 📊 Model Performance

**Pre-trained Model Specifications:**
- **Environment**: CartPole-v1
- **Training**: 50 epochs with MetaMAML
- **Parameters**: 6,744 trainable parameters
- **File Size**: 32 KB
- **Adaptation Mode**: Hybrid (with experience replay)

**Verification Results:**
- Average Reward: **9.40 ± 0.66**
- Min Reward: 8.0
- Max Reward: 10.0
- Consistency: ✅ Stable across 10 test episodes

## 🔬 Research Background

This implementation is based on:

- **MAML**: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) (Finn et al., 2017)
- **Meta-RL**: [Meta-Reinforcement Learning Survey](https://arxiv.org/abs/1910.03193)
- **State Space Models**: [Efficiently Modeling Long Sequences](https://arxiv.org/abs/2111.00396)

## 💡 Why This Matters

### Traditional RL Problem
Training RL agents from scratch is:
- ⏰ Time-consuming (hours or days)
- 📊 Data-hungry (millions of samples)
- 🎯 Task-specific (no transfer)

### Meta-RL Solution
With meta-learning:
- ⚡ Fast adaptation (minutes or seconds)
- 📉 Sample-efficient (few-shot learning)
- 🔄 Transferable (learns across tasks)

### Hybrid Adaptation Innovation
Our hybrid mode adds:
- 🧠 Experience replay for better adaptation
- 💪 More robust to distribution shift
- 🎯 Improved performance over standard adaptation

## 🎮 Use Cases

- **Robotics**: Quick adaptation to new environments or tasks
- **Game AI**: Fast learning of new game mechanics
- **Control Systems**: Rapid tuning for different operating conditions
- **Research**: Benchmark for meta-learning algorithms

## 📁 What's Inside

### Tabs Overview

**Tab 0: Load Pre-trained Model** ⭐ **Start Here!**
- One-click loading of pre-trained weights
- Instant access to meta-learned model
- Skip the 5-10 minute training wait

**Tab 1: Meta-Training (Optional)**
- Train your own model from scratch
- Customize hyperparameters
- Experiment with different configurations

**Tab 2: Test-Time Adaptation**
- Test the loaded model
- Compare Standard vs Hybrid adaptation
- Evaluate performance

**Tab 3: About**
- Detailed documentation
- Architecture overview
- Research references

## 🔗 Resources

### Model & Code
- **🤗 Pre-trained Model**: [stargatek1/ssm-metarl-cartpole](https://huggingface.co/stargatek1/ssm-metarl-cartpole)
- **💻 GitHub Repository**: [sunghunkwag/SSM-MetaRL-Unified](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
- **📄 Model Card**: Detailed documentation in Model Hub
- **📊 Training Logs**: Available in repository

### Documentation
- **Architecture Details**: See `ARCHITECTURE.md` in repository
- **Model Generation Report**: See `MODEL_GENERATION_REPORT.md`
- **Training Script**: `train_and_save_model.py`
- **Verification Script**: `verify_model.py`

### Papers & References
- [MAML Paper (Finn et al., 2017)](https://arxiv.org/abs/1703.03400)
- [Meta-RL Survey](https://arxiv.org/abs/1910.03193)
- [State Space Models](https://arxiv.org/abs/2111.00396)

## 🛠️ Technical Details

### Model Architecture
```python
StateSpaceModel(
    state_dim=32,      # Hidden state dimension
    input_dim=4,       # CartPole observation space
    output_dim=4,      # For state prediction
    hidden_dim=64      # Network hidden dimension
)
```

### Training Configuration
- **Algorithm**: MetaMAML (Model-Agnostic Meta-Learning)
- **Epochs**: 50
- **Tasks per Epoch**: 5
- **Inner Learning Rate**: 0.01 (task adaptation)
- **Outer Learning Rate**: 0.001 (meta-update)
- **Adaptation Mode**: Hybrid (with experience replay)
- **Experience Buffer**: 3,191 transitions

### Usage Example
```python
from core.ssm import StateSpaceModel

# Load pre-trained model
model = StateSpaceModel(
    state_dim=32,
    input_dim=4,
    output_dim=4,
    hidden_dim=64
)
model.load("cartpole_hybrid_real_model.pth")
model.eval()

# Use for inference
obs = env.reset()
hidden_state = model.init_hidden(batch_size=1)
action_logits, hidden_state = model(obs_tensor, hidden_state)
```

## 🤝 Contributing

Contributions are welcome! This is an open-source research project.

- **Report Issues**: [GitHub Issues](https://github.com/sunghunkwag/SSM-MetaRL-Unified/issues)
- **Submit PRs**: Improvements and extensions welcome
- **Share Results**: Try on different environments and share your findings

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{ssm_metarl_unified,
  title={SSM-MetaRL-Unified: State Space Model + Meta-Reinforcement Learning},
  author={stargatek1},
  year={2025},
  url={https://github.com/sunghunkwag/SSM-MetaRL-Unified}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sunghunkwag/SSM-MetaRL-Unified/blob/master/LICENSE) file for details.

## 🙏 Acknowledgments

- **MAML Algorithm**: Chelsea Finn et al.
- **Gymnasium**: OpenAI and Farama Foundation
- **Gradio**: Hugging Face team
- **PyTorch**: Meta AI Research

## 🌟 Star History

If you find this project useful, please consider:
- ⭐ Starring the [GitHub repository](https://github.com/sunghunkwag/SSM-MetaRL-Unified)
- 👍 Liking this Space
- 🔗 Sharing with others interested in Meta-RL

---

**Made with ❤️ for the Meta-RL and Reinforcement Learning community**

**Ready to try? Click the "Load Pre-trained Model" button above and start testing!** 🚀

