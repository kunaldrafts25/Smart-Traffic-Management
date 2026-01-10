---
license: apache-2.0
tags:
  - traffic-management
  - reinforcement-learning
  - smart-city
  - deep-learning
  - pytorch
---

# TMS2 - RL Traffic Management Models

## RL Traffic Signal Controller

Deep Q-Network (DQN) based agents for adaptive traffic signal control.

### Variants:
- **v2**: Baseline stable model optimized for throughput
- **v3**: Eco-friendly model with emissions optimization (14% CO2 reduction)
- **v4**: Challenging scenarios with incidents and demand spikes
- **v5/final**: Curriculum learning for generalization

### Architecture:
- Dueling Double DQN with soft target updates
- State space: 10-12 dimensions (queues, phase, time, scenarios)
- Action space: 2 (keep phase / change phase)

## Model Description

These models are part of the **Traffic Management System 2 (TMS2)** project, 
an intelligent traffic control system using deep learning and reinforcement learning.

## Training Details

- **Framework**: PyTorch
- **Training Platform**: Google Colab (T4 GPU)
- **Training Date**: December 2025

## Usage

```python
import torch

# Load model
model = torch.load('model.pt')
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

## License

Apache 2.0
