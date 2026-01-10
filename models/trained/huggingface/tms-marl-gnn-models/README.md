---
license: apache-2.0
tags:
  - traffic-management
  - reinforcement-learning
  - smart-city
  - deep-learning
  - pytorch
---

# TMS2 - MARL Traffic Management Models

## Multi-Agent Reinforcement Learning Models

Coordinated multi-intersection traffic control using MARL algorithms.

### Algorithms:
- **IQL**: Independent Q-Learning
- **VDN**: Value Decomposition Networks
- **QMIX**: Monotonic value factorization

### Features:
- Multi-intersection coordination
- Green wave optimization
- Network-wide throughput maximization

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
