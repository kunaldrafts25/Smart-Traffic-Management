---
license: apache-2.0
tags:
  - traffic-management
  - reinforcement-learning
  - smart-city
  - deep-learning
  - pytorch
---

# TMS2 - GNN Traffic Management Models

## Graph Neural Network Models

GNN-based traffic state encoders for learning road network representations.

### Architectures:
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks

### Use Cases:
- Traffic state encoding for RL agents
- Network-level congestion prediction
- Spatial dependency modeling

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
