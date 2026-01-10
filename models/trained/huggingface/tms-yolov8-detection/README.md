---
license: apache-2.0
tags:
  - traffic-management
  - reinforcement-learning
  - smart-city
  - deep-learning
  - pytorch
---

# TMS2 - YOLO Traffic Management Models

## YOLOv8 Vehicle Detection Models

Fine-tuned YOLOv8 models for traffic scene understanding.

### Detection Classes:
- Vehicles (cars, trucks, buses, motorcycles)
- Pedestrians
- Traffic signs
- Emergency vehicles

### Performance:
- Real-time detection at 30+ FPS
- Optimized for traffic camera footage

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
