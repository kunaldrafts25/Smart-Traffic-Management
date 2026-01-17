# 🚦 Smart Traffic Management System

AI-powered 4-lane intersection controller using computer vision and reinforcement learning for real-time traffic signal optimization.

**Built by Kunal Singh**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **YOLOv8 Detection** - Real-time vehicle counting on video feeds
- **LSTM Prediction** - Traffic flow forecasting using 9-feature time series
- **DQN Control** - Reinforcement learning for optimal signal timing
- **8-Phase Signals** - Protected right turns + straight/left movements
- **Live Metrics** - Throughput, wait time, CO₂ emissions, efficiency

## Architecture

```
Video Feed → YOLO Detection → LSTM Prediction → DQN Decision → Signal Control
                  ↓                  ↓                ↓
            Vehicle Count      Traffic Forecast    Phase Switch
```

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/Smart-Traffic-Management.git
cd Smart-Traffic-Management

# Install
pip install -r requirements.txt

# Run
streamlit run app.py --server.port 8501
```

## Usage

1. Upload 4 traffic videos (North, South, East, West lanes)
2. Click **Start Controller**
3. Watch real-time detection and signal control
4. Monitor metrics in the sidebar

## Models

All models were **custom-trained** for this project:

| Model | Purpose | File |
|-------|---------|------|
| YOLOv8n | Vehicle detection | `models/yolov8n.pt` |
| LSTM | Traffic prediction (9-feature, 15-timestep) | `models/lstm_best.h5` |
| Dueling DQN | RL signal control | `models/dqn_final.pt` |

## Signal Phases

| Phase | Green Lanes | Type |
|-------|-------------|------|
| 1 | North | Right turn |
| 2 | North + South | Straight + Left |
| 3 | South | Right turn |
| 4 | - | Transition |
| 5 | East | Right turn |
| 6 | East + West | Straight + Left |
| 7 | West | Right turn |
| 8 | - | Transition |

## Metrics

- **Throughput** - Vehicles processed through intersection
- **Wait Time** - Total vehicle-seconds at red lights
- **CO₂ Emissions** - Estimated emissions from idling vehicles
- **Efficiency** - `throughput / (throughput + wait_penalty)`

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo and set main file to `app.py`
4. Deploy

## Project Structure

```
Smart-Traffic-Management/
├── app.py                  # Main Streamlit app
├── main.py                 # CLI launcher
├── models/                 # Trained models
│   ├── yolov8n.pt
│   ├── lstm_best.h5
│   ├── dqn_final.pt
│   └── scalers.pkl
├── src/
│   └── core/
│       └── modern_vehicle_detector.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- TensorFlow 2.13+
- Streamlit 1.28+
- Ultralytics 8.0+

## License

MIT License
