
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime

st.set_page_config(page_title="4-Lane Traffic Controller", page_icon="🚦", layout="wide")



class Phase(Enum):
    NORTH_RIGHT = 0
    NS_STRAIGHT_LEFT = 1
    SOUTH_RIGHT = 2
    NS_TRANSITION = 3
    EAST_RIGHT = 4
    EW_STRAIGHT_LEFT = 5
    WEST_RIGHT = 6
    EW_TRANSITION = 7

PHASE_GREEN_LANES = {
    0: ['north'],
    1: ['north', 'south'],
    2: ['south'],
    3: [],
    4: ['east'],
    5: ['east', 'west'],
    6: ['west'],
    7: [],
}

MIN_THROUGH_TIME = 15
MIN_RIGHT_TIME = 10
MAX_THROUGH_TIME = 120
MAX_RIGHT_TIME = 60
TRANSITION_TIME = 3
YELLOW_TIME = 4
MIN_PEDESTRIAN_TIME = 10

PHASE_TIMING = {
    0: (MIN_RIGHT_TIME, MAX_RIGHT_TIME),
    1: (MIN_THROUGH_TIME, MAX_THROUGH_TIME),
    2: (MIN_RIGHT_TIME, MAX_RIGHT_TIME),
    3: (TRANSITION_TIME, TRANSITION_TIME),
    4: (MIN_RIGHT_TIME, MAX_RIGHT_TIME),
    5: (MIN_THROUGH_TIME, MAX_THROUGH_TIME),
    6: (MIN_RIGHT_TIME, MAX_RIGHT_TIME),
    7: (TRANSITION_TIME, TRANSITION_TIME),
}

EMISSION_FACTORS = {'car': 2.3, 'motorcycle': 1.2, 'bus': 8.5, 'truck': 6.8, 'bicycle': 0.0}
LEARNING_DURATION = 5
DETECTION_INTERVAL = 3



@st.cache_resource
def load_yolo():
    try:
        from src.core.modern_vehicle_detector import ModernVehicleDetector
        return ModernVehicleDetector()
    except Exception as e:
        print(f"YOLO error: {e}")
        return None

@st.cache_resource
def load_lstm():
    try:
        import tensorflow as tf
        import pickle
        
        model_path = Path("models/lstm_best.h5")
        scaler_path = Path("models/scalers.pkl")
        
        model = None
        scalers = None
        
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path), compile=False)
            print(f"LSTM loaded: {model_path}")
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            print(f"Scalers loaded: feature_scaler, target_scaler")
        
        return model, scalers
    except Exception as e:
        print(f"LSTM/Scaler error: {e}")
    return None, None

import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=18, action_dim=2, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

@st.cache_resource
def load_rl():
    try:
        path = Path("models/dqn_final.pt")
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = DuelingDQN(state_dim=18, action_dim=2, hidden_dim=256)
            model.load_state_dict(checkpoint['policy_net'])
            model.eval()
            print(f"✅ DQN loaded: {path}")
            return model
        else:
            print(f"❌ DQN not found: {path}")
    except Exception as e:
        print(f"❌ DQN error: {e}")
        import traceback
        traceback.print_exc()
    return None

def build_dqn_state(lanes: Dict, current_phase: Phase, phase_time: float) -> torch.Tensor:
    lane_order = ['north', 'south', 'east', 'west']
    state = []
    
    for ln in lane_order:
        if ln in lanes:
            lane = lanes[ln]
            state.append(lane.vehicle_count / 30.0)
            state.append(lane.predicted)
            state.append(min(1.0, lane.vehicle_count / 60.0))
            state.append(0.0)
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])
    
    state.append(current_phase.value / 7.0)
    state.append(min(1.0, phase_time / 30.0))
    
    return torch.FloatTensor(state).unsqueeze(0)

def rl_should_switch_phase(rl_model, current_phase: Phase, lanes: Dict, phase_time: float) -> tuple:
    green_lanes = PHASE_GREEN_LANES[current_phase.value]
    min_time, max_time = PHASE_TIMING.get(current_phase.value, (15, 120))
    
    if current_phase in [Phase.NS_TRANSITION, Phase.EW_TRANSITION]:
        if phase_time >= TRANSITION_TIME:
            return True, "Transition complete"
        return False, "In transition"
    
    if phase_time < min_time:
        return False, f"Min time: {min_time - phase_time:.1f}s"
    
    if phase_time >= max_time:
        return True, f"Max time ({max_time}s)"
    
    if rl_model is None:
        return False, "Continue (no RL)"
    
    try:
        state = build_dqn_state(lanes, current_phase, phase_time)
        with torch.no_grad():
            q_values = rl_model(state)
            action = q_values.argmax(dim=1).item()
            q_stay = q_values[0, 0].item()
            q_switch = q_values[0, 1].item()
        
        if action == 1:
            return True, f"DQN: switch (Q={q_switch:.2f} > {q_stay:.2f})"
        else:
            return False, f"DQN: stay (Q={q_stay:.2f} > {q_switch:.2f})"
    
    except Exception as e:
        print(f"DQN error: {e}")
        return False, f"DQN error: {str(e)[:20]}"



@dataclass
class LaneState:
    name: str
    vehicle_count: int = 0
    predicted: float = 0.0
    frame_count: int = 0
    
    waiting_time: float = 0.0
    total_emissions: float = 0.0
    throughput: int = 0
    last_update: float = field(default_factory=time.time)
    
    count_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def add_count(self, count: int, is_red: bool = False):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        if is_red and count > 0:
            self.waiting_time += count * dt
        
        if is_red and count > 0:
            avg_emission = 2.5
            self.total_emissions += count * avg_emission * (dt / 60.0)
        
        if not is_red and self.vehicle_count > 0:
            self.throughput += min(self.vehicle_count, 3)
        
        self.vehicle_count = count
        self.frame_count += 1
        self.count_history.append(count)
    
    def get_features(self) -> np.ndarray:
        count = self.vehicle_count
        density = min(1.0, count / 60.0)
        avg_speed = max(10, 55 - count * 0.5)
        
        now = datetime.now()
        hour = now.hour
        is_peak = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        counts = list(self.count_history)
        rolling_3 = np.mean(counts[-3:]) if len(counts) >= 3 else count
        rolling_6 = np.mean(counts[-6:]) if len(counts) >= 6 else count
        
        return np.array([
            count, density, avg_speed, hour, is_peak,
            hour_sin, hour_cos, rolling_3, rolling_6
        ], dtype=np.float32)
    
    def get_prediction(self, model, scalers=None) -> float:
        if model is None or len(self.count_history) < 15:
            if self.count_history:
                avg = np.mean(list(self.count_history)[-15:])
                return max(0.0, min(1.0, avg / 50.0))
            return 0.0
        
        try:
            sequence = []
            counts = list(self.count_history)
            
            now = datetime.now()
            hr = now.hour
            pk = 1 if hr in [7, 8, 9, 17, 18, 19] else 0
            h_sin = np.sin(2 * np.pi * hr / 24)
            h_cos = np.cos(2 * np.pi * hr / 24)
            
            start_idx = max(0, len(counts) - 15)
            for i in range(start_idx, len(counts)):
                c = counts[i]
                d = min(1.0, c / 60.0)
                spd = max(10, 55 - c * 0.5)
                r3 = np.mean(counts[max(0, i-2):i+1])
                r6 = np.mean(counts[max(0, i-5):i+1])
                sequence.append([c, d, spd, hr, pk, h_sin, h_cos, r3, r6])
            
            while len(sequence) < 15:
                sequence.insert(0, sequence[0] if sequence else [0]*9)
            
            features = np.array(sequence[-15:], dtype=np.float32)
            
            if scalers and 'feature_scaler' in scalers:
                features = scalers['feature_scaler'].transform(features)
            
            x = features.reshape(1, 15, 9)
            pred_scaled = model.predict(x, verbose=0)
            raw_pred = float(pred_scaled.flatten()[0])
            
            if scalers and 'target_scaler' in scalers:
                pred_vehicles = scalers['target_scaler'].inverse_transform([[raw_pred]])[0, 0]
                self.predicted = max(0.0, min(1.0, pred_vehicles / 50.0))
            else:
                self.predicted = max(0.0, min(1.0, (raw_pred + 0.5) / 1.5))
            
            print(f"[LSTM] {self.name}: raw={raw_pred:.4f} -> {self.predicted:.3f}")
            return self.predicted
            
        except Exception as e:
            print(f"LSTM error: {e}")
            if self.count_history:
                return max(0.0, min(1.0, np.mean(self.count_history) / 50.0))
            return 0.0

def draw_overlay(frame: np.ndarray, lane_name: str, is_green: bool, vehicle_count: int, 
                 phase_name: str, is_learning: bool = False, is_yellow: bool = False,
                 ped_active: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    
    if is_learning:
        color = (255, 165, 0)
        status = "LEARN"
    elif is_yellow:
        color = (255, 255, 0)
        status = "SLOW"
    elif is_green:
        color = (0, 255, 0)
        status = "GO"
    else:
        color = (0, 0, 255)
        status = "STOP"
    
    cv2.circle(frame, (w-40, 40), 25, color, -1)
    cv2.circle(frame, (w-40, 40), 25, (255, 255, 255), 2)
    
    cv2.putText(frame, lane_name.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if ped_active:
        cv2.putText(frame, "PED", (w-80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.putText(frame, status, (w-80, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    display_phase = "LEARNING..." if is_learning else phase_name[:15]
    cv2.putText(frame, display_phase, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if not is_green and not is_yellow and not is_learning:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 40), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "PAUSED", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    return frame

def main():
    st.markdown("# 🚦 8-Phase Traffic Controller")
    st.markdown("*RL-Controlled | Video Plays Only When Green*")
    
    yolo = load_yolo()
    lstm, scalers = load_lstm()
    rl_model = load_rl()
    
    if 'phase' not in st.session_state:
        st.session_state.phase = Phase.NORTH_RIGHT
        st.session_state.phase_start = time.time()
    if 'lanes' not in st.session_state:
        st.session_state.lanes = {ln: LaneState(ln) for ln in ['north', 'south', 'east', 'west']}
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'caps' not in st.session_state:
        st.session_state.caps = {}
    if 'last_frames' not in st.session_state:
        st.session_state.last_frames = {}
    if 'stats' not in st.session_state:
        st.session_state.stats = {'total_vehicles': 0, 'cycles': 0, 'rl_reason': ''}
    if 'yellow_active' not in st.session_state:
        st.session_state.yellow_active = False
        st.session_state.yellow_start = 0
    if 'pedestrian_requested' not in st.session_state:
        st.session_state.pedestrian_requested = False
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Signal Status")
        phase_ph = st.empty()
        st.divider()
        st.header("🚦 Lane Status")
        lane_ph = st.empty()
        st.divider()
        st.header("🧠 LSTM Predictions")
        lstm_ph = st.empty()
        st.divider()
        st.header("📈 Stats")
        stats_ph = st.empty()
    
    # Video upload
    st.subheader("Upload Lane Videos")
    cols = st.columns(4)
    files = {}
    for i, ln in enumerate(['north', 'south', 'east', 'west']):
        with cols[i]:
            f = st.file_uploader(ln.upper(), type=["mp4", "avi", "mov"], key=ln)
            if f:
                files[ln] = f
    
    # Controls
    c1, c2 = st.columns(2)
    with c1:
        start = st.button("▶️ Start", type="primary", disabled=len(files) < 1)
    with c2:
        stop = st.button("⏹️ Stop")
    
    if stop:
        st.session_state.running = False
        st.rerun()
    
    if start and files:
        st.session_state.running = True
        st.session_state.phase = Phase.NORTH_RIGHT
        st.session_state.phase_start = time.time()
        for ln, f in files.items():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(f.read())
                st.session_state.caps[ln] = cv2.VideoCapture(tmp.name)
        st.rerun()
    
    # Main loop
    if st.session_state.running:
        st.divider()
        
        # Video placeholders
        col1, col2 = st.columns(2)
        with col1:
            ph_n = st.empty()
            ph_e = st.empty()
        with col2:
            ph_s = st.empty()
            ph_w = st.empty()
        placeholders = {'north': ph_n, 'south': ph_s, 'east': ph_e, 'west': ph_w}
        
        info_ph = st.empty()
        
        frame_count = 0
        start_time = time.time()
        target_fps = 24
        frame_time = 1.0 / target_fps
        LEARNING_DURATION = 5  # seconds - all videos play during learning
        
        while st.session_state.running:
            loop_start = time.time()
            frame_count += 1
            elapsed_total = time.time() - start_time
            is_learning = elapsed_total < LEARNING_DURATION
            current_phase = st.session_state.phase
            green_lanes = PHASE_GREEN_LANES[current_phase.value]
            phase_elapsed = time.time() - st.session_state.phase_start
            is_yellow = st.session_state.yellow_active
            yellow_elapsed = time.time() - st.session_state.yellow_start if is_yellow else 0
            
            # Pedestrian consideration: ensure minimum crossing time
            ped_active = st.session_state.pedestrian_requested
            phase_min, phase_max = PHASE_TIMING.get(current_phase.value, (15, 120))
            min_time = max(phase_min, MIN_PEDESTRIAN_TIME if ped_active else phase_min)
            
            
            if is_learning:
                rl_reason = f"Learning: {LEARNING_DURATION - elapsed_total:.1f}s"
                should_switch = False
                green_lanes = ['north', 'south', 'east', 'west']  # All lanes active
                is_yellow = False
            elif is_yellow:
                if yellow_elapsed >= YELLOW_TIME:
                    # Yellow complete, now switch
                    st.session_state.yellow_active = False
                    next_idx = (current_phase.value + 1) % 8
                    st.session_state.phase = Phase(next_idx)
                    st.session_state.phase_start = time.time()
                    st.session_state.pedestrian_requested = False
                    current_phase = st.session_state.phase
                    green_lanes = PHASE_GREEN_LANES[current_phase.value]
                    print(f"[YELLOW->GREEN] Phase -> {current_phase.name}")
                    if current_phase == Phase.NORTH_RIGHT:
                        st.session_state.stats['cycles'] += 1
                    is_yellow = False
                else:
                    rl_reason = f"Yellow: {YELLOW_TIME - yellow_elapsed:.1f}s"
            else:
                # RL decides if we should switch phase
                should_switch, rl_reason = rl_should_switch_phase(
                    rl_model, current_phase, st.session_state.lanes, phase_elapsed
                )
                
                if should_switch and not is_learning:
                    # Enter yellow phase instead of immediate switch
                    st.session_state.yellow_active = True
                    st.session_state.yellow_start = time.time()
                    is_yellow = True
                    rl_reason = f"[YELLOW] {rl_reason}"
                    print(f"[RL] -> YELLOW | Reason: {rl_reason}")
            
            st.session_state.stats['rl_reason'] = rl_reason
            
            # Process each lane
            for ln, cap in st.session_state.caps.items():
                lane = st.session_state.lanes[ln]
                is_green = ln in green_lanes
                if not is_green and not is_learning:
                    lane.add_count(lane.vehicle_count, is_red=True)
                
                # Read frame if GREEN or LEARNING (video plays)
                if is_green or is_learning:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # YOLO detection
                        if yolo:
                            try:
                                result = yolo.detect_vehicles(frame_rgb, frame_id=frame_count)
                                frame_rgb = yolo.draw_detections(frame_rgb, result)
                                lane.add_count(result.vehicle_count, is_red=False)
                                st.session_state.stats['total_vehicles'] += result.vehicle_count
                            except:
                                pass
                        
                        lane.get_prediction(lstm, scalers)
                        
                        st.session_state.last_frames[ln] = frame_rgb
                
                # Get frame to display (current or last)
                if ln in st.session_state.last_frames:
                    display_frame = st.session_state.last_frames[ln].copy()
                else:
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Draw overlay
                phase_name = current_phase.name.replace('_', ' ')
                lane_is_yellow = is_yellow and ln in green_lanes
                display_frame = draw_overlay(display_frame, ln, is_green, lane.vehicle_count, 
                                             phase_name, is_learning, lane_is_yellow, ped_active)
                
                # Display
                if ln in placeholders:
                    placeholders[ln].image(display_frame, caption=ln.upper())
            
            # Update sidebar
            with phase_ph.container():
                st.write(f"**Phase:** {current_phase.name.replace('_', ' ')}")
                st.write(f"**Time in phase:** {phase_elapsed:.1f}s")
                st.write(f"**Green Lanes:** {', '.join(green_lanes) if green_lanes else 'None (transition)'}")
                st.write(f"**RL Decision:** {st.session_state.stats.get('rl_reason', '')}")
            
            with lane_ph.container():
                for ln, lane in st.session_state.lanes.items():
                    icon = "🟢" if ln in green_lanes else "🔴"
                    st.write(f"{icon} **{ln.upper()}**: {lane.vehicle_count}")
            
            with lstm_ph.container():
                for ln, lane in st.session_state.lanes.items():
                    st.write(f"**{ln.upper()}**: {lane.predicted:.3f}")
            
            with stats_ph.container():
                elapsed = time.time() - start_time
                fps = frame_count / max(0.1, elapsed)
                
                total_wait = sum(l.waiting_time for l in st.session_state.lanes.values())
                total_emissions = sum(l.total_emissions for l in st.session_state.lanes.values())
                total_throughput = sum(l.throughput for l in st.session_state.lanes.values())
                
                efficiency = 100 * total_throughput / max(1, total_throughput + total_wait/60)
                
                st.write(f"**FPS:** {fps:.1f}")
                st.write(f"**Cycles:** {st.session_state.stats['cycles']}")
                st.divider()
                st.write(f"**🚗 Throughput:** {total_throughput}")
                st.write(f"**⏱️ Wait Time:** {total_wait:.0f}s")
                st.write(f"**💨 CO₂ Emissions:** {total_emissions:.1f}g")
                st.write(f"**📊 Efficiency:** {efficiency:.1f}%")
            
            # Info bar
            info_ph.write(f"**Frame:** {frame_count} | **Phase:** {current_phase.name} | **Phase Time:** {phase_elapsed:.1f}s")
            
            # Maintain 24 FPS
            elapsed_loop = time.time() - loop_start
            if elapsed_loop < frame_time:
                time.sleep(frame_time - elapsed_loop)

if __name__ == "__main__":
    main()
