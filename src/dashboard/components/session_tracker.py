"""Session tracking for traffic analysis with real-time analytics."""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np


class SessionTracker:
    """Enhanced traffic analysis session data tracker with real-time analytics."""

    def __init__(self):
        self.session_start_time = None
        self.session_end_time = None
        self.vehicle_detections = []
        self.signal_decisions = []
        self.processing_times = []
        self.confidence_scores = []
        self.session_active = False

        # Enhanced analytics data
        self.vehicle_counts_timeline = []
        self.speed_data = []
        self.vehicle_types = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        self.traffic_density_history = []
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'frame_processing_times': [],
            'detection_accuracy': []
        }
        self.lstm_predictions = []
        self.rl_decisions = []
        self.directional_data = {
            'North': {'vehicles': [], 'speeds': [], 'density': []},
            'South': {'vehicles': [], 'speeds': [], 'density': []},
            'East': {'vehicles': [], 'speeds': [], 'density': []},
            'West': {'vehicles': [], 'speeds': [], 'density': []}
        }

        # Environmental impact data
        self.environmental_data = {
            'carbon_emissions': [],
            'fuel_consumption': [],
            'air_quality_impact': [],
            'idle_time_estimates': [],
            'optimization_benefits': [],
            'green_score_history': []
        }

    def start_session(self):
        """Start a new tracking session."""
        self.session_start_time = datetime.now()
        self.session_active = True
        self.vehicle_detections = []
        self.signal_decisions = []
        self.processing_times = []
        self.confidence_scores = []

    def add_detection(self, detection_data):
        """Add enhanced vehicle detection data to the session."""
        if self.session_active:
            timestamp = datetime.now()
            detection_entry = {
                'timestamp': timestamp,
                'vehicle_count': detection_data.get('vehicle_count', 0),
                'confidence': detection_data.get('confidence_scores', []),
                'processing_time': detection_data.get('processing_time', 0),
                'avg_speed': detection_data.get('avg_speed', 0.0),
                'vehicle_types': detection_data.get('vehicle_types', {}),
                'traffic_density': detection_data.get('traffic_density', 0.0),
                'direction': detection_data.get('direction', 'main')
            }
            self.vehicle_detections.append(detection_entry)

            self.vehicle_counts_timeline.append({
                'timestamp': timestamp,
                'count': detection_data.get('vehicle_count', 0)
            })

            if detection_data.get('avg_speed', 0) > 0:
                self.speed_data.append({
                    'timestamp': timestamp,
                    'speed': detection_data.get('avg_speed', 0)
                })

            vehicle_types = detection_data.get('vehicle_types', {})
            for vtype, count in vehicle_types.items():
                if vtype in self.vehicle_types:
                    self.vehicle_types[vtype] += count

            self.traffic_density_history.append({
                'timestamp': timestamp,
                'density': detection_data.get('traffic_density', 0.0)
            })

            direction = detection_data.get('direction', 'main')
            if direction in self.directional_data:
                self.directional_data[direction]['vehicles'].append({
                    'timestamp': timestamp,
                    'count': detection_data.get('vehicle_count', 0)
                })
                self.directional_data[direction]['speeds'].append({
                    'timestamp': timestamp,
                    'speed': detection_data.get('avg_speed', 0)
                })
                self.directional_data[direction]['density'].append({
                    'timestamp': timestamp,
                    'density': detection_data.get('traffic_density', 0.0)
                })

    def add_signal_decision(self, decision_data):
        """Add signal decision to session."""
        if self.session_active:
            timestamp = datetime.now()
            self.signal_decisions.append({
                'timestamp': timestamp,
                'action': decision_data.get('action', 'maintain'),
                'confidence': decision_data.get('confidence', 0.8),
                'reasoning': decision_data.get('reasoning', '')
            })

    def end_session(self):
        """End the tracking session."""
        self.session_end_time = datetime.now()
        self.session_active = False

    def get_session_duration(self):
        """Get session duration in minutes."""
        if self.session_start_time and self.session_end_time:
            return (self.session_end_time - self.session_start_time).total_seconds() / 60
        return 0

    def add_performance_metric(self, metric_type: str, value: float):
        """Add performance metric data."""
        if self.session_active and metric_type in self.performance_metrics:
            timestamp = datetime.now()
            self.performance_metrics[metric_type].append({
                'timestamp': timestamp,
                'value': value
            })

    def add_lstm_prediction(self, prediction_data):
        """Add LSTM prediction data."""
        if self.session_active:
            timestamp = datetime.now()
            self.lstm_predictions.append({
                'timestamp': timestamp,
                'predicted_count': prediction_data.get('predicted_count', 0),
                'confidence': prediction_data.get('confidence', 0.0),
                'trend': prediction_data.get('trend', 'stable'),
                'model_source': prediction_data.get('model_source', 'simulation')
            })

    def add_rl_decision(self, decision_data):
        """Add RL decision data."""
        if self.session_active:
            timestamp = datetime.now()
            self.rl_decisions.append({
                'timestamp': timestamp,
                'action': decision_data.get('action', 'maintain'),
                'confidence': decision_data.get('confidence', 0.0),
                'q_values': decision_data.get('q_values', []),
                'reasoning': decision_data.get('reasoning', ''),
                'model_source': decision_data.get('model_source', 'simulation')
            })

    def get_rolling_average(self, data_type: str, window_seconds: int = 30):
        """Get rolling average for specified data type."""
        if not self.session_active:
            return []

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=window_seconds)

        if data_type == 'vehicle_count':
            recent_data = [d for d in self.vehicle_counts_timeline if d['timestamp'] >= cutoff_time]
            return [d['count'] for d in recent_data]
        elif data_type == 'speed':
            recent_data = [d for d in self.speed_data if d['timestamp'] >= cutoff_time]
            return [d['speed'] for d in recent_data]
        elif data_type == 'density':
            recent_data = [d for d in self.traffic_density_history if d['timestamp'] >= cutoff_time]
            return [d['density'] for d in recent_data]

        return []

    def add_environmental_data(self, environmental_metrics: dict):
        """Add environmental impact data to the session."""
        if self.session_active:
            timestamp = datetime.now()
            for metric_type, value in environmental_metrics.items():
                if metric_type == 'green_score':
                    self.environmental_data['green_score_history'].append({
                        'timestamp': timestamp,
                        'value': value
                    })
                elif metric_type in self.environmental_data:
                    self.environmental_data[metric_type].append({
                        'timestamp': timestamp,
                        'value': value
                    })

    def calculate_environmental_impact(self, vehicle_count: int, avg_speed: float,
                                       traffic_density: float, vehicle_types: dict) -> dict:
        """Calculate environmental impact metrics based on traffic data."""
        emission_factors = {
            'car': {'co2': 208, 'nox': 0.4, 'pm': 0.02},
            'truck': {'co2': 520, 'nox': 2.1, 'pm': 0.08},
            'bus': {'co2': 650, 'nox': 3.2, 'pm': 0.12},
            'motorcycle': {'co2': 104, 'nox': 0.3, 'pm': 0.01}
        }

        fuel_factors = {
            'car': 0.089,
            'truck': 0.223,
            'bus': 0.278,
            'motorcycle': 0.045
        }

        total_co2 = 0
        total_fuel = 0

        total_type_vehicles = sum(vehicle_types.values()) if vehicle_types else 0
        scaling_factor = vehicle_count / max(total_type_vehicles, 1) if total_type_vehicles > 0 else 1

        for vehicle_type, count in vehicle_types.items():
            if vehicle_type in emission_factors and count > 0:
                speed_factor = 1.0
                if avg_speed < 20:
                    speed_factor = 1.5
                elif avg_speed > 60:
                    speed_factor = 0.8

                co2_per_vehicle = emission_factors[vehicle_type]['co2'] * speed_factor
                fuel_per_vehicle = fuel_factors[vehicle_type] * speed_factor

                total_co2 += co2_per_vehicle * count * scaling_factor
                total_fuel += fuel_per_vehicle * count * scaling_factor

        idle_time_factor = min(1.0, traffic_density * 2)
        estimated_idle_time = idle_time_factor * 60

        aqi_impact = (total_co2 / 1000) * 0.1

        base_score = 100
        emission_penalty = min(50, (total_co2 / 100))
        density_penalty = min(30, (traffic_density * 30))
        green_score = max(0, base_score - emission_penalty - density_penalty)

        return {
            'carbon_emissions': total_co2,
            'fuel_consumption': total_fuel,
            'air_quality_impact': aqi_impact,
            'idle_time_estimates': estimated_idle_time,
            'green_score': green_score
        }

    def calculate_rl_optimization_benefits(self, baseline_metrics: dict, optimized_metrics: dict) -> dict:
        """Calculate environmental benefits from RL optimization."""
        benefits = {}

        for metric in ['carbon_emissions', 'fuel_consumption', 'air_quality_impact']:
            if metric in baseline_metrics and metric in optimized_metrics:
                baseline_value = baseline_metrics[metric]
                optimized_value = optimized_metrics[metric]

                if baseline_value > 0:
                    reduction_percentage = ((baseline_value - optimized_value) / baseline_value) * 100
                    benefits[f'{metric}_reduction'] = max(0, reduction_percentage)
                else:
                    benefits[f'{metric}_reduction'] = 0

        return benefits

    def get_traffic_statistics(self):
        """Get comprehensive traffic statistics."""
        if not self.vehicle_detections:
            return {}

        vehicle_counts = [d['vehicle_count'] for d in self.vehicle_detections]
        speeds = [d.get('avg_speed', 0) for d in self.vehicle_detections if d.get('avg_speed', 0) > 0]
        densities = [d.get('traffic_density', 0) for d in self.vehicle_detections if d.get('traffic_density', 0) > 0]

        return {
            'vehicle_count': {
                'mean': np.mean(vehicle_counts) if vehicle_counts else 0,
                'median': np.median(vehicle_counts) if vehicle_counts else 0,
                'std': np.std(vehicle_counts) if vehicle_counts else 0,
                'min': min(vehicle_counts) if vehicle_counts else 0,
                'max': max(vehicle_counts) if vehicle_counts else 0
            },
            'speed': {
                'mean': np.mean(speeds) if speeds else 0,
                'median': np.median(speeds) if speeds else 0,
                'std': np.std(speeds) if speeds else 0,
                'min': min(speeds) if speeds else 0,
                'max': max(speeds) if speeds else 0
            },
            'density': {
                'mean': np.mean(densities) if densities else 0,
                'median': np.median(densities) if densities else 0,
                'std': np.std(densities) if densities else 0,
                'min': min(densities) if densities else 0,
                'max': max(densities) if densities else 0
            },
            'vehicle_types': self.vehicle_types.copy()
        }
