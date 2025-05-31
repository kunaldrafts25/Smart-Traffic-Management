"""
MIT License

Copyright (c) 2024 kunalsingh2514@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
LSTM Model Training Script for TMS2
Specialized script for training LSTM traffic prediction models.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_manager import init_config
from src.utils.logger import setup_logging, get_logger
from src.models.lstm_model import EnhancedLSTMModel
from src.core.modern_vehicle_detector import ModernVehicleDetector
from src.utils.data_processor import DataProcessor
from src.training.training_utils import TrafficDataGenerator, ModelEvaluator


def collect_lstm_training_data(video_paths, max_frames_per_video=1000):
    """Collect and prepare training data for LSTM models."""
    logger = get_logger("LSTMTraining")
    logger.info(f"Collecting LSTM training data from {len(video_paths)} videos")
    
    detector = ModernVehicleDetector()
    data_processor = DataProcessor()
    
    all_features = []
    all_timestamps = []
    
    try:
        for video_path in video_paths:
            logger.info(f"Processing video: {video_path}")
            
            frame_count = 0
            video_features = []
            
            for processed_frame in data_processor.process_video_file(video_path):
                if frame_count >= max_frames_per_video:
                    break
                
                # Detect vehicles
                detection_result = detector.detect_vehicles(
                    processed_frame.frame, frame_count
                )
                
                # Extract features
                features = {
                    'timestamp': processed_frame.metadata['processing_timestamp'],
                    'vehicle_count': detection_result.vehicle_count,
                    'traffic_density': detection_result.vehicle_count / 1000,  # Normalized
                    'avg_confidence': np.mean(detection_result.confidence_scores) if detection_result.confidence_scores else 0.0,
                    'hour': datetime.fromtimestamp(processed_frame.metadata['processing_timestamp']).hour,
                    'day_of_week': datetime.fromtimestamp(processed_frame.metadata['processing_timestamp']).weekday()
                }
                
                video_features.append(features)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames from {video_path}")
            
            all_features.extend(video_features)
            logger.info(f"Completed {video_path}: {len(video_features)} frames")
    
    finally:
        detector.cleanup()
        data_processor.cleanup()
    
    logger.info(f"Total features collected: {len(all_features)}")
    return all_features


def prepare_lstm_sequences(features, sequence_length=15):
    """Prepare sequential data for LSTM training."""
    logger = get_logger("LSTMTraining")
    logger.info(f"Preparing LSTM sequences with length {sequence_length}")
    
    if len(features) < sequence_length:
        logger.warning(f"Not enough data for sequences. Need at least {sequence_length}, got {len(features)}")
        return np.array([]), np.array([])
    
    df = pd.DataFrame(features)
    
    # Normalize temporal features
    df['hour_norm'] = df['hour'] / 24.0
    df['day_norm'] = df['day_of_week'] / 7.0
    
    # Select features for training
    feature_columns = ['vehicle_count', 'traffic_density', 'avg_confidence', 'hour_norm', 'day_norm']
    feature_data = df[feature_columns].values
    
    X, y = [], []
    for i in range(len(feature_data) - sequence_length):
        X.append(feature_data[i:i + sequence_length])
        y.append(feature_data[i + sequence_length, 0])  # Predict vehicle count
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} sequences with shape {X.shape}")
    return X, y


def train_lstm_model(X, y, model_type='standard', epochs=50, batch_size=32):
    """Train LSTM model with the prepared data."""
    logger = get_logger("LSTMTraining")
    logger.info(f"Training {model_type} LSTM model")
    
    # Initialize model
    lstm_model = EnhancedLSTMModel(
        model_type=model_type,
        sequence_length=X.shape[1],
        learning_rate=0.001
    )
    
    # Train the model
    training_results = lstm_model.train_model(X, y, epochs=epochs, batch_size=batch_size)
    
    # Evaluate the model
    evaluator = ModelEvaluator()
    
    # Split data for evaluation
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    evaluation_results = evaluator.evaluate_lstm_model(lstm_model.model, X_test, y_test)
    
    logger.info(f"Training completed. Final metrics: {evaluation_results}")
    
    return lstm_model, training_results, evaluation_results


def save_lstm_model(model, training_results, evaluation_results, model_type):
    """Save the trained LSTM model and results."""
    logger = get_logger("LSTMTraining")
    
    # Create models directory
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = models_dir / f"lstm_{model_type}_{timestamp}.h5"
    model.model.save(str(model_path))
    
    # Save training results
    results = {
        'model_type': model_type,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'timestamp': timestamp,
        'model_path': str(model_path)
    }
    
    results_path = models_dir / f"lstm_{model_type}_{timestamp}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Results saved: {results_path}")
    
    return str(model_path), str(results_path)


def main():
    """Main LSTM training function."""
    parser = argparse.ArgumentParser(description="Train LSTM Traffic Prediction Model")
    
    parser.add_argument('--video-dir', default='data/kaggle/highway-traffic-videos',
                       help='Directory containing training videos')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='Maximum number of videos to process')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum frames per video')
    parser.add_argument('--model-type', default='standard',
                       choices=['standard', 'bidirectional', 'attention', 'transformer'],
                       help='LSTM model type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--sequence-length', type=int, default=15,
                       help='LSTM sequence length')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data if video data is insufficient')
    
    args = parser.parse_args()
    
    # Initialize configuration and logging
    config = init_config()
    setup_logging(config.get_section('logging'))
    logger = get_logger("LSTMTraining")
    
    try:
        logger.info("Starting LSTM model training")
        
        # Collect training data
        video_dir = Path(args.video_dir)
        if video_dir.exists():
            video_files = list(video_dir.glob('*.avi'))[:args.max_videos]
            video_paths = [str(f) for f in video_files]
            
            if video_paths:
                print(f"📹 Collecting data from {len(video_paths)} videos...")
                features = collect_lstm_training_data(video_paths, args.max_frames)
            else:
                print("⚠️ No video files found, using synthetic data")
                features = []
        else:
            print(f"⚠️ Video directory not found: {video_dir}, using synthetic data")
            features = []
        
        # Use synthetic data if needed
        if len(features) < args.sequence_length * 10 or args.use_synthetic:
            print("🔄 Generating synthetic training data...")
            data_generator = TrafficDataGenerator()
            
            # Generate synthetic features
            synthetic_features = []
            for day in range(7):  # One week of data
                daily_pattern = data_generator.generate_daily_traffic_pattern(24)
                for hour, traffic_density in enumerate(daily_pattern):
                    synthetic_features.append({
                        'timestamp': datetime.now().timestamp(),
                        'vehicle_count': int(traffic_density * 50),
                        'traffic_density': traffic_density,
                        'avg_confidence': 0.8 + np.random.normal(0, 0.1),
                        'hour': hour,
                        'day_of_week': day
                    })
            
            features.extend(synthetic_features)
            print(f"✅ Added {len(synthetic_features)} synthetic data points")
        
        # Prepare sequences
        print(f"🔄 Preparing LSTM sequences...")
        X, y = prepare_lstm_sequences(features, args.sequence_length)
        
        if len(X) == 0:
            print("❌ No training data available")
            return 1
        
        print(f"✅ Prepared {len(X)} training sequences")
        
        # Train model
        print(f"🧠 Training {args.model_type} LSTM model...")
        model, training_results, evaluation_results = train_lstm_model(
            X, y, args.model_type, args.epochs, args.batch_size
        )
        
        # Save model
        print("💾 Saving trained model...")
        model_path, results_path = save_lstm_model(
            model, training_results, evaluation_results, args.model_type
        )
        
        # Print results
        print("\n🎉 LSTM Training Completed!")
        print(f"Model saved: {model_path}")
        print(f"Results saved: {results_path}")
        
        if 'mse' in evaluation_results:
            print(f"\n📊 Final Performance:")
            print(f"  MSE: {evaluation_results['mse']:.4f}")
            print(f"  MAE: {evaluation_results['mae']:.4f}")
            print(f"  R² Score: {evaluation_results.get('r2_score', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
