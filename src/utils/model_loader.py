"""
MIT License
Copyright (c) 2024 kunalsingh2514@gmail.com

Model Loader for TMS2 - Loads trained models from HuggingFace or local storage.

This module provides unified model loading for:
- YOLOv8 fine-tuned detection models
- LSTM traffic prediction models
- RL signal controller models (DQN, DoubleDQN, DuelingDQN)
- MARL models (VDN, QMIX, Independent)
- GNN encoders (GCN, GAT)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    path: str
    model_type: str
    loaded: bool
    metadata: Dict[str, Any]


class TMS2ModelLoader:
    """
    Unified model loader for TMS2 trained models.
    
    Supports loading from:
    - Local HuggingFace cache (models/trained/huggingface/)
    - Direct HuggingFace Hub download
    - Legacy model paths
    """
    
    # Model repositories on HuggingFace
    HF_REPOS = {
        'yolo': 'Kunalsinghh/tms-yolov8-detection',
        'lstm': 'Kunalsinghh/tms-lstm-predictor',
        'rl': 'Kunalsinghh/tms-rl-traffic-controller',
        'marl': 'Kunalsinghh/tms-marl-gnn-models',
        'gnn': 'Kunalsinghh/tms2-gnn-encoders',
    }
    
    # Default model files for each type
    DEFAULT_MODELS = {
        'yolo': 'best.pt',
        'lstm': 'lstm_traffic_predictor.h5',
        'rl_dqn': 'dqn_best.pt',
        'rl_double_dqn': 'double_dqn_best.pt',
        'rl_dueling_dqn': 'dueling_dqn_best.pt',
        'rl_adaptive': 'adaptive_dueling_final.pt',
        'rl_eco': 'rl_signal_controller_v3_eco.pt',
        'marl_vdn': 'vdn_best.pt',
        'marl_qmix': 'qmix_best.pt',
        'marl_independent': 'independent_best.pt',
        'gnn_gcn': 'gcn_best.pt',
        'gnn_gat': 'gat_best.pt',
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            base_path: Base path for models. Defaults to models/trained/huggingface/
        """
        if base_path is None:
            # Find project root
            current = Path(__file__).parent
            while current.name != 'Smart-Traffic-Management' and current.parent != current:
                current = current.parent
            base_path = current / 'models' / 'trained' / 'huggingface'
        
        self.base_path = Path(base_path)
        self.loaded_models: Dict[str, Any] = {}
        self._check_models_available()
    
    def _check_models_available(self) -> Dict[str, bool]:
        """Check which models are available locally."""
        availability = {}
        
        for model_type, repo in self.HF_REPOS.items():
            local_dir = self.base_path / repo.split('/')[-1]
            availability[model_type] = local_dir.exists()
            
            if availability[model_type]:
                logger.info(f"[OK] {model_type.upper()} models available at {local_dir}")
            else:
                logger.warning(f"[MISSING] {model_type.upper()} models not found at {local_dir}")
        
        return availability
    
    def download_models(self, model_types: Optional[list] = None, force: bool = False):
        """
        Download models from HuggingFace Hub.
        
        Args:
            model_types: List of model types to download. None = all.
            force: Force re-download even if exists.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return
        
        if model_types is None:
            model_types = list(self.HF_REPOS.keys())
        
        for model_type in model_types:
            if model_type not in self.HF_REPOS:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            repo_id = self.HF_REPOS[model_type]
            local_dir = self.base_path / repo_id.split('/')[-1]
            
            if local_dir.exists() and not force:
                logger.info(f"Skipping {model_type}: already exists at {local_dir}")
                continue
            
            logger.info(f"Downloading {model_type} from {repo_id}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"[OK] Downloaded {model_type} to {local_dir}")
            except Exception as e:
                logger.error(f"Failed to download {model_type}: {e}")
    
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """
        Get the path to a specific model file.
        
        Args:
            model_key: Model key (e.g., 'rl_double_dqn', 'gnn_gat', 'yolo')
            
        Returns:
            Path to model file or None if not found.
        """
        if model_key not in self.DEFAULT_MODELS:
            logger.warning(f"Unknown model key: {model_key}")
            return None
        
        filename = self.DEFAULT_MODELS[model_key]
        
        # Determine the repo directory
        if model_key.startswith('rl_'):
            repo_dir = 'tms-rl-traffic-controller'
        elif model_key.startswith('marl_'):
            repo_dir = 'tms-marl-gnn-models'
        elif model_key.startswith('gnn_'):
            repo_dir = 'tms2-gnn-encoders'
        elif model_key == 'yolo':
            repo_dir = 'tms-yolov8-detection'
        elif model_key == 'lstm':
            repo_dir = 'tms-lstm-predictor'
        else:
            logger.warning(f"Cannot determine repo for: {model_key}")
            return None
        
        model_path = self.base_path / repo_dir / filename
        
        if model_path.exists():
            return model_path
        
        # Try searching recursively
        search_dir = self.base_path / repo_dir
        if search_dir.exists():
            matches = list(search_dir.rglob(filename))
            if matches:
                return matches[0]
        
        logger.warning(f"Model not found: {model_path}")
        return None
    
    def load_rl_model(self, variant: str = 'double_dqn', device: str = 'cpu') -> Optional[Any]:
        """
        Load an RL model (DQN, DoubleDQN, DuelingDQN, etc.)
        
        Args:
            variant: Model variant ('dqn', 'double_dqn', 'dueling_dqn', 'adaptive', 'eco')
            device: Device to load on ('cpu', 'cuda')
            
        Returns:
            Loaded PyTorch model state dict or None.
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not installed. Run: pip install torch")
            return None
        
        model_key = f'rl_{variant}'
        model_path = self.get_model_path(model_key)
        
        if model_path is None:
            logger.error(f"RL model '{variant}' not found")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(f"[OK] Loaded RL model: {model_path}")
            
            self.loaded_models[model_key] = {
                'path': str(model_path),
                'checkpoint': checkpoint,
                'device': device
            }
            
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return None
    
    def load_lstm_model(self) -> Optional[Any]:
        """
        Load the LSTM traffic predictor model.
        
        Returns:
            Loaded Keras model or None.
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not installed. Run: pip install tensorflow")
            return None
        
        model_path = self.get_model_path('lstm')
        
        if model_path is None:
            logger.error("LSTM model not found")
            return None
        
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info(f"[OK] Loaded LSTM model: {model_path}")
            
            self.loaded_models['lstm'] = {
                'path': str(model_path),
                'model': model
            }
            
            return model
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return None
    
    def load_yolo_model(self) -> Optional[Any]:
        """
        Load the fine-tuned YOLOv8 model.
        
        Returns:
            Loaded YOLO model or None.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics not installed. Run: pip install ultralytics")
            return None
        
        model_path = self.get_model_path('yolo')
        
        if model_path is None:
            logger.error("YOLO model not found")
            return None
        
        try:
            model = YOLO(str(model_path))
            logger.info(f"[OK] Loaded YOLO model: {model_path}")
            
            self.loaded_models['yolo'] = {
                'path': str(model_path),
                'model': model
            }
            
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    
    def load_gnn_encoder(self, variant: str = 'gat', device: str = 'cpu') -> Optional[Any]:
        """
        Load a GNN encoder model (GCN or GAT).
        
        Args:
            variant: 'gcn' or 'gat'
            device: Device to load on
            
        Returns:
            Loaded PyTorch model state dict or None.
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not installed")
            return None
        
        model_key = f'gnn_{variant}'
        model_path = self.get_model_path(model_key)
        
        if model_path is None:
            logger.error(f"GNN model '{variant}' not found")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(f"[OK] Loaded GNN encoder: {model_path}")
            
            self.loaded_models[model_key] = {
                'path': str(model_path),
                'checkpoint': checkpoint,
                'device': device
            }
            
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load GNN model: {e}")
            return None
    
    def load_marl_model(self, variant: str = 'vdn', device: str = 'cpu') -> Optional[Any]:
        """
        Load a MARL model (VDN, QMIX, Independent).
        
        Args:
            variant: 'vdn', 'qmix', or 'independent'
            device: Device to load on
            
        Returns:
            Loaded PyTorch model state dict or None.
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not installed")
            return None
        
        model_key = f'marl_{variant}'
        model_path = self.get_model_path(model_key)
        
        if model_path is None:
            logger.error(f"MARL model '{variant}' not found")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(f"[OK] Loaded MARL model: {model_path}")
            
            self.loaded_models[model_key] = {
                'path': str(model_path),
                'checkpoint': checkpoint,
                'device': device
            }
            
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load MARL model: {e}")
            return None
    
    def get_all_available_models(self) -> Dict[str, list]:
        """
        Get list of all available model files organized by type.
        
        Returns:
            Dictionary mapping model type to list of available files.
        """
        available = {}
        
        for model_type, repo in self.HF_REPOS.items():
            repo_dir = self.base_path / repo.split('/')[-1]
            available[model_type] = []
            
            if repo_dir.exists():
                for pt_file in repo_dir.rglob('*.pt'):
                    available[model_type].append(str(pt_file.relative_to(repo_dir)))
                for h5_file in repo_dir.rglob('*.h5'):
                    available[model_type].append(str(h5_file.relative_to(repo_dir)))
        
        return available
    
    def print_status(self):
        """Print status of all models."""
        print("\n" + "="*60)
        print("TMS2 Model Loader Status")
        print("="*60)
        
        available = self.get_all_available_models()
        
        for model_type, files in available.items():
            status = "[OK]" if files else "[X]"
            print(f"\n{status} {model_type.upper()} ({len(files)} files)")
            for f in files[:5]:  # Show first 5
                print(f"   • {f}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
        
        print("\n" + "="*60)


# Global instance for convenience
_model_loader: Optional[TMS2ModelLoader] = None


def get_model_loader() -> TMS2ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = TMS2ModelLoader()
    return _model_loader


def load_trained_rl_model(variant: str = 'double_dqn', device: str = 'cpu'):
    """Convenience function to load RL model."""
    return get_model_loader().load_rl_model(variant, device)


def load_trained_lstm_model():
    """Convenience function to load LSTM model."""
    return get_model_loader().load_lstm_model()


def load_trained_yolo_model():
    """Convenience function to load YOLO model."""
    return get_model_loader().load_yolo_model()


# CLI for testing
if __name__ == '__main__':
    import sys
    
    loader = TMS2ModelLoader()
    loader.print_status()
    
    if '--download' in sys.argv:
        print("\nDownloading models from HuggingFace...")
        loader.download_models()
        loader.print_status()
    
    if '--test-rl' in sys.argv:
        print("\nTesting RL model loading...")
        model = loader.load_rl_model('double_dqn')
        if model:
            print(f"Model keys: {list(model.keys()) if isinstance(model, dict) else 'loaded'}")
    
    if '--test-lstm' in sys.argv:
        print("\nTesting LSTM model loading...")
        model = loader.load_lstm_model()
        if model:
            print(f"Model summary: {model.summary()}")
