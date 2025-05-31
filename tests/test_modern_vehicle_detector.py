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
Unit tests for ModernVehicleDetector

Tests modern vehicle detection functionality with YOLOv8/YOLOv11,
camera management, and error handling.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.core.modern_vehicle_detector import (
    ModernVehicleDetector, CameraManager, CameraInfo, DetectionResult, BoundingBox
)
from src.utils.error_handler import VehicleDetectionError, ModelLoadingError, CameraConnectionError


class TestCameraManager:
    """Test cases for CameraManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch('cv2.VideoCapture')
    def test_camera_enumeration(self, mock_videocapture):
        """Test camera enumeration functionality."""
        # Mock successful camera
        mock_cap_success = Mock()
        mock_cap_success.isOpened.return_value = True
        mock_cap_success.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        mock_cap_success.read.return_value = (True, self.test_image)

        # Mock failed camera
        mock_cap_fail = Mock()
        mock_cap_fail.isOpened.return_value = False

        # Setup mock to return success for camera 0, fail for others
        def mock_videocapture_side_effect(index):
            if index == 0:
                return mock_cap_success
            else:
                return mock_cap_fail

        mock_videocapture.side_effect = mock_videocapture_side_effect

        # Test camera enumeration
        camera_manager = CameraManager()

        # Should find at least one camera
        available_cameras = camera_manager.get_available_cameras()
        assert len(available_cameras) >= 1

        # Check camera 0 properties
        camera_0 = available_cameras[0]
        assert camera_0.index == 0
        assert camera_0.is_available is True
        assert camera_0.resolution == (1920, 1080)
        assert camera_0.fps == 30.0

    @patch('cv2.VideoCapture')
    def test_no_cameras_available(self, mock_videocapture):
        """Test behavior when no cameras are available."""
        # Mock all cameras as unavailable
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        camera_manager = CameraManager()

        available_cameras = camera_manager.get_available_cameras()
        assert len(available_cameras) == 0

        best_camera = camera_manager.get_best_camera()
        assert best_camera is None

    @patch('cv2.VideoCapture')
    def test_camera_testing(self, mock_videocapture):
        """Test individual camera testing functionality."""
        # Mock successful camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.test_image)
        mock_videocapture.return_value = mock_cap

        camera_manager = CameraManager()

        is_working, message = camera_manager.test_camera(0)
        assert is_working is True
        assert "working" in message.lower()

    @patch('cv2.VideoCapture')
    def test_camera_testing_failure(self, mock_videocapture):
        """Test camera testing with failed camera."""
        # Mock failed camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        camera_manager = CameraManager()

        is_working, message = camera_manager.test_camera(0)
        assert is_working is False
        assert "cannot open" in message.lower()

    def test_troubleshooting_info(self):
        """Test camera troubleshooting information generation."""
        camera_manager = CameraManager()

        troubleshooting = camera_manager.get_camera_troubleshooting_info()

        assert isinstance(troubleshooting, str)
        assert len(troubleshooting) > 100  # Should be substantial
        assert "Camera" in troubleshooting
        assert "permissions" in troubleshooting.lower()


class TestModernVehicleDetector:
    """Test cases for ModernVehicleDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch('src.core.modern_vehicle_detector.get_config')
    def test_ultralytics_not_available(self, mock_get_config):
        """Test behavior when ultralytics package is not available."""
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'models.yolo.model_name': 'yolov8n.pt',
            'models.yolo.confidence_threshold': 0.5,
            'models.yolo.device': 'auto'
        }.get(key, default)

        with patch('src.core.modern_vehicle_detector.ModernVehicleDetector._check_ultralytics_available', return_value=False):
            with pytest.raises(ModelLoadingError):
                ModernVehicleDetector()

    def create_mock_detector(self):
        """Create a mock ModernVehicleDetector for testing."""
        # Mock configuration values
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            'models.yolo.model_name': 'yolov8n.pt',
            'models.yolo.confidence_threshold': 0.5,
            'models.yolo.device': 'cpu',
            'models.yolo.vehicle_classes': ['car', 'motorcycle', 'bus', 'truck', 'bicycle'],
            'models.yolo.class_mapping': {
                'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7, 'bicycle': 1
            },
            'camera.fallback_cameras': [0, 1, 2],
            'camera.test_duration': 3.0,
            'traffic_cameras.sources': []
        }.get(key, default)

        with patch('src.core.modern_vehicle_detector.get_config', return_value=mock_config), \
             patch('src.core.modern_vehicle_detector.CameraManager'), \
             patch('src.core.modern_vehicle_detector.ModernVehicleDetector._check_ultralytics_available', return_value=True), \
             patch('src.core.modern_vehicle_detector.ModernVehicleDetector._load_model'):

            detector = ModernVehicleDetector()

            # Mock the model
            detector.model = Mock()
            detector.model.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

            return detector

    def test_detector_initialization(self):
        """Test ModernVehicleDetector initialization."""
        detector = self.create_mock_detector()

        assert detector.model_name == 'yolov8n.pt'
        assert detector.confidence_threshold == 0.5
        assert detector.vehicle_classes == ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
        assert detector.vehicle_class_ids == [2, 3, 5, 7, 1]

    def test_detect_vehicles_valid_input(self):
        """Test vehicle detection with valid input."""
        detector = self.create_mock_detector()

        # Mock YOLO results
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 150]  # x1, y1, x2, y2
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.85
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 2  # car class

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        detector.model.return_value = [mock_result]

        result = detector.detect_vehicles(self.test_image, frame_id=1)

        assert isinstance(result, DetectionResult)
        assert result.frame_id == 1
        assert result.vehicle_count == 1
        assert len(result.detections) == 1
        assert len(result.confidence_scores) == 1
        assert result.confidence_scores[0] == 0.85
        assert result.model_name == 'yolov8n.pt'

    def test_detect_vehicles_no_detections(self):
        """Test vehicle detection with no vehicles detected."""
        detector = self.create_mock_detector()

        # Mock YOLO results with no detections
        mock_result = Mock()
        mock_result.boxes = None

        detector.model.return_value = [mock_result]

        result = detector.detect_vehicles(self.test_image, frame_id=1)

        assert isinstance(result, DetectionResult)
        assert result.vehicle_count == 0
        assert len(result.detections) == 0
        assert len(result.confidence_scores) == 0

    def test_detect_vehicles_invalid_input(self):
        """Test vehicle detection with invalid input."""
        detector = self.create_mock_detector()

        # Test with None input
        with pytest.raises(VehicleDetectionError):
            detector.detect_vehicles(None)

        # Test with empty array
        with pytest.raises(VehicleDetectionError):
            detector.detect_vehicles(np.array([]))

    @patch('cv2.VideoCapture')
    def test_open_video_source_camera(self, mock_videocapture):
        """Test opening camera video source."""
        detector = self.create_mock_detector()

        # Mock successful camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        # Mock camera manager
        mock_camera_info = CameraInfo(
            index=0, name="Camera 0", resolution=(640, 480),
            fps=30.0, is_available=True
        )
        detector.camera_manager.test_camera.return_value = (True, "Camera working")
        detector.camera_manager.get_available_cameras.return_value = [mock_camera_info]
        detector.camera_manager.get_best_camera.return_value = mock_camera_info

        cap = detector.open_video_source(0)

        assert cap is not None
        assert cap.isOpened()

    @patch('cv2.VideoCapture')
    def test_open_video_source_file(self, mock_videocapture):
        """Test opening video file source."""
        detector = self.create_mock_detector()

        # Mock successful file opening
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            cap = detector.open_video_source(temp_path)
            assert cap is not None
            assert cap.isOpened()
        finally:
            os.unlink(temp_path)

    @patch('cv2.VideoCapture')
    def test_open_video_source_camera_fallback(self, mock_videocapture):
        """Test camera fallback when primary camera fails."""
        detector = self.create_mock_detector()

        # Mock successful camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap

        # Mock camera manager with fallback
        mock_camera_info = CameraInfo(
            index=1, name="Camera 1", resolution=(640, 480),
            fps=30.0, is_available=True
        )
        detector.camera_manager.test_camera.side_effect = [
            (False, "Camera 0 not working"),  # Primary camera fails
            (True, "Camera 1 working")        # Fallback camera works
        ]
        detector.camera_manager.get_available_cameras.return_value = [mock_camera_info]
        detector.camera_manager.get_best_camera.return_value = mock_camera_info

        cap = detector.open_video_source(0)  # Request camera 0

        assert cap is not None
        assert cap.isOpened()

    def test_draw_detections(self):
        """Test drawing detections on frame."""
        detector = self.create_mock_detector()

        # Create test detection result
        bbox = BoundingBox(x=100, y=100, width=50, height=80,
                          confidence=0.85, class_id=2, class_name='car')

        detection_result = DetectionResult(
            vehicle_count=1,
            detections=[{
                'bbox': bbox,
                'confidence': 0.85,
                'class_id': 2,
                'class_name': 'car',
                'center_x': 125,
                'center_y': 140
            }],
            confidence_scores=[0.85],
            processing_time=0.1,
            frame_id=1,
            timestamp=1234567890.0,
            model_name='yolov8n.pt'
        )

        # Draw detections
        output_frame = detector.draw_detections(self.test_image.copy(), detection_result)

        # Verify output
        assert output_frame.shape == self.test_image.shape
        assert not np.array_equal(output_frame, self.test_image)  # Should be modified

    def test_detect_vehicles_batch(self):
        """Test batch vehicle detection."""
        detector = self.create_mock_detector()

        # Mock successful detection
        with patch.object(detector, 'detect_vehicles') as mock_detect:
            mock_result = DetectionResult(
                vehicle_count=2,
                detections=[],
                confidence_scores=[0.8, 0.9],
                processing_time=0.1,
                frame_id=0,
                timestamp=1234567890.0,
                model_name='yolov8n.pt'
            )
            mock_detect.return_value = mock_result

            frames = [self.test_image, self.test_image, self.test_image]
            results = detector.detect_vehicles_batch(frames)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)
        assert mock_detect.call_count == 3

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        detector = self.create_mock_detector()

        # Initially should have zero stats
        stats = detector.get_performance_stats()
        assert stats['frames_processed'] == 0
        assert stats['average_fps'] == 0.0
        assert stats['model_name'] == 'yolov8n.pt'

        # Mock some processing
        detector.frame_count = 10
        detector.total_processing_time = 1.0

        stats = detector.get_performance_stats()
        assert stats['frames_processed'] == 10
        assert stats['average_fps'] == 10.0  # 1/0.1
        assert stats['average_processing_time'] == 0.1

    def test_reset_stats(self):
        """Test resetting performance statistics."""
        detector = self.create_mock_detector()

        # Set some stats
        detector.frame_count = 10
        detector.total_processing_time = 1.0
        detector.detection_history = ['test']

        detector.reset_stats()

        # Verify reset
        stats = detector.get_performance_stats()
        assert stats['frames_processed'] == 0
        assert stats['total_processing_time'] == 0.0
        assert len(detector.detection_history) == 0

    def test_cleanup(self):
        """Test resource cleanup."""
        detector = self.create_mock_detector()

        # Set some resources
        detector.model = Mock()
        detector.detection_history = ['test']

        # Cleanup
        detector.cleanup()

        # Verify cleanup
        assert detector.model is None
        assert len(detector.detection_history) == 0

    def test_is_model_loaded(self):
        """Test model loading status check."""
        detector = self.create_mock_detector()

        # With model loaded
        detector.model = Mock()
        assert detector.is_model_loaded() is True

        # Without model loaded
        detector.model = None
        assert detector.is_model_loaded() is False


class TestModernVehicleDetectorIntegration:
    """Integration tests for ModernVehicleDetector."""

    @patch('src.core.modern_vehicle_detector.ModernVehicleDetector._check_ultralytics_available', return_value=False)
    def test_fallback_to_legacy_when_ultralytics_unavailable(self, mock_ultralytics_check):
        """Test that system falls back gracefully when ultralytics is not available."""
        with patch('src.core.modern_vehicle_detector.get_config'):
            with pytest.raises(ModelLoadingError):
                ModernVehicleDetector()


if __name__ == '__main__':
    pytest.main([__file__])
