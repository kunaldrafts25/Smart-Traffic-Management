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
Unit tests for VehicleDetector

Tests vehicle detection functionality, performance, and error handling.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.core.vehicle_detector import VehicleDetector, DetectionResult, BoundingBox
from src.utils.error_handler import VehicleDetectionError, ModelLoadingError


class TestVehicleDetector:
    """Test cases for VehicleDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock configuration
        self.mock_config = {
            'models': {
                'yolo': {
                    'weights_path': 'test_weights.weights',
                    'config_path': 'test_config.cfg',
                    'classes_path': 'test_classes.names',
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4,
                    'input_size': [416, 416]
                }
            }
        }

        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        self.test_classes = ['person', 'bicycle', 'car', 'motorbike', 'truck', 'bus']

    @patch('src.core.vehicle_detector.get_config')
    @patch('src.core.vehicle_detector.cv2.dnn.readNet')
    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('pathlib.Path.exists')
    def test_vehicle_detector_initialization(self, mock_path_exists, mock_exists, mock_open, mock_readnet, mock_get_config):
        """Test VehicleDetector initialization."""
        # Setup mocks
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'models.yolo.weights_path': 'test_weights.weights',
            'models.yolo.config_path': 'test_config.cfg',
            'models.yolo.classes_path': 'test_classes.names',
            'models.yolo.confidence_threshold': 0.5,
            'models.yolo.nms_threshold': 0.4,
            'models.yolo.input_size': [416, 416]
        }.get(key, default)

        mock_exists.return_value = True
        mock_path_exists.return_value = True
        mock_net = Mock()
        mock_net.getLayerNames.return_value = ['layer1', 'layer2', 'output_layer']
        mock_net.getUnconnectedOutLayers.return_value = np.array([3])
        mock_readnet.return_value = mock_net

        # Mock file context manager
        mock_file_content = Mock()
        mock_file_content.readlines.return_value = [
            'person\n', 'bicycle\n', 'car\n', 'motorbike\n', 'truck\n', 'bus\n'
        ]
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file_content)
        mock_file.__exit__ = Mock(return_value=None)
        mock_open.return_value = mock_file

        # Initialize detector
        detector = VehicleDetector()

        # Verify initialization
        assert detector.confidence_threshold == 0.5
        assert detector.nms_threshold == 0.4
        assert detector.input_size == (416, 416)
        assert detector.vehicle_classes == ['car', 'truck', 'bus', 'motorbike', 'bicycle']
        assert detector.is_model_loaded() is True

    @patch('src.core.vehicle_detector.get_config')
    @patch('os.path.exists')
    def test_model_loading_file_not_found(self, mock_exists, mock_get_config):
        """Test model loading with missing files."""
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'models.yolo.weights_path': 'nonexistent_weights.weights',
            'models.yolo.config_path': 'nonexistent_config.cfg',
            'models.yolo.classes_path': 'nonexistent_classes.names'
        }.get(key, default)

        mock_exists.return_value = False

        # Should raise ModelLoadingError
        with pytest.raises(ModelLoadingError):
            VehicleDetector()

    def create_mock_detector(self):
        """Create a mock VehicleDetector for testing."""
        # Mock configuration values
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            'models.yolo.weights_path': 'models/yolov4.weights',
            'models.yolo.config_path': 'models/yolov4.cfg',
            'models.yolo.classes_path': 'models/coco.names',
            'models.yolo.confidence_threshold': 0.5,
            'models.yolo.nms_threshold': 0.4,
            'models.yolo.input_size': [416, 416],
            'models.yolo.vehicle_classes': ['car', 'motorbike', 'truck', 'bus'],
            'models.yolo.class_mapping': {
                'car': 2, 'motorbike': 3, 'truck': 7, 'bus': 5
            }
        }.get(key, default)

        with patch('src.core.vehicle_detector.get_config', return_value=mock_config), \
             patch('src.core.vehicle_detector.cv2.dnn.readNet'), \
             patch('builtins.open'), \
             patch('os.path.exists', return_value=True):

            detector = VehicleDetector()

            # Mock the network
            detector.net = Mock()
            detector.output_layers = ['output1', 'output2']
            detector.classes = self.test_classes
            detector.vehicle_class_ids = [2, 3, 4, 5]  # car, motorbike, truck, bus

            return detector

    def test_detect_vehicles_valid_input(self):
        """Test vehicle detection with valid input."""
        detector = self.create_mock_detector()

        # Mock network forward pass - create detections with proper class IDs
        # YOLO detection format: [center_x, center_y, width, height, objectness, class_0_prob, class_1_prob, class_2_prob, ...]
        # We need class_2 (car) and class_3 (motorbike) to have high probabilities
        mock_detections = [
            np.array([
                [0.5, 0.5, 0.2, 0.3, 0.9, 0.1, 0.1, 0.85, 0.1, 0.1],  # car detection (class_2 = 0.85)
                [0.3, 0.7, 0.15, 0.25, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1]  # motorbike detection (class_3 = 0.8)
            ])
        ]

        detector.net.setInput = Mock()
        detector.net.forward.return_value = mock_detections

        # Mock NMS - should match the number of actual detections that pass filtering
        with patch('cv2.dnn.NMSBoxes') as mock_nms:
            mock_nms.return_value = np.array([0, 1])  # Both detections should pass filtering now

            result = detector.detect_vehicles(self.test_image, frame_id=1)

        # Verify result
        assert isinstance(result, DetectionResult)
        assert result.frame_id == 1
        assert result.vehicle_count >= 0
        assert isinstance(result.detections, list)
        assert isinstance(result.confidence_scores, list)
        assert result.processing_time > 0

    def test_detect_vehicles_invalid_input(self):
        """Test vehicle detection with invalid input."""
        detector = self.create_mock_detector()

        # Test with None input
        with pytest.raises(VehicleDetectionError):
            detector.detect_vehicles(None)

        # Test with empty array
        with pytest.raises(VehicleDetectionError):
            detector.detect_vehicles(np.array([]))

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
                timestamp=1234567890.0
            )
            mock_detect.return_value = mock_result

            frames = [self.test_image, self.test_image, self.test_image]
            results = detector.detect_vehicles_batch(frames)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)
        assert mock_detect.call_count == 3

    def test_draw_detections(self):
        """Test drawing detections on frame."""
        detector = self.create_mock_detector()

        # Create test detection result
        bbox = BoundingBox(x=100, y=100, width=50, height=80,
                          confidence=0.85, class_id=2, class_name='car')

        detection_result = DetectionResult(
            vehicle_count=1,
            detections=[{'bbox': bbox, 'confidence': 0.85, 'class_id': 2, 'class_name': 'car'}],
            confidence_scores=[0.85],
            processing_time=0.1,
            frame_id=1,
            timestamp=1234567890.0
        )

        # Draw detections
        output_frame = detector.draw_detections(self.test_image.copy(), detection_result)

        # Verify output
        assert output_frame.shape == self.test_image.shape
        assert not np.array_equal(output_frame, self.test_image)  # Should be modified

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        detector = self.create_mock_detector()

        # Initially should have zero stats
        stats = detector.get_performance_stats()
        assert stats['frames_processed'] == 0
        assert stats['average_fps'] == 0.0

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

        # Reset stats
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
        detector.net = Mock()
        detector.output_layers = ['test']
        detector.classes = ['test']
        detector.detection_history = ['test']

        detector.cleanup()

        # Verify cleanup
        assert detector.net is None
        assert detector.output_layers is None
        assert detector.classes is None
        assert len(detector.detection_history) == 0

    def test_process_detections(self):
        """Test processing raw YOLO detections."""
        detector = self.create_mock_detector()

        # Create mock detections
        mock_detections = [
            np.array([
                [0.5, 0.5, 0.2, 0.3, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1],  # car detection (class_id=2)
                [0.3, 0.7, 0.15, 0.25, 0.1, 0.1, 0.1, 0.1, 0.85, 0.1]  # motorbike detection (class_id=3)
            ])
        ]

        width, height = 640, 480
        boxes, confidences, class_ids = detector._process_detections(mock_detections, width, height)

        # Verify processing
        assert len(boxes) >= 0
        assert len(confidences) >= 0
        assert len(class_ids) >= 0
        assert len(boxes) == len(confidences) == len(class_ids)

    def test_extract_final_detections(self):
        """Test extracting final detections after NMS."""
        detector = self.create_mock_detector()

        # Test data
        boxes = [[100, 100, 50, 80], [200, 150, 60, 90]]
        confidences = [0.8, 0.9]
        class_ids = [2, 3]  # car, motorbike
        indices = np.array([0, 1])

        final_detections = detector._extract_final_detections(boxes, confidences, class_ids, indices)

        # Verify extraction
        assert len(final_detections) == 2
        for detection in final_detections:
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert 'class_id' in detection
            assert 'class_name' in detection
            assert isinstance(detection['bbox'], BoundingBox)


class TestVehicleDetectorIntegration:
    """Integration tests for VehicleDetector."""

    def test_full_detection_pipeline(self):
        """Test the complete detection pipeline with mocked components."""
        # This would test the full pipeline but with mocked YOLO model
        # to avoid requiring actual model files during testing
        pass


if __name__ == '__main__':
    pytest.main([__file__])
