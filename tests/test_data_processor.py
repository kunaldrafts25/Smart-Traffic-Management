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
Unit tests for DataProcessor

Tests data processing functionality, video handling, and validation.
"""

import pytest
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.utils.data_processor import DataProcessor, VideoStreamInfo, ProcessedFrame
from src.utils.error_handler import DataProcessingError, CameraConnectionError


class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'data': {
                'preprocessing': {
                    'resize_frames': True,
                    'target_size': [640, 480],
                    'normalize': True
                },
                'input_sources': [],
                'output_path': 'test_output/',
                'backup_path': 'test_backup/'
            },
            'performance': {
                'frame_skip_ratio': 2
            }
        }

        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    @patch('src.utils.data_processor.get_config')
    @patch('pathlib.Path.mkdir')
    def test_data_processor_initialization(self, mock_mkdir, mock_get_config):
        """Test DataProcessor initialization."""
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'data.preprocessing.resize_frames': True,
            'data.preprocessing.target_size': [640, 480],
            'data.preprocessing.normalize': True,
            'data.input_sources': [],
            'data.output_path': 'test_output/',
            'data.backup_path': 'test_backup/',
            'performance.frame_skip_ratio': 2
        }.get(key, default)

        processor = DataProcessor()

        assert processor.resize_frames is True
        assert processor.target_size == (640, 480)
        assert processor.normalize is True
        assert processor.frame_skip_ratio == 2
        assert processor.frames_processed == 0

    @patch('src.utils.data_processor.get_config')
    @patch('pathlib.Path.mkdir')
    @patch('cv2.VideoCapture')
    def test_open_video_stream_file(self, mock_videocapture, mock_mkdir, mock_get_config):
        """Test opening video stream from file."""
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'data.preprocessing.resize_frames': True,
            'data.preprocessing.target_size': [640, 480],
            'data.preprocessing.normalize': True,
            'data.input_sources': [],
            'data.output_path': 'test_output/',
            'data.backup_path': 'test_backup/',
            'performance.frame_skip_ratio': 2
        }.get(key, default)

        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900
        }.get(prop, 0)
        mock_videocapture.return_value = mock_cap

        processor = DataProcessor()
        stream_info = processor.open_video_stream('test_video.mp4', 'test_stream')

        assert isinstance(stream_info, VideoStreamInfo)
        assert stream_info.width == 1920
        assert stream_info.height == 1080
        assert stream_info.fps == 30.0
        assert stream_info.frame_count == 900
        assert stream_info.is_live is False
        assert stream_info.duration == 30.0  # 900 frames / 30 fps

    @patch('src.utils.data_processor.get_config')
    @patch('pathlib.Path.mkdir')
    @patch('cv2.VideoCapture')
    def test_open_video_stream_camera(self, mock_videocapture, mock_mkdir, mock_get_config):
        """Test opening video stream from camera."""
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'data.preprocessing.resize_frames': True,
            'data.preprocessing.target_size': [640, 480],
            'data.preprocessing.normalize': True,
            'data.input_sources': [],
            'data.output_path': 'test_output/',
            'data.backup_path': 'test_backup/',
            'performance.frame_skip_ratio': 2
        }.get(key, default)

        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: -1
        }.get(prop, 0)
        mock_videocapture.return_value = mock_cap

        processor = DataProcessor()
        stream_info = processor.open_video_stream(0, 'camera_stream')

        assert isinstance(stream_info, VideoStreamInfo)
        assert stream_info.is_live is True
        assert stream_info.duration == 0.0

    @patch('src.utils.data_processor.get_config')
    @patch('pathlib.Path.mkdir')
    @patch('cv2.VideoCapture')
    def test_open_video_stream_failure(self, mock_videocapture, mock_mkdir, mock_get_config):
        """Test video stream opening failure."""
        mock_get_config.return_value.get.side_effect = lambda key, default=None: {
            'data.preprocessing.resize_frames': True,
            'data.preprocessing.target_size': [640, 480],
            'data.preprocessing.normalize': True,
            'data.input_sources': [],
            'data.output_path': 'test_output/',
            'data.backup_path': 'test_backup/',
            'performance.frame_skip_ratio': 2
        }.get(key, default)

        # Mock failed video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        processor = DataProcessor()

        with pytest.raises(CameraConnectionError):
            processor.open_video_stream('invalid_source', 'test_stream')

    def create_mock_processor(self):
        """Create a mock DataProcessor for testing."""
        with patch('src.utils.data_processor.get_config'), \
             patch('pathlib.Path.mkdir'):

            processor = DataProcessor()
            processor.resize_frames = True
            processor.target_size = (640, 480)
            processor.normalize = True

            return processor

    def test_process_frame(self):
        """Test frame processing."""
        processor = self.create_mock_processor()

        # Test frame processing
        processed_frame = processor._process_frame(self.test_image, 'test_stream')

        assert isinstance(processed_frame, ProcessedFrame)
        assert processed_frame.frame.shape[:2] == (480, 640)  # Should be resized
        assert processed_frame.metadata['stream_id'] == 'test_stream'
        assert processed_frame.metadata['resized'] is True
        assert processed_frame.metadata['normalized'] is True
        assert processed_frame.metadata['processing_time'] > 0

        # Check normalization
        assert processed_frame.frame.dtype == np.float32
        assert 0 <= processed_frame.frame.min() <= 1
        assert 0 <= processed_frame.frame.max() <= 1

    def test_read_frame(self):
        """Test reading frame from stream."""
        processor = self.create_mock_processor()

        # Mock video capture
        mock_cap = Mock()
        mock_cap.read.return_value = (True, self.test_image)
        processor.video_streams['test_stream'] = mock_cap

        # Read frame
        processed_frame = processor.read_frame('test_stream')

        assert isinstance(processed_frame, ProcessedFrame)
        assert processed_frame.frame is not None
        assert processor.frames_processed == 1

    def test_read_frame_no_stream(self):
        """Test reading frame from non-existent stream."""
        processor = self.create_mock_processor()

        with pytest.raises(DataProcessingError):
            processor.read_frame('nonexistent_stream')

    def test_read_frame_end_of_stream(self):
        """Test reading frame at end of stream."""
        processor = self.create_mock_processor()

        # Mock video capture returning no frame
        mock_cap = Mock()
        mock_cap.read.return_value = (False, None)
        processor.video_streams['test_stream'] = mock_cap

        # Should return None at end of stream
        result = processor.read_frame('test_stream')
        assert result is None

    def test_read_frames_batch(self):
        """Test reading multiple frames."""
        processor = self.create_mock_processor()

        # Mock successful frame reading
        with patch.object(processor, 'read_frame') as mock_read:
            mock_frame = ProcessedFrame(
                frame=self.test_image,
                frame_id=1,
                timestamp=1234567890.0,
                metadata={}
            )
            mock_read.return_value = mock_frame

            frames = processor.read_frames_batch('test_stream', batch_size=3)

        assert len(frames) == 3
        assert all(isinstance(f, ProcessedFrame) for f in frames)
        assert mock_read.call_count == 3

    def test_extract_features(self):
        """Test feature extraction from frame."""
        processor = self.create_mock_processor()

        features = processor.extract_features(self.test_image)

        assert isinstance(features, dict)
        assert 'mean_rgb' in features
        assert 'std_rgb' in features
        assert 'edge_density' in features
        assert 'contrast' in features
        assert 'brightness' in features

        # Check feature types
        assert isinstance(features['mean_rgb'], list)
        assert len(features['mean_rgb']) == 3  # RGB channels
        assert isinstance(features['edge_density'], float)
        assert 0 <= features['edge_density'] <= 1

    def test_extract_features_grayscale(self):
        """Test feature extraction from grayscale frame."""
        processor = self.create_mock_processor()

        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        features = processor.extract_features(gray_image)

        assert isinstance(features, dict)
        assert 'mean_intensity' in features
        assert 'std_intensity' in features
        assert 'edge_density' in features

    def test_validate_data_numpy_array(self):
        """Test data validation for numpy arrays."""
        processor = self.create_mock_processor()

        # Valid array
        valid_array = np.random.rand(100, 100, 3).astype(np.float32)
        is_valid, errors = processor.validate_data(valid_array)
        assert is_valid is True
        assert len(errors) == 0

        # Array with NaN values
        invalid_array = valid_array.copy()
        invalid_array[0, 0, 0] = np.nan
        is_valid, errors = processor.validate_data(invalid_array)
        assert is_valid is False
        assert any('NaN' in error for error in errors)

        # Empty array
        empty_array = np.array([])
        is_valid, errors = processor.validate_data(empty_array)
        assert is_valid is False
        assert any('empty' in error for error in errors)

    def test_validate_data_dataframe(self):
        """Test data validation for pandas DataFrames."""
        processor = self.create_mock_processor()

        # Valid DataFrame
        valid_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        is_valid, errors = processor.validate_data(valid_df)
        assert is_valid is True
        assert len(errors) == 0

        # DataFrame with null values
        invalid_df = valid_df.copy()
        invalid_df.loc[0, 'col1'] = None
        is_valid, errors = processor.validate_data(invalid_df)
        assert is_valid is False
        assert any('null' in error for error in errors)

        # Empty DataFrame
        empty_df = pd.DataFrame()
        is_valid, errors = processor.validate_data(empty_df)
        assert is_valid is False
        assert any('empty' in error for error in errors)

    def test_validate_data_dictionary(self):
        """Test data validation for dictionaries."""
        processor = self.create_mock_processor()

        # Valid dictionary
        valid_dict = {'key1': 'value1', 'key2': 'value2'}
        is_valid, errors = processor.validate_data(valid_dict)
        assert is_valid is True
        assert len(errors) == 0

        # Dictionary with None values
        invalid_dict = {'key1': 'value1', 'key2': None}
        is_valid, errors = processor.validate_data(invalid_dict)
        assert is_valid is False
        assert any('None' in error for error in errors)

        # Empty dictionary
        empty_dict = {}
        is_valid, errors = processor.validate_data(empty_dict)
        assert is_valid is False
        assert any('empty' in error for error in errors)

    def test_save_processed_data_json(self):
        """Test saving data as JSON."""
        processor = self.create_mock_processor()

        test_data = {'key1': 'value1', 'key2': [1, 2, 3]}

        with tempfile.TemporaryDirectory() as temp_dir:
            processor.output_path = temp_dir

            saved_path = processor.save_processed_data(test_data, 'test.json', 'json')

            assert os.path.exists(saved_path)

            # Verify content
            import json
            with open(saved_path, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data

    def test_save_processed_data_csv(self):
        """Test saving DataFrame as CSV."""
        processor = self.create_mock_processor()

        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            processor.output_path = temp_dir

            saved_path = processor.save_processed_data(test_df, 'test.csv', 'csv')

            assert os.path.exists(saved_path)

            # Verify content
            loaded_df = pd.read_csv(saved_path)
            pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_save_processed_data_npy(self):
        """Test saving numpy array as NPY."""
        processor = self.create_mock_processor()

        test_array = np.random.rand(10, 10)

        with tempfile.TemporaryDirectory() as temp_dir:
            processor.output_path = temp_dir

            saved_path = processor.save_processed_data(test_array, 'test.npy', 'npy')

            assert os.path.exists(saved_path)

            # Verify content
            loaded_array = np.load(saved_path)
            np.testing.assert_array_equal(loaded_array, test_array)

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        processor = self.create_mock_processor()

        # Initially should have zero stats
        stats = processor.get_performance_stats()
        assert stats['frames_processed'] == 0
        assert stats['processing_fps'] == 0.0

        # Mock some processing
        processor.frames_processed = 10
        processor.total_processing_time = 1.0

        stats = processor.get_performance_stats()
        assert stats['frames_processed'] == 10
        assert stats['average_processing_time'] == 0.1
        assert stats['processing_fps'] == 10.0

    def test_reset_stats(self):
        """Test resetting performance statistics."""
        processor = self.create_mock_processor()

        # Set some stats
        processor.frames_processed = 10
        processor.total_processing_time = 1.0
        processor.errors_count = 5

        processor.reset_stats()

        # Verify reset
        stats = processor.get_performance_stats()
        assert stats['frames_processed'] == 0
        assert stats['total_processing_time'] == 0.0
        assert stats['errors_count'] == 0

    def test_close_video_stream(self):
        """Test closing video stream."""
        processor = self.create_mock_processor()

        # Mock video capture
        mock_cap = Mock()
        processor.video_streams['test_stream'] = mock_cap
        processor.stream_info['test_stream'] = VideoStreamInfo(
            source='test', width=640, height=480, fps=30.0,
            frame_count=100, duration=3.33, is_live=False
        )

        # Close stream
        processor.close_video_stream('test_stream')

        # Verify cleanup
        assert 'test_stream' not in processor.video_streams
        assert 'test_stream' not in processor.stream_info
        mock_cap.release.assert_called_once()

    def test_cleanup(self):
        """Test resource cleanup."""
        processor = self.create_mock_processor()

        mock_cap1 = Mock()
        mock_cap2 = Mock()
        processor.video_streams['stream1'] = mock_cap1
        processor.video_streams['stream2'] = mock_cap2

        processor.cleanup()

        # Verify all streams are closed
        assert len(processor.video_streams) == 0
        mock_cap1.release.assert_called_once()
        mock_cap2.release.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
