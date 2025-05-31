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
Unit tests for ConfigManager

Tests configuration loading, validation, and management functionality.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path

from src.utils.config_manager import ConfigManager, get_config, init_config


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            'app': {
                'name': 'Test TMS',
                'version': '1.0.0',
                'debug': True
            },
            'models': {
                'yolo': {
                    'weights_path': 'test_weights.weights',
                    'config_path': 'test_config.cfg',
                    'confidence_threshold': 0.6
                }
            },
            'data': {
                'input_sources': [],
                'output_path': 'test_output/',
                'backup_path': 'test_backup/'
            },
            'traffic_signals': {
                'intersections': [],
                'timing_constraints': {
                    'min_green': 15,
                    'max_green': 120
                }
            },
            'database': {
                'type': 'sqlite',
                'path': 'test.db'
            }
        }

    def test_config_loading(self):
        """Test configuration loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            assert config_manager.get('app.name') == 'Test TMS'
            assert config_manager.get('app.version') == '1.0.0'
            assert config_manager.get('models.yolo.confidence_threshold') == 0.6

        finally:
            os.unlink(config_path)

    def test_config_get_with_default(self):
        """Test getting configuration values with defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # Existing value
            assert config_manager.get('app.name') == 'Test TMS'

            # Non-existing value with default
            assert config_manager.get('nonexistent.key', 'default_value') == 'default_value'

            # Non-existing value without default
            assert config_manager.get('nonexistent.key') is None

        finally:
            os.unlink(config_path)

    def test_config_set(self):
        """Test setting configuration values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # Set new value
            config_manager.set('new.nested.key', 'new_value')
            assert config_manager.get('new.nested.key') == 'new_value'

            # Update existing value
            config_manager.set('app.name', 'Updated TMS')
            assert config_manager.get('app.name') == 'Updated TMS'

        finally:
            os.unlink(config_path)

    def test_config_section(self):
        """Test getting entire configuration sections."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            app_section = config_manager.get_section('app')
            assert app_section['name'] == 'Test TMS'
            assert app_section['version'] == '1.0.0'
            assert app_section['debug'] is True

            models_section = config_manager.get_section('models')
            assert 'yolo' in models_section

        finally:
            os.unlink(config_path)

    def test_config_validation(self):
        """Test configuration validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # Should pass validation (has required sections)
            is_valid = config_manager.validate_config()
            assert is_valid is True

        finally:
            os.unlink(config_path)

    def test_config_validation_missing_sections(self):
        """Test configuration validation with missing sections."""
        incomplete_config = {
            'app': {
                'name': 'Test TMS'
            }
            # Missing required sections: models, data, traffic_signals
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(incomplete_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # Should fail validation (missing required sections)
            is_valid = config_manager.validate_config()
            assert is_valid is False

        finally:
            os.unlink(config_path)

    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            # Set environment variable
            os.environ['TMS_APP_DEBUG'] = 'false'

            config_manager = ConfigManager(config_path)

            # Should be overridden by environment variable
            assert config_manager.get('app.debug') is False

        finally:
            # Clean up
            if 'TMS_APP_DEBUG' in os.environ:
                del os.environ['TMS_APP_DEBUG']
            os.unlink(config_path)

    def test_config_save(self):
        """Test saving configuration to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # Modify configuration
            config_manager.set('app.name', 'Modified TMS')

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
                save_path = f2.name

            config_manager.save_config(save_path)

            # Load saved configuration
            new_config_manager = ConfigManager(save_path)
            assert new_config_manager.get('app.name') == 'Modified TMS'

            os.unlink(save_path)

        finally:
            os.unlink(config_path)

    def test_config_reload(self):
        """Test configuration reloading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            original_name = config_manager.get('app.name')

            # Modify the file externally
            modified_config = self.test_config.copy()
            modified_config['app']['name'] = 'Externally Modified TMS'

            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Reload configuration
            config_manager.reload_config()

            # Should reflect the external change
            assert config_manager.get('app.name') == 'Externally Modified TMS'
            assert config_manager.get('app.name') != original_name

        finally:
            os.unlink(config_path)

    def test_global_config_instance(self):
        """Test global configuration instance management."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            config1 = init_config(config_path)
            config2 = get_config()

            # Should be the same instance
            assert config1 is config2
            assert config1.get('app.name') == config2.get('app.name')

        finally:
            os.unlink(config_path)


class TestConfigManagerErrors:
    """Test error handling in ConfigManager."""

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager('nonexistent_config.yaml')

    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__])
