#!/usr/bin/env python
"""Tests for config classes."""

from omegaconf import OmegaConf

from pytorch_tabular.config import TrainerConfig


class TestTrainerConfig:
    """Tests for TrainerConfig class."""

    def test_devices_list_to_devices_conversion(self):
        """Test that devices_list is properly converted to devices."""
        # Test with a list of devices
        trainer_config = TrainerConfig(devices_list=[0, 1])
        assert trainer_config.devices == [0, 1]

        # Wrap with OmegaConf as done in TabularModel
        config = OmegaConf.structured(trainer_config)
        assert config.devices == [0, 1]

    def test_devices_list_multiple_gpus(self):
        """Test devices_list with multiple GPU IDs as documented."""
        trainer_config = TrainerConfig(devices_list=[1, 2, 3, 4])
        assert trainer_config.devices == [1, 2, 3, 4]

        config = OmegaConf.structured(trainer_config)
        assert config.devices == [1, 2, 3, 4]

    def test_devices_int_value(self):
        """Test that devices accepts integer values."""
        trainer_config = TrainerConfig(devices=2)
        assert trainer_config.devices == 2

        config = OmegaConf.structured(trainer_config)
        assert config.devices == 2

    def test_devices_default_value(self):
        """Test that devices has default value of -1."""
        trainer_config = TrainerConfig()
        assert trainer_config.devices == -1

        config = OmegaConf.structured(trainer_config)
        assert config.devices == -1

    def test_devices_list_single_device(self):
        """Test devices_list with a single device."""
        trainer_config = TrainerConfig(devices_list=[0])
        assert trainer_config.devices == [0]

        config = OmegaConf.structured(trainer_config)
        assert config.devices == [0]

    def test_devices_list_precedence(self):
        """Test that devices_list takes precedence over devices."""
        # When both are provided, devices_list should take precedence
        trainer_config = TrainerConfig(devices=2, devices_list=[0, 1])
        assert trainer_config.devices == [0, 1]

        config = OmegaConf.structured(trainer_config)
        assert config.devices == [0, 1]

    def test_omegaconf_merge_compatibility(self):
        """Test that config works correctly with OmegaConf.merge."""
        trainer_config = TrainerConfig(devices_list=[0, 1], max_epochs=10)
        config = OmegaConf.structured(trainer_config)

        # Simulate merging as done in TabularModel
        merged = OmegaConf.merge(OmegaConf.to_container(config), {"accelerator": "gpu"})

        assert merged.devices == [0, 1]
        assert merged.max_epochs == 10
        assert merged.accelerator == "gpu"
