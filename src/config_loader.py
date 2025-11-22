"""Configuration loader for HMM POS Tagger."""

import configparser
import os


def load_config(config_path='config.cfg'):
    """Load configuration from .cfg file."""
    config = configparser.ConfigParser()
    
    if not os.path.isabs(config_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config.read(config_path)
    return config


def get_dataset_config(config):
    """Get dataset configuration."""
    return {
        'dataset_dir': config.get('dataset', 'dataset_dir'),
        'train_file': config.get('dataset', 'train_file'),
        'dev_file': config.get('dataset', 'dev_file'),
        'test_file': config.get('dataset', 'test_file'),
    }


def get_training_config(config):
    """Get training configuration."""
    return {
        'save_model': config.getboolean('training', 'save_model'),
        'model_name': config.get('training', 'model_name'),
        'model_dir': config.get('training', 'model_dir'),
    }


def get_output_config(config):
    """Get output configuration."""
    return {
        'verbose': config.getboolean('output', 'verbose'),
        'print_hmm_summary': config.getboolean('output', 'print_hmm_summary'),
        'print_dataset_stats': config.getboolean('output', 'print_dataset_stats'),
    }
