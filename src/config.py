"""
Project configuration and directory paths.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dataset (download separately)
DATASET_DIR = PROJECT_ROOT / "fer2013"

# Model file
EMOTION_MODEL = MODELS_DIR / "model_v6_23.hdf5"