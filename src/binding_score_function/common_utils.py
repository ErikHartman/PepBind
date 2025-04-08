"""
Common utility functions used across the binding score function modules.
"""

import os
import logging
from typing import Dict
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_directory_structure() -> Dict[str, str]:
    """
    Set up and create the directory structure for the pipeline.
    Uses environment variables if available, otherwise defaults to standard paths.

    Returns:
        Dict[str, str]: A dictionary containing all path definitions
    """
    load_dotenv()
    base_dir = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")

    # Define primary directories
    paths = {
        "base_dir": base_dir,
        "index_dir": os.path.abspath(os.path.join(base_dir, "inputs/pdbbind_index_files")),
        "output_dir": os.path.abspath(os.path.join(base_dir, "outputs/binding_score_function")),
    }

    # Create directories if they don't exist
    for path_name, path_value in paths.items():
        if path_name.endswith("_dir") and not os.path.exists(path_value):
            os.makedirs(path_value, exist_ok=True)
            logger.info(f"Created directory: {path_value}")

    return paths


def set_permissions(directory: str) -> None:
    """
    Recursively sets read/write/execute permissions (777) on all files and subdirectories.
    
    Args:
        directory (str): The root directory to start permission changes
    """
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions for {d}: {str(e)}")
                
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions for {f}: {str(e)}")
    
    logger.info(f"Set permissions to 777 for all files in {directory}")


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to create
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
