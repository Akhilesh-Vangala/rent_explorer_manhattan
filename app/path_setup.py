"""
Path setup utility for Streamlit pages.

This module must be imported FIRST before any src imports to ensure
the base directory is in sys.path.
"""

import os
import sys
from pathlib import Path


def setup_paths():
    """
    Add the project base directory to sys.path.
    
    This function calculates the base directory relative to the app folder
    and adds it to sys.path so that src imports work correctly.
    
    Should be called at the very beginning of each Streamlit page,
    before any imports from src.
    """
    # Get the directory of this file (app/path_setup.py)
    current_file = Path(__file__).resolve()
    # Go up one level: app/path_setup.py -> app/ -> project_root/
    base_dir = current_file.parent.parent
    
    base_dir_str = str(base_dir)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)
    
    return base_dir
