import os
import sys
from pathlib import Path

def setup_paths():
    current_file = Path(__file__).resolve()
    base_dir = current_file.parent.parent
    base_dir_str = str(base_dir)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)
    return base_dir
