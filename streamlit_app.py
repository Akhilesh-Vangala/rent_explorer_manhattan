"""
Streamlit Cloud Entry Point
This file is the root-level entry point for Streamlit Cloud deployment.
It executes the main app from app/Rent_Estimation.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)

# Setup paths first
try:
    from app.path_setup import setup_paths
    setup_paths()
except Exception:
    pass

# Read and execute the main app file
main_app_path = project_root / 'app' / 'Rent_Estimation.py'
with open(main_app_path, 'r', encoding='utf-8') as f:
    code = f.read()
    
# Execute the code in the current namespace
exec(compile(code, str(main_app_path), 'exec'), globals())

