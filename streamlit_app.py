"""
Streamlit Cloud Entry Point
This file is the root-level entry point for Streamlit Cloud deployment.
It imports and runs the main app from app/Rent_Estimation.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the main app
from app.Rent_Estimation import *

