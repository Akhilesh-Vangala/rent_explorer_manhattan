"""
Streamlit Cloud Entry Point
This file is the root-level entry point for Streamlit Cloud deployment.
It redirects to the main app in app/Rent_Estimation.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the main app module (this will execute all the Streamlit code)
import app.Rent_Estimation

