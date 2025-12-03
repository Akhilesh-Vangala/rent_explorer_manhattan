import streamlit as st

# IMPORTANT: Setup paths BEFORE any src imports
import sys
from pathlib import Path
# Add code directory to path for app imports
code_dir = Path(__file__).resolve().parent.parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
from app.path_setup import setup_paths
setup_paths()

from app.shared_styles import apply_shared_styles

# Apply shared styles
apply_shared_styles()

st.title('Project Overview')
st.markdown('**SmartRent Manhattan - ML-Powered Rental Price Prediction**')

st.markdown('---')

st.subheader('About This Project')

st.markdown("""
SmartRent Manhattan is a comprehensive rental price prediction system that uses machine learning 
to estimate monthly rent for Manhattan apartments based on property characteristics, amenities, and neighborhood.

The system demonstrates end-to-end data science workflow including data preprocessing, feature engineering, 
model training, interpretability analysis, and deployment via an interactive web application.
""")

st.markdown('---')

st.subheader('Dataset Description')

st.markdown("""
**Source:** StreetEasy-style NYC rental listings dataset

**Size:** 3,539 rental listings

**Features:** 18 property characteristics including:
- Property details (bedrooms, bathrooms, size, floor, building age)
- Location (neighborhood, distance to subway)
- Amenities (doorman, elevator, gym, roofdeck, etc.)

**Data Quality:** No missing values across all columns
""")

st.markdown('---')

st.subheader('Architecture Overview')

st.markdown("""
**Pipeline:**
1. Data Preprocessing (cleaning, type casting, outlier handling)
2. Feature Engineering (one-hot encoding, log transforms, rent per sqft)
3. Model Training (Linear Regression, Random Forest, XGBoost)
4. Model Evaluation (RMSE, MAE, R²)
5. Model Persistence (best model saved)
6. Web Application Deployment (Streamlit)
""")

st.markdown('---')

st.subheader('Final Deliverables')

st.markdown("""
- **Best Model:** XGBoost Regressor
- **Performance:** RMSE of $8.34, MAE of $2.51, R² of 0.9999
- **Feature Count:** 49 features after preprocessing
- **Training Samples:** 2,831
- **Test Samples:** 708
- **Neighborhoods:** 32 Manhattan neighborhoods
- **Interactive Web Application:** Multi-page Streamlit dashboard
""")

st.markdown('---')

if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')

