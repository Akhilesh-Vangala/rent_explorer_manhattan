import streamlit as st
import sys
from pathlib import Path
code_dir = Path(__file__).resolve().parent.parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
from app.path_setup import setup_paths
setup_paths()
from app.shared_styles import apply_shared_styles
apply_shared_styles()
st.title('Project Overview')
st.markdown('**SmartRent Manhattan - ML-Powered Rental Price Prediction**')
st.markdown('---')
st.subheader('About This Project')
st.markdown('\nSmartRent Manhattan is a comprehensive rental price prediction system that uses machine learning\nto estimate monthly rent for Manhattan apartments based on property characteristics, amenities, and neighborhood.\n\nThe system demonstrates end-to-end data science workflow including data preprocessing, feature engineering,\nmodel training, interpretability analysis, and deployment via an interactive web application.\n')
st.markdown('---')
st.subheader('Dataset Description')
st.markdown('\n**Source:** StreetEasy-style NYC rental listings dataset\n\n**Size:** 3,539 rental listings\n\n**Features:** 18 property characteristics including:\n- Property details (bedrooms, bathrooms, size, floor, building age)\n- Location (neighborhood, distance to subway)\n- Amenities (doorman, elevator, gym, roofdeck, etc.)\n\n**Data Quality:** No missing values across all columns\n')
st.markdown('---')
st.subheader('Architecture Overview')
st.markdown('\n**Pipeline:**\n1. Data Preprocessing (cleaning, type casting, outlier handling)\n2. Feature Engineering (one-hot encoding, log transforms, rent per sqft)\n3. Model Training (Linear Regression, Random Forest, XGBoost)\n4. Model Evaluation (RMSE, MAE, R²)\n5. Model Persistence (best model saved)\n6. Web Application Deployment (Streamlit)\n')
st.markdown('---')
st.subheader('Final Deliverables')
st.markdown('\n- **Best Model:** XGBoost Regressor\n- **Performance:** Test R² of 0.812, RMSE of $1,370, MAE of $762\n- **Cross-Validation:** R² = 0.849 ± 0.007 (5-fold)\n- **Feature Count:** 47 features after preprocessing\n- **Training Samples:** 2,478\n- **Validation Samples:** 531\n- **Test Samples:** 530\n- **Neighborhoods:** 32 Manhattan neighborhoods\n- **Interactive Web Application:** Multi-page Streamlit dashboard\n')
st.markdown('---')
if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')
