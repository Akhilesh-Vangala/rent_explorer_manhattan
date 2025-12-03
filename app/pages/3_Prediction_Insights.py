import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
code_dir = Path(__file__).resolve().parent.parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
from app.path_setup import setup_paths
setup_paths()
from src.visualization import plot_feature_importance
from src.utils import load_data, load_model
from app.shared_styles import apply_shared_styles
apply_shared_styles()
model = load_model()
(df_raw, df_preprocessed) = load_data()
if 'prediction' not in st.session_state or st.session_state.prediction is None:
    st.title('Prediction Insights')
    st.warning('No prediction found. Please return to the Home page and make a prediction first.')
    if st.button('Go to Home Page'):
        st.switch_page('Rent_Estimation.py')
    st.stop()
st.title('Prediction Insights')
predicted_rent = st.session_state.prediction
inputs = st.session_state.inputs
st.markdown('---')
st.subheader('Top Feature Importance')
feature_importance = model.feature_importances_
feature_names = model.feature_names_in_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
fig = plot_feature_importance(importance_df, top_n=15, exclude_features=['log_rent', 'rent_per_sqft'], figsize=(9, 5))
st.pyplot(fig)
plt.close()
st.subheader('Key Drivers for Your Prediction')
key_features = {'Size': inputs['size_sqft'], 'Bedrooms': inputs['bedrooms'], 'Bathrooms': inputs['bathrooms'], 'Neighborhood': inputs['neighborhood'], 'Floor': inputs['floor']}
st.markdown('**Primary Factors:**')
for (feature, value) in key_features.items():
    st.write(f'- **{feature}**: {value}')
amenity_list = []
amenities_map = {'no_fee': 'No Fee', 'has_roofdeck': 'Roofdeck', 'has_washer_dryer': 'Washer/Dryer', 'has_doorman': 'Doorman', 'has_elevator': 'Elevator', 'has_dishwasher': 'Dishwasher', 'has_patio': 'Patio', 'has_gym': 'Gym'}
for (key, label) in amenities_map.items():
    if inputs.get(key, False):
        amenity_list.append(label)
if amenity_list:
    st.markdown('**Selected Amenities:**')
    st.write(', '.join(amenity_list))
else:
    st.markdown('**Selected Amenities:** None')
st.markdown('---')
if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')
