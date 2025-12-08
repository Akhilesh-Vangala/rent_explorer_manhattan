import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path

# CRITICAL: st.set_page_config MUST be the first Streamlit command
st.set_page_config(page_title='Rent Estimation Tool', page_icon=None, layout='wide', initial_sidebar_state='expanded')

# Setup paths FIRST before any other imports
try:
    from app.path_setup import setup_paths
    base_dir = setup_paths()
except Exception as e:
    st.error(f"Path setup failed: {str(e)}")
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
    st.stop()

# Import after path setup
try:
    from src.utils import load_data, load_model, get_neighborhoods
    from app.shared_styles import apply_shared_styles
except Exception as e:
    st.error(f"Import failed: {str(e)}")
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
        st.code(f"Python path: {sys.path[:5]}")
    st.stop()

# Apply styles
try:
    apply_shared_styles()
except Exception as e:
    st.warning(f"Style loading failed: {str(e)}")

# Load data and model with error handling
try:
    from src.config.config import Config
    
    # Debug info in expander (optional to view)
    with st.expander("🔍 Debug Info (click to view)", expanded=False):
        st.write(f"Base directory: {Config.BASE_DIR}")
        st.write(f"Data file path: {Config.RAW_DATA_FILE}")
        st.write(f"Model file path: {Config.BEST_MODEL_FILE}")
        st.write(f"Data exists: {Config.RAW_DATA_FILE.exists()}")
        st.write(f"Model exists: {Config.BEST_MODEL_FILE.exists()}")
        st.write(f"Current directory: {Path.cwd()}")
    
    model = load_model()
    (df_raw, df_preprocessed) = load_data()
    neighborhoods = get_neighborhoods(df_preprocessed)
    
except FileNotFoundError as e:
    st.error(f"❌ File not found: {str(e)}")
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
        st.write(f"Looking for files in: {Config.BASE_DIR if 'Config' in dir() else 'Unknown'}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading app: {str(e)}")
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
        try:
            st.write(f"Current directory: {Path.cwd()}")
            st.write(f"Python path (first 5): {sys.path[:5]}")
        except:
            pass
    st.stop()

AFFORDABILITY_CRITERIA = {-15: ['Major Affordable', '#10b981'], -5: ['Affordable', '#34d399'], 5: ['Fairly Priced', '#fbbf24'], 15: ['Premium', '#f97316']}
with st.sidebar:
    st.markdown('\n    <div class="sidebar-header" style="margin-bottom: 0.5rem;">\n        <div style="font-size: 1rem; font-weight: 700; color: #93c5fd; margin-bottom: 0.1rem; padding-bottom: 0.15rem;">Content</div>\n        <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 0.1rem; line-height: 1.3;">Use the pages below to explore different aspects of the rental prediction system.</div>\n    </div>\n    ', unsafe_allow_html=True)
    st.markdown('\n    <div class="sidebar-header" style="margin-bottom: 0.5rem; padding-top: 0.5rem;">\n        <h2 style="color: #93c5fd; margin: 0; font-size: 1.5rem; font-weight: 700;">Prediction Tool</h2>\n        <p style="color: #ffffff; margin: 0.25rem 0 0 0; font-size: 0.9rem;">SmartRent Manhattan</p>\n        <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.2); margin: 0.5rem 0;">\n    </div>\n    ', unsafe_allow_html=True)
if 'inputs' not in st.session_state:
    st.session_state.inputs = {'bedrooms': 2, 'bathrooms': 1, 'size_sqft': 1000, 'min_to_subway': 5, 'floor': 0, 'building_age_yrs': 5, 'no_fee': False, 'has_roofdeck': False, 'has_washer_dryer': False, 'has_doorman': False, 'has_elevator': False, 'has_dishwasher': False, 'has_patio': False, 'has_gym': False, 'neighborhood': 'Select Neighborhood...'}
neighborhoods_with_placeholder = ['Select Neighborhood...'] + sorted(neighborhoods)
st.markdown('<h1 style="text-align: center; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-top: 0.25rem; padding-top: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.15rem;">SmartRent Manhattan</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #cbd5e1; margin-bottom: 0.5rem; font-size: 1rem;"><strong>ML-Powered Rental Price Prediction for Manhattan Apartments</strong></p>', unsafe_allow_html=True)
if 'prediction' not in st.session_state:
    st.markdown('\n    <div class="info-box">\n        Fill out the form below and click "Predict Rent" to see your prediction.\n    </div>\n    ', unsafe_allow_html=True)
st.markdown('---')
st.markdown('### Property Details')
st.markdown('#### Basic Information')
with st.form('input_form'):
    (col1, col2) = st.columns(2)
    with col1:
        bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=st.session_state.inputs['bedrooms'], key='form_bedrooms')
    with col2:
        size_sqft = st.number_input('Size (sqft)', min_value=200, max_value=3500, value=st.session_state.inputs['size_sqft'], key='form_size_sqft')
    bathrooms = st.number_input('Bathrooms', min_value=1, max_value=5, value=st.session_state.inputs['bathrooms'], key='form_bathrooms')
    st.markdown('---')
    st.markdown('#### Location & Building')
    (col1, col2) = st.columns(2)
    with col1:
        neighborhood_idx = neighborhoods_with_placeholder.index(st.session_state.inputs['neighborhood']) if st.session_state.inputs['neighborhood'] in neighborhoods_with_placeholder else 0
        neighborhood = st.selectbox('Neighborhood', neighborhoods_with_placeholder, index=neighborhood_idx, key='form_neighborhood')
        min_to_subway = st.number_input('Minutes to Subway', min_value=0, max_value=30, value=st.session_state.inputs['min_to_subway'], key='form_min_to_subway')
    with col2:
        floor = st.number_input('Floor', min_value=0, max_value=60, value=st.session_state.inputs['floor'], key='form_floor')
        building_age_yrs = st.number_input('Building Age (years)', min_value=0, max_value=150, value=st.session_state.inputs['building_age_yrs'], key='form_building_age_yrs')
    st.markdown('---')
    st.markdown('#### Amenities')
    (amenity_col1, amenity_col2) = st.columns(2)
    with amenity_col1:
        no_fee = st.checkbox('No Fee', value=st.session_state.inputs['no_fee'], key='form_no_fee')
        has_roofdeck = st.checkbox('Has Roofdeck', value=st.session_state.inputs['has_roofdeck'], key='form_has_roofdeck')
        has_washer_dryer = st.checkbox('Has Washer/Dryer', value=st.session_state.inputs['has_washer_dryer'], key='form_has_washer_dryer')
        has_doorman = st.checkbox('Has Doorman', value=st.session_state.inputs['has_doorman'], key='form_has_doorman')
    with amenity_col2:
        has_elevator = st.checkbox('Has Elevator', value=st.session_state.inputs['has_elevator'], key='form_has_elevator')
        has_dishwasher = st.checkbox('Has Dishwasher', value=st.session_state.inputs['has_dishwasher'], key='form_has_dishwasher')
        has_patio = st.checkbox('Has Patio', value=st.session_state.inputs['has_patio'], key='form_has_patio')
        has_gym = st.checkbox('Has Gym', value=st.session_state.inputs['has_gym'], key='form_has_gym')
    predict_button = st.form_submit_button('Predict Rent', type='primary')
if predict_button:
    if neighborhood == 'Select Neighborhood...' or neighborhood not in neighborhoods:
        st.error('Please select a neighborhood.')
        st.stop()
    with st.spinner('Calculating rent prediction...'):
        neighborhood_col = f'neighborhood_{neighborhood}'
        neighborhood_data = df_preprocessed[df_preprocessed[neighborhood_col] == 1]
        if neighborhood_data.empty:
            avg_rent_per_sqft = float(df_preprocessed['rent_per_sqft'].mean())
            avg_log_rent = float(df_preprocessed['log_rent'].mean())
            st.session_state.stats = {'avg_rent': float(df_preprocessed['rent'].mean()), 'median_rent': float(df_preprocessed['rent'].median()), 'rent_per_sqft': float(df_preprocessed['rent_per_sqft'].mean())}
        else:
            avg_rent_per_sqft = float(neighborhood_data['rent_per_sqft'].mean())
            avg_log_rent = float(neighborhood_data['log_rent'].mean())
            st.session_state.stats = {'avg_rent': float(neighborhood_data['rent'].mean()), 'median_rent': float(neighborhood_data['rent'].median()), 'rent_per_sqft': float(neighborhood_data['rent_per_sqft'].mean())}
        input_data = {'bedrooms': bedrooms, 'bathrooms': bathrooms, 'size_sqft': size_sqft, 'min_to_subway': min_to_subway, 'floor': floor, 'building_age_yrs': building_age_yrs, 'no_fee': int(no_fee), 'has_roofdeck': int(has_roofdeck), 'has_washer_dryer': int(has_washer_dryer), 'has_doorman': int(has_doorman), 'has_elevator': int(has_elevator), 'has_dishwasher': int(has_dishwasher), 'has_patio': int(has_patio), 'has_gym': int(has_gym), 'neighborhood': neighborhood}
        for n in neighborhoods:
            input_data[f'neighborhood_{n}'] = 1 if n == neighborhood else 0
        input_data['log_size_sqft'] = np.log1p(size_sqft)
        input_data['rent_per_sqft'] = avg_rent_per_sqft
        input_data['log_rent'] = avg_log_rent
        input_df = pd.DataFrame([input_data])
        if hasattr(model, 'feature_names_in_'):
            feature_order = list(model.feature_names_in_)
            for feature in feature_order:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[feature_order]
        else:
            if 'rent_per_sqft' not in input_df.columns:
                input_df['rent_per_sqft'] = avg_rent_per_sqft
            if 'log_rent' not in input_df.columns:
                input_df['log_rent'] = avg_log_rent
        prediction = model.predict(input_df)
        st.session_state.prediction = float(prediction[0])
        st.session_state.inputs = input_data
    st.success('Prediction calculated successfully!')
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
if 'prediction' in st.session_state:
    st.markdown('---')
    diff_avg = (st.session_state.prediction - st.session_state.stats['avg_rent']) / st.session_state.stats['avg_rent'] * 100
    diff_median = (st.session_state.prediction - st.session_state.stats['median_rent']) / st.session_state.stats['median_rent'] * 100
    for (affordability_threshold, (label, color)) in AFFORDABILITY_CRITERIA.items():
        if diff_avg <= affordability_threshold:
            affordability_label = label
            affordability_color = color
            break
    else:
        affordability_label = 'Luxury'
        affordability_color = '#ef4444'
    (col1, col2, col3) = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'\n        <div class="prediction-card">\n            <h2 style="color: #ffffff; margin-bottom: 0.25rem; font-size: 1.1rem; font-weight: 600;">Predicted Monthly Rent</h2>\n            <h1 style="color: #ffffff !important; font-size: 3rem; margin: 0; font-weight: 900; text-shadow: 0 2px 8px rgba(0,0,0,0.9), 0 0 4px rgba(0,0,0,0.8); letter-spacing: 1px; -webkit-text-fill-color: #ffffff !important;">${st.session_state.prediction:,.2f}</h1>\n            <p style="color: #ffffff; margin-top: 0.5rem; font-size: 1rem; font-weight: 600; background: {affordability_color}; padding: 0.4rem 1rem; border-radius: 8px; display: inline-block; margin-bottom: 0.25rem;">{affordability_label}</p>\n            <p style="color: #ffffff; margin-top: 0.25rem; font-size: 0.85rem;">Based on your property details</p>\n        </div>\n        ', unsafe_allow_html=True)
    (col1, col2, col3) = st.columns(3)
    with col1:
        st.metric('Neighborhood Average Rent', f"${st.session_state.stats['avg_rent']:,.2f}")
        st.caption(f"{abs(diff_avg):.1f}% {('below' if diff_avg < 0 else 'above')} average")
    with col2:
        st.metric('Neighborhood Median Rent', f"${st.session_state.stats['median_rent']:,.2f}")
        st.caption(f"{abs(diff_median):.1f}% {('below' if diff_median < 0 else 'above')} median")
    with col3:
        st.metric('Average Rent per sqft', f"${st.session_state.stats['rent_per_sqft']:.2f}")
