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
from src.visualization import plot_rent_comparison, plot_size_vs_rent, plot_rent_distribution, plot_neighborhood_comparison, plot_amenity_impact, plot_bedroom_rent_distribution, plot_feature_values
from src.utils import load_data, get_neighborhoods, get_neighborhood_data
from app.shared_styles import apply_shared_styles
apply_shared_styles()
(df_raw, df_preprocessed) = load_data()
neighborhoods = get_neighborhoods(df_preprocessed)
if 'prediction' not in st.session_state or st.session_state.prediction is None:
    st.title('Detailed Analysis')
    st.warning('No prediction found. Please return to the Home page and make a prediction first.')
    if st.button('Go to Home Page'):
        st.switch_page('Rent_Estimation.py')
    st.stop()
st.title('Detailed Analysis')
predicted_rent = st.session_state.prediction
inputs = st.session_state.inputs
neighborhood = inputs['neighborhood']
stats = st.session_state.stats
st.markdown(f'**Predicted Rent:** ${predicted_rent:,.2f} | **Neighborhood:** {neighborhood}')
st.markdown('---')
st.subheader('Rent Comparison')
neighborhood_data = get_neighborhood_data(df_preprocessed, neighborhood)
avg_rent = stats['avg_rent']
fig = plot_rent_comparison(avg_rent, predicted_rent, neighborhood)
st.pyplot(fig)
plt.close()
st.subheader('Size vs Rent Analysis')
fig = plot_size_vs_rent(neighborhood_data, inputs['size_sqft'], predicted_rent, neighborhood)
st.pyplot(fig)
plt.close()
st.subheader('Neighborhood Rent Distribution')
fig = plot_rent_distribution(neighborhood_data, predicted_rent, avg_rent, neighborhood, bins=30)
st.pyplot(fig)
plt.close()
st.subheader('Similar Neighborhood Comparison')
all_neighborhood_avgs = []
for n in neighborhoods:
    try:
        n_data = get_neighborhood_data(df_preprocessed, n)
        if len(n_data) > 0:
            all_neighborhood_avgs.append({'neighborhood': n, 'avg_rent': n_data['rent'].mean()})
    except ValueError:
        continue
neighborhood_df = pd.DataFrame(all_neighborhood_avgs)
fig = plot_neighborhood_comparison(neighborhood_df, top_n=10, highlight_neighborhood=neighborhood, ascending=False, figsize=(9, 5))
st.pyplot(fig)
plt.close()
st.subheader('Amenity Impact Analysis')
amenities = ['no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']
amenity_impact = []
for amenity in amenities:
    with_amenity = neighborhood_data[neighborhood_data[amenity] == 1]['rent'].mean()
    without_amenity = neighborhood_data[neighborhood_data[amenity] == 0]['rent'].mean()
    impact = with_amenity - without_amenity
    amenity_impact.append({'amenity': amenity.replace('has_', '').replace('_', ' ').title(), 'impact': impact})
amenity_df = pd.DataFrame(amenity_impact)
fig = plot_amenity_impact(amenity_df, neighborhood)
st.pyplot(fig)
plt.close()
st.subheader('Bedroom-based Rent Distribution')
similar_bedrooms = neighborhood_data[neighborhood_data['bedrooms'] == inputs['bedrooms']]
if len(similar_bedrooms) > 0:
    fig = plot_bedroom_rent_distribution(similar_bedrooms, predicted_rent, inputs['bedrooms'], neighborhood)
    st.pyplot(fig)
    plt.close()
else:
    st.info(f"No data available for {inputs['bedrooms']}-bedroom units in {neighborhood}.")
st.subheader('Feature Contribution to Prediction')
features = {'Size (sqft)': inputs['size_sqft'], 'Bedrooms': inputs['bedrooms'], 'Bathrooms': inputs['bathrooms'], 'Floor': inputs['floor'], 'Building Age': inputs['building_age_yrs'], 'Minutes to Subway': inputs['min_to_subway']}
fig = plot_feature_values(features)
st.pyplot(fig)
plt.close()
st.markdown('---')
if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')
