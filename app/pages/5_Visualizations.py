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
from src.visualization import plot_global_rent_distribution, plot_size_vs_rent_scatter, plot_bedroom_distribution, plot_neighborhood_comparison
from src.utils import load_data, get_neighborhoods, get_neighborhood_data
from app.shared_styles import apply_shared_styles
apply_shared_styles()
(df_raw, df_preprocessed) = load_data()
neighborhoods = get_neighborhoods(df_preprocessed)
st.title('Visualizations')
st.markdown('**Exploratory Data Analysis of Manhattan Rental Listings**')
st.markdown('---')
st.subheader('Global Rent Distribution')
fig = plot_global_rent_distribution(df_raw['rent'], bins=50)
st.pyplot(fig)
plt.close()
(col1, col2, col3, col4) = st.columns(4)
with col1:
    st.metric('Mean Rent', f"${df_raw['rent'].mean():,.2f}")
with col2:
    st.metric('Median Rent', f"${df_raw['rent'].median():,.2f}")
with col3:
    st.metric('Min Rent', f"${df_raw['rent'].min():,.2f}")
with col4:
    st.metric('Max Rent', f"${df_raw['rent'].max():,.2f}")
st.markdown('---')
st.subheader('Size vs Rent Relationship')
fig = plot_size_vs_rent_scatter(df_raw['size_sqft'], df_raw['rent'])
st.pyplot(fig)
plt.close()
correlation = df_raw['size_sqft'].corr(df_raw['rent'])
st.metric('Correlation Coefficient', f'{correlation:.3f}')
st.markdown('---')
st.subheader('Bedroom Distribution')
bedroom_counts = df_raw['bedrooms'].value_counts().sort_index()
fig = plot_bedroom_distribution(bedroom_counts)
st.pyplot(fig)
plt.close()
st.markdown('---')
st.subheader('Neighborhood Average Rent Comparison')
neighborhood_avgs = []
for n in neighborhoods:
    n_data = get_neighborhood_data(df_preprocessed, n)
    if len(n_data) > 0:
        neighborhood_avgs.append({'neighborhood': n, 'avg_rent': n_data['rent'].mean(), 'count': len(n_data)})
neighborhood_df = pd.DataFrame(neighborhood_avgs)
fig = plot_neighborhood_comparison(neighborhood_df, top_n=20, ascending=True, figsize=(9, 8))
st.pyplot(fig)
plt.close()
st.markdown('---')
if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')
