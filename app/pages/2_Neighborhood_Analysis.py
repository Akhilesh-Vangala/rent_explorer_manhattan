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
from src.visualization import plot_neighborhood_comparison
from src.utils import load_data, get_neighborhoods, get_neighborhood_data
from app.shared_styles import apply_shared_styles
apply_shared_styles()
(df_raw, df_preprocessed) = load_data()
neighborhoods = get_neighborhoods(df_preprocessed)
st.title('Neighborhood Analysis')
st.markdown('---')
neighborhood_stats = []
for n in neighborhoods:
    n_data = get_neighborhood_data(df_preprocessed, n)
    if len(n_data) > 0:
        rent_values = n_data['rent'].values
        neighborhood_stats.append({'Neighborhood': n, 'Average Rent': rent_values.mean(), 'Median Rent': np.median(rent_values), 'Min Rent': rent_values.min(), 'Max Rent': rent_values.max(), 'Pct25': np.percentile(rent_values, 25), 'Pct75': np.percentile(rent_values, 75), 'Count': len(n_data)})
neighborhood_df = pd.DataFrame(neighborhood_stats)
neighborhood_df = neighborhood_df.sort_values('Average Rent', ascending=False)
st.subheader('Neighborhood Statistics Summary')
display_df = neighborhood_df.copy()
display_df['Average Rent'] = display_df['Average Rent'].round(2)
display_df['Median Rent'] = display_df['Median Rent'].round(2)
display_df['Min Rent'] = display_df['Min Rent'].round(2)
display_df['Max Rent'] = display_df['Max Rent'].round(2)
display_df['Pct25'] = display_df['Pct25'].round(2)
display_df['Pct75'] = display_df['Pct75'].round(2)
display_df['Count'] = display_df['Count'].astype(int)
st.dataframe(display_df, use_container_width=True, hide_index=True)
st.markdown('---')
st.subheader('Neighborhood Comparison Chart')
neighborhood_df_viz = neighborhood_df.rename(columns={'Average Rent': 'avg_rent', 'Neighborhood': 'neighborhood'})
fig = plot_neighborhood_comparison(neighborhood_df_viz, top_n=20, ascending=False, figsize=(9, 8))
st.pyplot(fig)
plt.close()
st.markdown('---')
if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')
