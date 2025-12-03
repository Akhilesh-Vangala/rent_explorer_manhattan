import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from pathlib import Path
code_dir = Path(__file__).resolve().parent.parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
from app.path_setup import setup_paths
setup_paths()
from src.visualization import plot_model_comparison_rmse_mae, plot_model_comparison_r2, plot_feature_importance
from src.utils import load_data, load_model
from app.shared_styles import apply_shared_styles
apply_shared_styles()
model = load_model()
(df_raw, df_preprocessed) = load_data(train=True)
st.title('Model Performance')
st.markdown('---')
from src.modeling import train_test_split_df
(X_train, X_val, X_test, y_train, y_val, y_test) = train_test_split_df(df_preprocessed, 'rent')
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader('Performance Metrics')
(col1, col2, col3) = st.columns(3)
with col1:
    st.metric('RMSE', f'${rmse:,.2f}')
with col2:
    st.metric('MAE', f'${mae:,.2f}')
with col3:
    st.metric('R² Score', f'{r2:.4f}')
st.markdown('---')
st.subheader('Model Comparison')
from src.modeling import train_model, evaluate_model
model_results = []
for (model_name, display_name) in [('linear_regression', 'Linear Regression'), ('random_forest', 'Random Forest'), ('xgboost', 'XGBoost')]:
    (trained_model, _) = train_model(model_name, X_train, y_train)
    test_metrics = evaluate_model(trained_model, X_test, y_test)
    model_results.append({'Model': display_name, 'RMSE': test_metrics['rmse'], 'MAE': test_metrics['mae'], 'R²': test_metrics['r2']})
model_comparison = pd.DataFrame(model_results)
st.dataframe(model_comparison, use_container_width=True, hide_index=True)
st.markdown('---')
(col1, col2) = st.columns(2)
with col1:
    st.markdown('#### RMSE and MAE Comparison')
    fig = plot_model_comparison_rmse_mae(model_comparison)
    st.pyplot(fig)
    plt.close()
with col2:
    st.markdown('#### R² Score Comparison')
    fig = plot_model_comparison_r2(model_comparison, ylim=(0.75, 0.85))
    st.pyplot(fig)
    plt.close()
st.markdown('---')
st.subheader('XGBoost Feature Importance')
feature_importance = model.feature_importances_
feature_names = model.feature_names_in_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
fig = plot_feature_importance(importance_df, top_n=20, exclude_features=['log_rent', 'rent_per_sqft'], figsize=(9, 8))
st.pyplot(fig)
plt.close()
st.markdown('---')
if st.button('Back to Home Page'):
    st.switch_page('Rent_Estimation.py')
