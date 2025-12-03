import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

def plot_rent_comparison(avg_rent: float, predicted_rent: float, neighborhood: str, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    bars = ax.bar(['Average Rent', 'Predicted Rent'], [avg_rent, predicted_rent], color=['#3498db', '#2ecc71'])
    ax.set_ylabel('Rent ($)', fontsize=12)
    ax.set_title(f'Rent Comparison for {neighborhood}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for (i, (bar, val)) in enumerate(zip(bars, [avg_rent, predicted_rent])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'${val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_size_vs_rent(neighborhood_data: pd.DataFrame, size_sqft: float, predicted_rent: float, neighborhood: str, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.scatter(neighborhood_data['size_sqft'], neighborhood_data['rent'], alpha=0.5, s=50, color='#3498db', label='Neighborhood Listings')
    ax.scatter(size_sqft, predicted_rent, s=300, color='#e74c3c', marker='*', edgecolors='black', linewidths=2, label='Your Prediction', zorder=5)
    ax.set_xlabel('Size (sqft)', fontsize=12)
    ax.set_ylabel('Rent ($)', fontsize=12)
    ax.set_title(f'Size vs Rent: {neighborhood} with Your Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_rent_distribution(neighborhood_data: pd.DataFrame, predicted_rent: Optional[float]=None, avg_rent: Optional[float]=None, neighborhood: Optional[str]=None, bins: int=30, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.hist(neighborhood_data['rent'], bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    if predicted_rent is not None:
        ax.axvline(predicted_rent, color='#e74c3c', linestyle='--', linewidth=2, label=f'Your Prediction: ${predicted_rent:,.0f}')
    if avg_rent is not None:
        ax.axvline(avg_rent, color='#2ecc71', linestyle='--', linewidth=2, label=f'Average: ${avg_rent:,.0f}')
    ax.set_xlabel('Rent ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    title = f'Rent Distribution in {neighborhood}' if neighborhood else 'Rent Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold')
    if predicted_rent is not None or avg_rent is not None:
        ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_neighborhood_comparison(neighborhood_df: pd.DataFrame, top_n: int=10, highlight_neighborhood: Optional[str]=None, ascending: bool=False, figsize: Tuple[int, int]=(9, 8)) -> plt.Figure:
    sorted_df = neighborhood_df.sort_values('avg_rent', ascending=ascending).head(top_n)
    (fig, ax) = plt.subplots(figsize=figsize)
    if highlight_neighborhood:
        colors = ['#e74c3c' if n == highlight_neighborhood else '#3498db' for n in sorted_df['neighborhood']]
    else:
        colors = '#3498db'
    bars = ax.barh(range(len(sorted_df)), sorted_df['avg_rent'], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['neighborhood'], fontsize=10)
    ax.set_xlabel('Average Rent ($)', fontsize=12)
    ax.set_title(f'Top {top_n} Neighborhoods by Average Rent', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_amenity_impact(amenity_df: pd.DataFrame, neighborhood: str, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    sorted_df = amenity_df.sort_values('impact', ascending=True)
    (fig, ax) = plt.subplots(figsize=figsize)
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in sorted_df['impact']]
    bars = ax.barh(range(len(sorted_df)), sorted_df['impact'], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['amenity'], fontsize=10)
    ax.set_xlabel('Rent Impact ($)', fontsize=12)
    ax.set_title(f'Amenity Impact on Rent in {neighborhood}', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_bedroom_rent_distribution(bedroom_data: pd.DataFrame, predicted_rent: float, bedrooms: int, neighborhood: str, bins: int=20, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.hist(bedroom_data['rent'], bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    avg_rent = bedroom_data['rent'].mean()
    ax.axvline(predicted_rent, color='#e74c3c', linestyle='--', linewidth=2, label=f'Your Prediction: ${predicted_rent:,.0f}')
    ax.axvline(avg_rent, color='#2ecc71', linestyle='--', linewidth=2, label=f'Average ({bedrooms} BR): ${avg_rent:,.0f}')
    ax.set_xlabel('Rent ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Rent Distribution for {bedrooms}-Bedroom Units in {neighborhood}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_feature_values(features: Dict[str, float], figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    feature_names = list(features.keys())
    feature_values = list(features.values())
    (fig, ax) = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(feature_names)), feature_values, color='#3498db')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel('Feature Value', fontsize=12)
    ax.set_title('Your Property Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int=20, exclude_features: Optional[List[str]]=None, figsize: Tuple[int, int]=(9, 8)) -> plt.Figure:
    if exclude_features:
        importance_df = importance_df[~importance_df['Feature'].isin(exclude_features)]
    sorted_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    (fig, ax) = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(sorted_df)), sorted_df['Importance'], color='#3498db')
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['Feature'], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_model_comparison_rmse_mae(model_comparison_df: pd.DataFrame, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    x = np.arange(len(model_comparison_df))
    width = 0.35
    bars1 = ax.bar(x - width / 2, model_comparison_df['RMSE'], width, label='RMSE', color='#e74c3c')
    bars2 = ax.bar(x + width / 2, model_comparison_df['MAE'], width, label='MAE', color='#3498db')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error ($)', fontsize=12)
    ax.set_title('RMSE and MAE by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_comparison_df['Model'], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_model_comparison_r2(model_comparison_df: pd.DataFrame, ylim: Optional[Tuple[float, float]]=None, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    bars = ax.bar(model_comparison_df['Model'], model_comparison_df['R²'], color='#2ecc71')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score by Model', fontsize=14, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig

def plot_global_rent_distribution(rent_data: pd.Series, bins: int=50, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.hist(rent_data, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Rent ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Monthly Rent Across All Manhattan Listings', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_size_vs_rent_scatter(size_data: pd.Series, rent_data: pd.Series, title: str='Size vs Rent: All Manhattan Listings', alpha: float=0.4, s: int=20, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.scatter(size_data, rent_data, alpha=alpha, s=s, color='#3498db')
    ax.set_xlabel('Size (sqft)', fontsize=12)
    ax.set_ylabel('Rent ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_bedroom_distribution(bedroom_counts: pd.Series, figsize: Tuple[int, int]=(9, 5)) -> plt.Figure:
    (fig, ax) = plt.subplots(figsize=figsize)
    bars = ax.bar(bedroom_counts.index, bedroom_counts.values, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Bedrooms', fontsize=12)
    ax.set_ylabel('Number of Listings', fontsize=12)
    ax.set_title('Distribution of Bedrooms Across All Listings', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig
