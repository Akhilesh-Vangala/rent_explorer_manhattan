import time
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from src.config.constants import DATA_CONSTANTS, MODEL_CONSTANTS
from src.config.config import Config

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")

def train_test_split_df(df: pd.DataFrame, target_column: str=DATA_CONSTANTS.TARGET_COLUMN) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print(f'\n[STEP 2] Train/Validation/Test Split ({100 - (MODEL_CONSTANTS.TEST_SIZE + MODEL_CONSTANTS.VAL_SIZE) * 100}%/{MODEL_CONSTANTS.VAL_SIZE * 100}%/{MODEL_CONSTANTS.TEST_SIZE * 100}%)...')
    X = df.drop(columns=[target_column])
    y = df[target_column]
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=MODEL_CONSTANTS.TEST_SIZE, random_state=MODEL_CONSTANTS.RANDOM_STATE)
    val_size = MODEL_CONSTANTS.VAL_SIZE / (1 - MODEL_CONSTANTS.TEST_SIZE)
    (X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=val_size, random_state=MODEL_CONSTANTS.RANDOM_STATE)
    return (X_train, X_val, X_test, y_train, y_val, y_test)

def train_model(model_name, X_train, y_train, X_val=None, y_val=None):
    
    start_time = time.time()
    loss_history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [], 'iterations': []}
    
    if model_name == 'linear_regression':
        print('\n   [3a] Linear Regression...')
        model = LinearRegression(**MODEL_CONSTANTS.LINEAR_REGRESSION_PARAMS)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        loss_history['train_loss'].append(train_rmse)
        loss_history['train_rmse'].append(train_rmse)
        loss_history['iterations'].append(1)
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            loss_history['val_loss'].append(val_rmse)
            loss_history['val_rmse'].append(val_rmse)
        
    elif model_name == 'random_forest':
        print('\n   [3b] Random Forest...')
        
        rf_params = MODEL_CONSTANTS.RF_PARAMS.copy()
        n_estimators = rf_params.pop('n_estimators', 200)
        
        
        step_size = max(1, n_estimators // 20)  
        model = RandomForestRegressor(warm_start=True, n_estimators=step_size, **rf_params)
        model.fit(X_train, y_train)  
        
        
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        loss_history['train_loss'].append(train_rmse)
        loss_history['train_rmse'].append(train_rmse)
        loss_history['iterations'].append(step_size)
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            loss_history['val_loss'].append(val_rmse)
            loss_history['val_rmse'].append(val_rmse)
        
        for n_trees in range(step_size * 2, n_estimators + 1, step_size):
            model.n_estimators = n_trees
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            loss_history['train_loss'].append(train_rmse)
            loss_history['train_rmse'].append(train_rmse)
            loss_history['iterations'].append(n_trees)
            
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                loss_history['val_loss'].append(val_rmse)
                loss_history['val_rmse'].append(val_rmse)
        
        
        model.n_estimators = n_estimators
        model.fit(X_train, y_train)
        
    elif model_name == 'xgboost':
        print('\n   [3c] XGBoost (with early stopping)...')
        xgb_params = MODEL_CONSTANTS.XGB_PARAMS.copy()
        
        xgb_params['eval_metric'] = 'rmse'
        
        
        eval_set = [(X_train, y_train)]
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        
        if hasattr(model, 'evals_result_') and model.evals_result_ is not None:
            evals_result = model.evals_result_
            
            
            train_key = None
            val_key = None
            
            
            for key in evals_result.keys():
                if 'train' in key.lower() or '0' in key or key == 'validation_0':
                    train_key = key
                elif 'val' in key.lower() or '1' in key or key == 'validation_1':
                    val_key = key
            
            
            if train_key is None and len(evals_result) > 0:
                keys = list(evals_result.keys())
                train_key = keys[0]
                if len(keys) > 1:
                    val_key = keys[1]
            
            if train_key and 'rmse' in evals_result[train_key]:
                train_rmse = evals_result[train_key]['rmse']
                n_iterations = len(train_rmse)
                loss_history['iterations'] = list(range(1, n_iterations + 1))
                loss_history['train_rmse'] = train_rmse
                loss_history['train_loss'] = train_rmse
                
                if val_key and 'rmse' in evals_result[val_key]:
                    val_rmse = evals_result[val_key]['rmse']
                    loss_history['val_rmse'] = val_rmse
                    loss_history['val_loss'] = val_rmse
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
    training_time = time.time() - start_time
    print(f'      Model training time: {training_time:.4f} seconds')
    return (model, training_time, loss_history)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def cross_validate_model(model, X_train, y_train, cv=5, scoring='r2'):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    return scores

def get_cv_metrics(model, X_train, y_train, cv=5):
    print(f'\n[STEP 4] Cross-Validation ({cv}-fold)...')
    r2_scores = cross_validate_model(model, X_train, y_train, cv=cv, scoring='r2')
    rmse_scores = -cross_validate_model(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_validate_model(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
    return {'r2_mean': np.mean(r2_scores), 'r2_std': np.std(r2_scores), 'rmse_mean': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores), 'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores)}

def model_comparison_table(results):
    df = pd.DataFrame(results)
    df = df.T
    return df

def print_model_metrics(metrics):
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    test_metrics = metrics['test']
    print(f"      Training time: {metrics['time']:.4f} seconds")
    print(f"      Train R2: {train_metrics['r2']:.6f}, RMSE: ${train_metrics['rmse']:.2f}")
    print(f"      Val R2: {val_metrics['r2']:.6f}, RMSE: ${val_metrics['rmse']:.2f}")
    print(f"      Test R2: {test_metrics['r2']:.6f}, RMSE: ${test_metrics['rmse']:.2f}")


def plot_prediction_vs_actual(y_true, y_pred, title: str, ax=None, figsize=(8, 6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)
    
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Rent ($)', fontsize=11)
    ax.set_ylabel('Predicted Rent ($)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title: str, ax=None, figsize=(8, 6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Rent ($)', fontsize=11)
    ax.set_ylabel('Residuals ($)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_residual_distribution(y_true, y_pred, title: str, ax=None, figsize=(8, 6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    residuals = y_true - y_pred
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_model_comparison_metrics(all_metrics: Dict[str, Dict], figsize=(12, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    models = list(all_metrics.keys())
    
    
    train_r2 = [all_metrics[m]['train']['r2'] for m in models]
    val_r2 = [all_metrics[m]['val']['r2'] for m in models]
    test_r2 = [all_metrics[m]['test']['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    axes[0, 0].bar(x - width, train_r2, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x, val_r2, width, label='Validation', alpha=0.8)
    axes[0, 0].bar(x + width, test_r2, width, label='Test', alpha=0.8)
    axes[0, 0].set_xlabel('Model', fontsize=10)
    axes[0, 0].set_ylabel('R² Score', fontsize=10)
    axes[0, 0].set_title('R² Score Comparison', fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    
    train_rmse = [all_metrics[m]['train']['rmse'] for m in models]
    val_rmse = [all_metrics[m]['val']['rmse'] for m in models]
    test_rmse = [all_metrics[m]['test']['rmse'] for m in models]
    
    axes[0, 1].bar(x - width, train_rmse, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x, val_rmse, width, label='Validation', alpha=0.8)
    axes[0, 1].bar(x + width, test_rmse, width, label='Test', alpha=0.8)
    axes[0, 1].set_xlabel('Model', fontsize=10)
    axes[0, 1].set_ylabel('RMSE ($)', fontsize=10)
    axes[0, 1].set_title('RMSE Comparison', fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    
    train_mae = [all_metrics[m]['train']['mae'] for m in models]
    val_mae = [all_metrics[m]['val']['mae'] for m in models]
    test_mae = [all_metrics[m]['test']['mae'] for m in models]
    
    axes[1, 0].bar(x - width, train_mae, width, label='Train', alpha=0.8)
    axes[1, 0].bar(x, val_mae, width, label='Validation', alpha=0.8)
    axes[1, 0].bar(x + width, test_mae, width, label='Test', alpha=0.8)
    axes[1, 0].set_xlabel('Model', fontsize=10)
    axes[1, 0].set_ylabel('MAE ($)', fontsize=10)
    axes[1, 0].set_title('MAE Comparison', fontsize=11, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    
    training_times = [all_metrics[m]['time'] for m in models]
    axes[1, 1].bar(models, training_times, alpha=0.8, color='purple')
    axes[1, 1].set_xlabel('Model', fontsize=10)
    axes[1, 1].set_ylabel('Training Time (seconds)', fontsize=10)
    axes[1, 1].set_title('Training Time Comparison', fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=15, ha='right')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_feature_importance_tree(model, feature_names, model_name: str, top_n: int = 20, figsize=(10, 8)):
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importances[indices], alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


def plot_learning_curve_xgb(model, model_name: str, figsize=(10, 6)):
    if not hasattr(model, 'evals_result_') or model.evals_result_ is None:
        return None
    
    evals_result = model.evals_result_
    fig, ax = plt.subplots(figsize=figsize)
    
    for eval_set in evals_result.keys():
        for metric in evals_result[eval_set].keys():
            ax.plot(evals_result[eval_set][metric], label=f'{eval_set} - {metric}')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title(f'Learning Curve - {model_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_loss_curves(loss_history: Dict[str, list], model_name: str, figsize=(12, 5)):
    if not loss_history or not loss_history.get('iterations'):
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    iterations = loss_history['iterations']
    
    
    if loss_history.get('train_rmse'):
        axes[0].plot(iterations, loss_history['train_rmse'], 'b-', label='Train RMSE', linewidth=2, alpha=0.8)
    
    if loss_history.get('val_rmse'):
        axes[0].plot(iterations, loss_history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2, alpha=0.8)
    
    axes[0].set_xlabel('Iteration / Number of Trees', fontsize=11)
    axes[0].set_ylabel('RMSE ($)', fontsize=11)
    axes[0].set_title(f'Training & Validation Loss (RMSE) - {model_name}', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    
    if loss_history.get('train_loss'):
        axes[1].plot(iterations, loss_history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    
    if loss_history.get('val_loss'):
        axes[1].plot(iterations, loss_history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    axes[1].set_xlabel('Iteration / Number of Trees', fontsize=11)
    axes[1].set_ylabel('Loss (RMSE) ($)', fontsize=11)
    axes[1].set_title(f'Loss Curves - {model_name}', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cv_scores(cv_metrics: Dict[str, Any], model_name: str, figsize=(10, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics_to_plot = ['r2', 'rmse', 'mae']
    metric_labels = ['R² Score', 'RMSE ($)', 'MAE ($)']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        
        if mean_key in cv_metrics and std_key in cv_metrics:
            mean_val = cv_metrics[mean_key]
            std_val = cv_metrics[std_key]
            
            
            x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
            y = np.exp(-0.5 * ((x - mean_val) / std_val) ** 2) / (std_val * np.sqrt(2 * np.pi))
            
            axes[idx].plot(x, y, 'b-', lw=2, label=f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}')
            axes[idx].axvline(mean_val, color='r', linestyle='--', lw=2, label='Mean')
            axes[idx].fill_between(x, y, alpha=0.3)
            axes[idx].set_xlabel(label, fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
            axes[idx].set_title(f'CV {label} Distribution', fontsize=10, fontweight='bold')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(alpha=0.3)
    
    plt.suptitle(f'Cross-Validation Metrics - {model_name}', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def save_training_visualizations(
    models: Dict[str, Any],
    all_metrics: Dict[str, Dict],
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    cv_metrics: Optional[Dict[str, Any]] = None,
    loss_histories: Optional[Dict[str, Dict]] = None,
    output_dir: Optional[Path] = None
):
    if output_dir is None:
        output_dir = Config.OUTPUTS_DIR / 'training_plots'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n[STEP 9] Generating training visualizations...')
    print(f'   Saving plots to: {output_dir}')
    
    feature_names = X_train.columns.tolist()
    
    print('   Creating model comparison plots...')
    fig = plot_model_comparison_metrics(all_metrics)
    fig.savefig(output_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    for model_name, model in models.items():
        print(f'   Creating plots for {model_name}...')
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        fig = plot_prediction_vs_actual(y_train, y_train_pred, f'{model_name} - Train Set')
        fig.savefig(output_dir / f'{model_name}_pred_vs_actual_train.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        fig = plot_prediction_vs_actual(y_val, y_val_pred, f'{model_name} - Validation Set')
        fig.savefig(output_dir / f'{model_name}_pred_vs_actual_val.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        fig = plot_prediction_vs_actual(y_test, y_test_pred, f'{model_name} - Test Set')
        fig.savefig(output_dir / f'{model_name}_pred_vs_actual_test.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
        fig = plot_residuals(y_train, y_train_pred, f'{model_name} - Train Residuals')
        fig.savefig(output_dir / f'{model_name}_residuals_train.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        fig = plot_residuals(y_val, y_val_pred, f'{model_name} - Validation Residuals')
        fig.savefig(output_dir / f'{model_name}_residuals_val.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        fig = plot_residuals(y_test, y_test_pred, f'{model_name} - Test Residuals')
        fig.savefig(output_dir / f'{model_name}_residuals_test.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
        fig = plot_residual_distribution(y_test, y_test_pred, f'{model_name} - Test Residual Distribution')
        fig.savefig(output_dir / f'{model_name}_residual_distribution_test.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
        if hasattr(model, 'feature_importances_'):
            if model_name == 'xgboost':
                
                print(f'   Creating feature importance plot using XGBoost built-in function...')
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    xgb.plot_importance(model, ax=ax, importance_type='gain', max_num_features=20, 
                                       title=f'XGBoost Feature Importance (Gain) - {model_name}')
                    plt.tight_layout()
                    fig.savefig(output_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    xgb.plot_importance(model, ax=ax, importance_type='weight', max_num_features=20,
                                       title=f'XGBoost Feature Importance (Weight) - {model_name}')
                    plt.tight_layout()
                    fig.savefig(output_dir / f'{model_name}_feature_importance_weight.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f'   Warning: Could not use XGBoost built-in plot_importance: {e}')
                    
                    fig = plot_feature_importance_tree(model, feature_names, model_name)
                    if fig is not None:
                        fig.savefig(output_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
                        plt.close(fig)
            else:
                
                fig = plot_feature_importance_tree(model, feature_names, model_name)
                if fig is not None:
                    fig.savefig(output_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        
        if model_name == 'xgboost':
            
            if hasattr(model, 'evals_result_') and model.evals_result_ is not None:
                fig = plot_learning_curve_xgb(model, model_name)
                if fig is not None:
                    fig.savefig(output_dir / f'{model_name}_learning_curve.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            
            print(f'   Creating tree visualizations using XGBoost built-in function...')
            try:
                
                for tree_idx in range(min(3, model.get_booster().num_boosted_rounds())):
                    fig, ax = plt.subplots(figsize=(20, 10))
                    xgb.plot_tree(model, num_trees=tree_idx, ax=ax, rankdir='LR')
                    plt.title(f'XGBoost Tree {tree_idx}', fontsize=14, fontweight='bold', pad=20)
                    plt.tight_layout()
                    fig.savefig(output_dir / f'{model_name}_tree_{tree_idx}.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                print(f'   Warning: Could not create tree plots (graphviz may not be installed): {e}')
                print(f'   Install graphviz with: pip install graphviz (and system package)')
        
        
        if loss_histories and model_name in loss_histories:
            loss_history = loss_histories[model_name]
            if loss_history and loss_history.get('iterations'):
                print(f'   Creating loss curves for {model_name}...')
                fig = plot_loss_curves(loss_history, model_name)
                if fig is not None:
                    fig.savefig(output_dir / f'{model_name}_loss_curves.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
    
    
    if cv_metrics is not None:
        print('   Creating cross-validation plots...')
        best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['test']['r2'])
        fig = plot_cv_scores(cv_metrics, best_model_name)
        fig.savefig(output_dir / 'cross_validation_scores.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    
    print('   Creating comprehensive summary plot...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['test']['r2'])
    best_model = models[best_model_name]
    y_test_pred_best = best_model.predict(X_test)
    
    
    plot_prediction_vs_actual(y_test, y_test_pred_best, f'{best_model_name} - Test Set (Best Model)', ax=axes[0, 0])
    
    
    plot_residuals(y_test, y_test_pred_best, f'{best_model_name} - Test Residuals (Best Model)', ax=axes[0, 1])
    
    
    models_list = list(all_metrics.keys())
    test_r2_scores = [all_metrics[m]['test']['r2'] for m in models_list]
    axes[1, 0].bar(models_list, test_r2_scores, alpha=0.8, color='green')
    axes[1, 0].set_ylabel('Test R² Score', fontsize=11)
    axes[1, 0].set_title('Test R² Score Comparison', fontsize=11, fontweight='bold')
    axes[1, 0].set_xticks(range(len(models_list)))
    axes[1, 0].set_xticklabels(models_list, rotation=15, ha='right')
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, (model, score) in enumerate(zip(models_list, test_r2_scores)):
        axes[1, 0].text(i, score, f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    
    test_rmse_scores = [all_metrics[m]['test']['rmse'] for m in models_list]
    axes[1, 1].bar(models_list, test_rmse_scores, alpha=0.8, color='red')
    axes[1, 1].set_ylabel('Test RMSE ($)', fontsize=11)
    axes[1, 1].set_title('Test RMSE Comparison', fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(range(len(models_list)))
    axes[1, 1].set_xticklabels(models_list, rotation=15, ha='right')
    axes[1, 1].grid(alpha=0.3, axis='y')
    for i, (model, score) in enumerate(zip(models_list, test_rmse_scores)):
        axes[1, 1].text(i, score, f'${score:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Training Summary - Model Performance Overview', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f'   ✓ All visualizations saved to {output_dir}')
    print(f'   ✓ Generated {len(list(output_dir.glob("*.png")))} plot files')
