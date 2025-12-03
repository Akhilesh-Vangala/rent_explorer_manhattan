import sys
import os
from pathlib import Path
current_file = Path(__file__).resolve()
base_dir = current_file.parent.parent
base_dir_str = str(base_dir)
if base_dir_str not in sys.path:
    sys.path.insert(0, base_dir_str)
import pandas as pd
import joblib
from src.config.config import Config
from src.utils import load_data
from src.modeling import train_test_split_df, train_model, evaluate_model, get_cv_metrics, print_model_metrics, save_training_visualizations

def main():
    print('=' * 70)
    print('RETRAINING FINAL MODEL - PROFESSIONAL STANDARDS')
    print('=' * 70)
    (df_raw, df_preprocessed) = load_data(train=True)
    print(f'   Raw data shape: {df_raw.shape}')
    print(f'   Preprocessed data shape: {df_preprocessed.shape}')
    print(f'   Features: {df_preprocessed.shape[1] - 1}')
    (X_train, X_val, X_test, y_train, y_val, y_test) = train_test_split_df(df_preprocessed)
    print(f'   Train: {len(X_train)} samples')
    print(f'   Validation: {len(X_val)} samples')
    print(f'   Test: {len(X_test)} samples')
    print('\n[STEP 3] Training models...')
    (lr_model, lr_time, lr_loss_history) = train_model('linear_regression', X_train, y_train, X_val, y_val)
    lr_metrics = {'train': evaluate_model(lr_model, X_train, y_train), 'val': evaluate_model(lr_model, X_val, y_val), 'test': evaluate_model(lr_model, X_test, y_test), 'time': lr_time}
    print_model_metrics(lr_metrics)
    (rf_model, rf_time, rf_loss_history) = train_model('random_forest', X_train, y_train, X_val, y_val)
    rf_metrics = {'train': evaluate_model(rf_model, X_train, y_train), 'val': evaluate_model(rf_model, X_val, y_val), 'test': evaluate_model(rf_model, X_test, y_test), 'time': rf_time}
    print_model_metrics(rf_metrics)
    (xgb_model, xgb_time, xgb_loss_history) = train_model('xgboost', X_train, y_train, X_val, y_val)
    xgb_metrics = {'train': evaluate_model(xgb_model, X_train, y_train), 'val': evaluate_model(xgb_model, X_val, y_val), 'test': evaluate_model(xgb_model, X_test, y_test), 'time': xgb_time}
    print_model_metrics(xgb_metrics)
    if hasattr(xgb_model, 'best_iteration'):
        print(f'      Best iteration: {xgb_model.best_iteration}')
    xgb_cv_metrics = get_cv_metrics(xgb_model, X_train, y_train, cv=5)
    print(f"   CV R2: {xgb_cv_metrics['r2_mean']:.6f} +/- {xgb_cv_metrics['r2_std']:.6f}")
    print(f"   CV RMSE: ${xgb_cv_metrics['rmse_mean']:.2f} +/- ${xgb_cv_metrics['rmse_std']:.2f}")
    print(f"   CV MAE: ${xgb_cv_metrics['mae_mean']:.2f} +/- ${xgb_cv_metrics['mae_std']:.2f}")
    print('\n[STEP 5] Overfitting Analysis...')
    xgb_train_test_gap = xgb_metrics['train']['r2'] - xgb_metrics['test']['r2']
    xgb_train_val_gap = xgb_metrics['train']['r2'] - xgb_metrics['val']['r2']
    print(f'   Train-Test R2 gap: {xgb_train_test_gap:.6f} ({xgb_train_test_gap * 100:.2f}%)')
    print(f'   Train-Val R2 gap: {xgb_train_val_gap:.6f} ({xgb_train_val_gap * 100:.2f}%)')
    if xgb_train_test_gap < 0.05:
        print('   Status: Minimal overfitting - Excellent generalization!')
    elif xgb_train_test_gap < 0.1:
        print('   Status: Acceptable overfitting - Good performance')
    else:
        print('   Status: Moderate overfitting - May need further regularization')
    print('\n[STEP 6] Saving best model (XGBoost)...')
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(xgb_model, Config.BEST_MODEL_FILE)
    print(f'   Model saved to {Config.BEST_MODEL_FILE}')
    with open(Config.FEATURE_NAMES_FILE, 'w') as f:
        for col in X_train.columns:
            f.write(f'{col}\n')
    print(f'   Feature names saved to {Config.FEATURE_NAMES_FILE}')
    print('\n[STEP 7] Saving metrics...')
    with open(Config.MODEL_METRICS_FILE, 'w') as f:
        f.write(f'Model: XGBoost\n')
        f.write(f"Train R2: {xgb_metrics['train']['r2']:.6f}\n")
        f.write(f"Train RMSE: {xgb_metrics['train']['rmse']:.2f}\n")
        f.write(f"Train MAE: {xgb_metrics['train']['mae']:.2f}\n")
        f.write(f"Validation R2: {xgb_metrics['val']['r2']:.6f}\n")
        f.write(f"Validation RMSE: {xgb_metrics['val']['rmse']:.2f}\n")
        f.write(f"Validation MAE: {xgb_metrics['val']['mae']:.2f}\n")
        f.write(f"Test R2: {xgb_metrics['test']['r2']:.6f}\n")
        f.write(f"Test RMSE: {xgb_metrics['test']['rmse']:.2f}\n")
        f.write(f"Test MAE: {xgb_metrics['test']['mae']:.2f}\n")
        f.write(f'Train-Test R2 Gap: {xgb_train_test_gap:.6f}\n')
        f.write(f"CV R2 Mean: {xgb_cv_metrics['r2_mean']:.6f}\n")
        f.write(f"CV R2 Std: {xgb_cv_metrics['r2_std']:.6f}\n")
        f.write(f"CV RMSE Mean: {xgb_cv_metrics['rmse_mean']:.2f}\n")
        f.write(f"CV RMSE Std: {xgb_cv_metrics['rmse_std']:.2f}\n")
        f.write(f"CV MAE Mean: {xgb_cv_metrics['mae_mean']:.2f}\n")
        f.write(f"CV MAE Std: {xgb_cv_metrics['mae_std']:.2f}\n")
    with open(Config.CV_METRICS_FILE, 'w') as f:
        f.write(f"CV_R2_MEAN: {xgb_cv_metrics['r2_mean']:.6f}\n")
        f.write(f"CV_R2_STD: {xgb_cv_metrics['r2_std']:.6f}\n")
        f.write(f"CV_RMSE_MEAN: {xgb_cv_metrics['rmse_mean']:.2f}\n")
        f.write(f"CV_RMSE_STD: {xgb_cv_metrics['rmse_std']:.2f}\n")
        f.write(f"CV_MAE_MEAN: {xgb_cv_metrics['mae_mean']:.2f}\n")
        f.write(f"CV_MAE_STD: {xgb_cv_metrics['mae_std']:.2f}\n")
    print(f'   Metrics saved to {Config.MODEL_METRICS_FILE} and {Config.CV_METRICS_FILE}')
    print('\n[STEP 8] Model Comparison Summary...')
    print('\n' + '=' * 70)
    print('FINAL MODEL COMPARISON')
    print('=' * 70)
    print(f"\n{'Model':<20} {'Train R2':<12} {'Val R2':<12} {'Test R2':<12} {'Test RMSE':<12}")
    print('-' * 70)
    print(f"{'Linear Regression':<20} {lr_metrics['train']['r2']:<12.6f} {lr_metrics['val']['r2']:<12.6f} {lr_metrics['test']['r2']:<12.6f} ${lr_metrics['test']['rmse']:<11.2f}")
    print(f"{'Random Forest':<20} {rf_metrics['train']['r2']:<12.6f} {rf_metrics['val']['r2']:<12.6f} {rf_metrics['test']['r2']:<12.6f} ${rf_metrics['test']['rmse']:<11.2f}")
    print(f"{'XGBoost (Best)':<20} {xgb_metrics['train']['r2']:<12.6f} {xgb_metrics['val']['r2']:<12.6f} {xgb_metrics['test']['r2']:<12.6f} ${xgb_metrics['test']['rmse']:<11.2f}")
    print('=' * 70)
    print('\n[SUCCESS] Model retraining complete!')
    print(f'\nFinal XGBoost Performance:')
    print(f"   Test R2: {xgb_metrics['test']['r2']:.6f} ({xgb_metrics['test']['r2'] * 100:.2f}%)")
    print(f"   Test RMSE: ${xgb_metrics['test']['rmse']:.2f}")
    print(f"   Test MAE: ${xgb_metrics['test']['mae']:.2f}")
    print(f'   Overfitting gap: {xgb_train_test_gap * 100:.2f}%')
    
    save_training_visualizations(
        models={'linear_regression': lr_model, 'random_forest': rf_model, 'xgboost': xgb_model},
        all_metrics={'linear_regression': lr_metrics, 'random_forest': rf_metrics, 'xgboost': xgb_metrics},
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        cv_metrics=xgb_cv_metrics,
        loss_histories={'linear_regression': lr_loss_history, 'random_forest': rf_loss_history, 'xgboost': xgb_loss_history}
    )
if __name__ == '__main__':
    main()
