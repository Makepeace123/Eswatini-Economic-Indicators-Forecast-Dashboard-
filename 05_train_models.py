# -*- coding: utf-8 -*-
"""05_train_models.py"""

import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam

# Load configuration
from02_configuration import TARGET_VARIABLES, TEST_SIZE, RANDOM_STATE, FORECAST_HORIZON

# Load processed data
df = pd.read_csv('data/processed_data.csv', index_col='Date', parse_dates=True)
print(f"Loaded processed data with shape: {df.shape}")

def ensemble_feature_selection(X, y, random_state=42):
    """Ensemble feature selection using Lasso and Random Forest"""
    # Lasso feature selection
    lasso = Lasso(alpha=0.01, random_state=random_state)
    lasso.fit(X, y)
    lasso_importance = np.abs(lasso.coef_)
    
    # Random Forest feature selection
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Combine importances (average of normalized importances)
    lasso_norm = lasso_importance / (np.max(lasso_importance) + 1e-10)
    rf_norm = rf_importance / (np.max(rf_importance) + 1e-10)
    combined_importance = (lasso_norm + rf_norm) / 2
    
    return combined_importance

def train_for_variable(target_variable, data):
    """Train all models for a single target variable"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS FOR: {target_variable}")
    print(f"{'='*60}")
    
    # Prepare data
    features = [col for col in data.columns if col != target_variable]
    X = data[features]
    y = data[target_variable]
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['day_of_week', 'month', 'quarter']
    numeric_features = [f for f in numeric_features if f not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    numeric_feature_names = numeric_features
    cat_encoder = preprocessor.named_transformers_['cat']
    categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])
    
    # Time-based train-test split (use latest days for testing)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Ensemble feature selection
    print("Performing ensemble feature selection...")
    feature_importance = ensemble_feature_selection(X_train, y_train, RANDOM_STATE)
    
    # Select top features
    top_k = min(30, len(all_feature_names))
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    selected_features = all_feature_names[top_indices].tolist()
    
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]
    
    print(f"Selected {len(selected_features)} most important features")
    
    models = {}
    metrics = {}
    
    # 1. XGBoost Model
    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train_selected, y_train)
    models['xgb'] = xgb_model
    y_pred_xgb = xgb_model.predict(X_test_selected)
    
    # 2. MLP Model
    print("Training MLP...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    mlp_model.fit(X_train_selected, y_train)
    models['mlp'] = mlp_model
    y_pred_mlp = mlp_model.predict(X_test_selected)
    
    # 3. GRU Model (simplified - using tabular data as sequences)
    print("Training GRU...")
    # Reshape data for GRU (samples, timesteps=1, features)
    X_train_gru = X_train_selected.reshape(X_train_selected.shape[0], 1, X_train_selected.shape[1])
    X_test_gru = X_test_selected.reshape(X_test_selected.shape[0], 1, X_test_selected.shape[1])
    
    gru_model = Sequential([
        GRU(32, activation='relu', input_shape=(1, X_train_selected.shape[1])),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    gru_model.fit(X_train_gru, y_train, epochs=50, batch_size=32, verbose=0)
    models['gru'] = gru_model
    y_pred_gru = gru_model.predict(X_test_gru).flatten()
    
    # Evaluate models
    best_model = None
    best_mae = float('inf')
    
    for model_name, y_pred in [('xgb', y_pred_xgb), ('mlp', y_pred_mlp), ('gru', y_pred_gru)]:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{model_name.upper():6} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model_name
    
    print(f"Best model: {best_model.upper()} with MAE: {best_mae:.4f}")
    
    # Generate forecast (using latest available data)
    latest_data = X_test_selected[-1].reshape(1, -1)
    if best_model == 'gru':
        latest_data_gru = latest_data.reshape(1, 1, -1)
        forecast = gru_model.predict(latest_data_gru).flatten()[0]
    else:
        forecast = models[best_model].predict(latest_data)[0]
    
    # Prepare artifacts
    import os
    artifact_path = f"model_artifacts/{target_variable.replace(' ', '_').replace('/', '_')}"
    os.makedirs(artifact_path, exist_ok=True)
    
    # Save artifacts
    with open(f"{artifact_path}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    forecast_df = pd.DataFrame({
        'date': [data.index[-1] + pd.Timedelta(days=i) for i in range(1, FORECAST_HORIZON+1)],
        'forecast': [forecast] * FORECAST_HORIZON
    })
    forecast_df.to_csv(f"{artifact_path}/forecast.csv", index=False)
    
    # SHAP explanations for XGBoost
    if best_model == 'xgb':
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_selected)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_selected, feature_names=selected_features, show=False)
        plt.tight_layout()
        plt.savefig(f"{artifact_path}/global_shap.png")
        plt.close()
    
    # Save best model
    if best_model == 'gru':
        gru_model.save(f"{artifact_path}/model.h5")
    else:
        joblib.dump(models[best_model], f"{artifact_path}/model.joblib")
    
    print(f"Artifacts saved to: {artifact_path}")
    return metrics, best_model

# Train for all target variables
results = {}
for target_var in TARGET_VARIABLES:
    try:
        metrics, best_model = train_for_variable(target_var, df)
        results[target_var] = {'best_model': best_model, 'metrics': metrics[best_model]}
    except Exception as e:
        print(f"Error with {target_var}: {str(e)}")
        continue

# Save summary
with open('model_artifacts/training_summary.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nTraining completed for all variables!")
