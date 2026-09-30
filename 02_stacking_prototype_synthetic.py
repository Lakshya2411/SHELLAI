import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering class for creating additional features"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def engineer_features(self, X):
        X_engineered = X.copy()
        # 1. Polynomial features for numerical columns
        numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                # Interaction features
                X_engineered[f"interaction_{col1}_{col2}"] = X_engineered[col1] * X_engineered[col2]
                # Ratio features (with safety check)
                if (X_engineered[col2] != 0).all():
                    X_engineered[f"ratio_{col1}_{col2}"] = X_engineered[col1] / (X_engineered[col2] + 1e-8)
        # 2. Statistical features
        for col in numerical_cols:
            # Rolling statistics (if enough data)
            if len(X_engineered) > 10:
                X_engineered[f'{col}_rolling_mean'] = X_engineered[col].rolling(window=5, min_periods=1).mean()
                X_engineered[f'{col}_rolling_std'] = X_engineered[col].rolling(window=5, min_periods=1).std()
            # Percentile features
            X_engineered[f'{col}_percentile_25'] = X_engineered[col].quantile(0.25)
            X_engineered[f'{col}_percentile_75'] = X_engineered[col].quantile(0.75)
        # 3. Aggregation features
        X_engineered['sum_features'] = X_engineered[numerical_cols].sum(axis=1)
        X_engineered['mean_features'] = X_engineered[numerical_cols].mean(axis=1)
        X_engineered['std_features'] = X_engineered[numerical_cols].std(axis=1)
        X_engineered['max_features'] = X_engineered[numerical_cols].max(axis=1)
        X_engineered['min_features'] = X_engineered[numerical_cols].min(axis=1)
        # 4. Non-linear transformations
        for col in numerical_cols:
            X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
            X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X_engineered[col]))
            X_engineered[f'{col}_log'] = np.log1p(np.abs(X_engineered[col]))
        # 5. Handle missing values
        X_engineered = X_engineered.fillna(X_engineered.mean())
        return X_engineered

    def fit_transform(self, X):
        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.fit_transform(X_engineered)
        self.is_fitted = True
        return X_scaled

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        X_engineered = self.engineer_features(X)
        return self.scaler.transform(X_engineered)

class BaseModels:
    """Collection of base models for stacking"""
    def __init__(self):
        self.models = {}
        self.model_names = []
    def create_models(self):
        # Gradient Boosting Models
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            feature_fraction=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.models['gbr'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        # Tree-based Models
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=42,
            n_jobs=-1
        )
        # Linear Models
        self.models['ridge'] = Ridge(alpha=10, random_state=42)
        self.models['lasso'] = Lasso(alpha=0.1, random_state=42, max_iter=2000)
        self.models['linear'] = LinearRegression()
        # Other Models
        self.models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        self.models['knn'] = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        self.models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.1,
            max_iter=500,
            random_state=42
        )
        self.model_names = list(self.models.keys())
        return self.models

class StackingRegressor:
    """Stacking ensemble with multiple base models and one meta model"""
    def __init__(self, base_models=None, meta_model=None, n_folds=5, random_state=42):
        self.base_models = base_models or BaseModels().create_models()
        self.meta_model = meta_model or Ridge(alpha=1, random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
    def _get_oof_predictions(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_predictions = np.zeros((len(X), len(self.base_models)))
        print("Generating out-of-fold predictions for base models...")
        for i, (model_name, model) in enumerate(self.base_models.items()):
            print(f"Processing {model_name}...")
            fold_predictions = np.zeros(len(X))
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                model.fit(X_train_fold, y_train_fold)
                fold_predictions[val_idx] = model.predict(X_val_fold)
            oof_predictions[:, i] = fold_predictions
        return oof_predictions
    def fit(self, X, y):
        print("Starting stacking ensemble training...")
        # Feature engineering
        print("Applying feature engineering...")
        X_engineered = self.feature_engineer.fit_transform(X)
        # Get out-of-fold predictions from base models
        oof_predictions = self._get_oof_predictions(X_engineered, y)
        # Train meta model on out-of-fold predictions
        print("Training meta model...")
        self.meta_model.fit(oof_predictions, y)
        # Retrain base models on full dataset
        print("Retraining base models on full dataset...")
        for model_name, model in self.base_models.items():
            print(f"Retraining {model_name}...")
            model.fit(X_engineered, y)
        self.is_fitted = True
        print("Stacking ensemble training completed!")
        return self
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X_engineered = self.feature_engineer.transform(X)
        base_predictions = np.column_stack([
            model.predict(X_engineered) for model in self.base_models.values()
        ])
        final_prediction = self.meta_model.predict(base_predictions)
        return final_prediction
    def get_feature_importance(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        importance_dict = {}
        if hasattr(self.meta_model, 'coef_'):
            for i, model_name in enumerate(self.base_models.keys()):
                importance_dict[model_name] = self.meta_model.coef_[i]
        else:
            for model_name in self.base_models.keys():
                importance_dict[model_name] = 0
        return importance_dict

def create_synthetic_data(n_samples=200, n_features=15, noise=0.05, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    y = (
        X[:, 0] ** 2 +
        X[:, 1] * X[:, 2] +
        np.sin(X[:, 3]) +
        np.exp(X[:, 4] * 0.1) +
        X[:, 5:].sum(axis=1) * 0.1 +
        np.random.normal(0, noise, n_samples)
    )
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=pd.Index(feature_names))
    return X_df, y

def evaluate_model(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    return mse, rmse, r2

def main():
    print("=== Stacking Ensemble with Feature Engineering ===\n")
    # Create synthetic data
    print("Creating synthetic dataset...")
    X, y = create_synthetic_data(n_samples=200, n_features=15, noise=0.05)
    print(f"Dataset shape: {np.array(X).shape}")
    print(f"Target shape: {np.array(y).shape}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {np.array(X_train).shape}")
    print(f"Test set: {np.array(X_test).shape}")
    # Create and train stacking ensemble
    print("Creating and training stacking ensemble...")
    stacking_model = StackingRegressor(n_folds=5, random_state=42)
    stacking_model.fit(X_train, y_train)
    # Make predictions
    print("Making predictions...")
    y_pred_stacking = stacking_model.predict(X_test)
    # Evaluate stacking ensemble
    evaluate_model(y_test, y_pred_stacking, "Stacking Ensemble")
    # Compare with individual base models
    print("Comparing with individual base models...")
    X_train_engineered = stacking_model.feature_engineer.transform(X_train)
    X_test_engineered = stacking_model.feature_engineer.transform(X_test)
    for model_name, model in stacking_model.base_models.items():
        y_pred_single = model.predict(X_test_engineered)
        evaluate_model(y_test, y_pred_single, f"Base Model: {model_name}")
    # Show feature importance
    print("Meta Model Feature Importance (Base Model Weights):")
    importance = stacking_model.get_feature_importance()
    for model_name, weight in sorted(importance.items(), key=lambda x: abs(x[1]) if isinstance(x, tuple) else abs(x[1]) if isinstance(x, list) else abs(weight), reverse=True):
        print(f"{model_name}: {weight:.4f}")
    # Cross-validation comparison
    print("Cross-validation comparison...")
    stacking_cv_scores = cross_val_score(
        stacking_model, X, y, cv=5, scoring='r2', n_jobs=-1
    )
    print(f"Stacking Ensemble CV R²: {stacking_cv_scores.mean():.4f} (+/- {stacking_cv_scores.std() * 2:.4f})")
    for model_name, model in stacking_model.base_models.items():
        cv_scores = cross_val_score(
            model, X_train_engineered, y_train, cv=5, scoring='r2', n_jobs=-1
        )
        print(f"{model_name} CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("Stacking ensemble demonstration completed!")

if __name__ == "__main__":
    main() 