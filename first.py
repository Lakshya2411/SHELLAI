import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class FuelBlendPredictor:
    """Base predictor class for fuel blend prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def prepare_features(self, X):
        """Prepare features for training/prediction"""
        # Basic feature preparation - you can extend this
        X_prepared = X.copy()
        
        # Handle missing values
        X_prepared = X_prepared.fillna(X_prepared.mean())
        
        # Add polynomial features for key components (example)
        if len(X_prepared.columns) >= 5:
            X_prepared['feature_interaction_1'] = X_prepared.iloc[:, 0] * X_prepared.iloc[:, 1]
            X_prepared['feature_interaction_2'] = X_prepared.iloc[:, 2] * X_prepared.iloc[:, 3]
            X_prepared['feature_sum'] = X_prepared.iloc[:, :5].sum(axis=1)
        
        return X_prepared
    
    def create_models(self):
        """Create default models"""
        models = {}
        
        # Default models with basic parameters
        models['xgb'] = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        
        models['lgb'] = lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            feature_fraction=0.8,  # Correct LightGBM parameter name
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            num_leaves=64,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        models['rf'] = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        models['gbr'] = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        models['ridge'] = MultiOutputRegressor(Ridge(alpha=10.0))
        models['elastic'] = MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000))
        
        return models
    
    def fit(self, X, y):
        """Fit all models"""
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler.fit_transform(X_prepared)
        
        self.models = self.create_models()
        
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                if name in ['xgb', 'lgb']:
                    # Use early stopping for gradient boosting models
                    split_idx = int(0.9 * len(X_scaled))
                    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    if name == 'xgb':
                        model.fit(X_train, y_train, 
                                 eval_set=[(X_val, y_val)],
                                 early_stopping_rounds=50, verbose=False)
                    else:  # lgb
                        model.fit(X_train, y_train, 
                                 eval_set=[(X_val, y_val)],
                                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                else:
                    model.fit(X_scaled, y)
            except Exception as e:
                print(f"Error training {name}: {e}")
                # Fallback to simple fit without early stopping
                model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using ensemble of models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Simple ensemble: average predictions
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        return ensemble_pred

class HyperparameterOptimizer:
    def __init__(self, X, y, cv_folds=5):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.scaler = RobustScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
    def mape_scorer(self, y_true, y_pred):
        """Custom MAPE scorer for multi-output regression"""
        return mean_absolute_percentage_error(y_true, y_pred)
    
    def objective_xgb(self, trial):
        """Objective function for XGBoost hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(self.X_scaled):
            X_train_fold = self.X_scaled[train_idx]
            X_val_fold = self.X_scaled[val_idx]
            y_train_fold = self.y[train_idx]
            y_val_fold = self.y[val_idx]
            
            # Simple fit without early stopping for hyperparameter optimization
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict(X_val_fold)
            score = self.mape_scorer(y_val_fold, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def objective_lgb(self, trial):
        """Objective function for LightGBM hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),  # LightGBM parameter name
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(self.X_scaled):
            X_train_fold = self.X_scaled[train_idx]
            X_val_fold = self.X_scaled[val_idx]
            y_train_fold = self.y[train_idx]
            y_val_fold = self.y[val_idx]
            
            # Simple fit without early stopping for hyperparameter optimization
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict(X_val_fold)
            score = self.mape_scorer(y_val_fold, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def objective_rf(self, trial):
        """Objective function for Random Forest hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        
        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(self.X_scaled):
            X_train_fold = self.X_scaled[train_idx]
            X_val_fold = self.X_scaled[val_idx]
            y_train_fold = self.y[train_idx]
            y_val_fold = self.y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            score = self.mape_scorer(y_val_fold, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_all_models(self, n_trials=100):
        """Optimize hyperparameters for all models"""
        results = {}
        
        # Optimize XGBoost
        print("Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(self.objective_xgb, n_trials=n_trials)
        results['xgb'] = {
            'best_params': study_xgb.best_params,
            'best_score': study_xgb.best_value
        }
        print(f"XGBoost best score: {study_xgb.best_value:.4f}")
        
        # Optimize LightGBM
        print("Optimizing LightGBM...")
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(self.objective_lgb, n_trials=n_trials)
        results['lgb'] = {
            'best_params': study_lgb.best_params,
            'best_score': study_lgb.best_value
        }
        print(f"LightGBM best score: {study_lgb.best_value:.4f}")
        
        # Optimize Random Forest
        print("Optimizing Random Forest...")
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(self.objective_rf, n_trials=n_trials)
        results['rf'] = {
            'best_params': study_rf.best_params,
            'best_score': study_rf.best_value
        }
        print(f"Random Forest best score: {study_rf.best_value:.4f}")
        
        return results

# Enhanced FuelBlendPredictor with optimized hyperparameters
class OptimizedFuelBlendPredictor(FuelBlendPredictor):
    def __init__(self, optimized_params=None):
        super().__init__()
        self.optimized_params = optimized_params or {}
    
    def create_models(self):
        """Create models with optimized hyperparameters"""
        models = {}
        
        # XGBoost with optimized parameters
        xgb_params = self.optimized_params.get('xgb', {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'random_state': 42,
            'n_jobs': -1
        })
        models['xgb'] = xgb.XGBRegressor(**xgb_params)
        
        # LightGBM with optimized parameters
        lgb_params = self.optimized_params.get('lgb', {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'feature_fraction': 0.8,  # Correct LightGBM parameter name
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'num_leaves': 64,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        })
        models['lgb'] = lgb.LGBMRegressor(**lgb_params)
        
        # Random Forest with optimized parameters
        rf_params = self.optimized_params.get('rf', {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        })
        models['rf'] = RandomForestRegressor(**rf_params)
        
        # Keep other models from parent class
        models['gbr'] = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        models['ridge'] = MultiOutputRegressor(Ridge(alpha=10.0))
        models['elastic'] = MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000))
        
        return models

# Main optimization and training script
def main_with_optimization():
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Prepare features and targets
    feature_columns = train_df.columns[:55]
    target_columns = [f'BlendProperty{i}' for i in range(1, 11)]
    
    X_train = train_df[feature_columns]
    y_train = train_df[target_columns].values
    X_test = test_df[feature_columns]
    
    # Create basic predictor to prepare features
    basic_predictor = FuelBlendPredictor()
    X_train_prepared = basic_predictor.prepare_features(X_train)
    
    # Optimize hyperparameters
    print("Starting hyperparameter optimization...")
    optimizer = HyperparameterOptimizer(X_train_prepared, y_train)
    optimized_params = optimizer.optimize_all_models(n_trials=50)  # Reduce trials for faster execution
    
    # Print optimized parameters
    print("\nOptimized parameters:")
    for model_name, params in optimized_params.items():
        print(f"{model_name}: {params['best_params']}")
        print(f"  Best score: {params['best_score']:.4f}")
    
    # Train with optimized parameters
    print("\nTraining with optimized parameters...")
    predictor = OptimizedFuelBlendPredictor(
        optimized_params={k: v['best_params'] for k, v in optimized_params.items()}
    )
    predictor.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Create submission file
    submission_df = pd.DataFrame(predictions, columns=target_columns)
    submission_df.to_csv('optimized_submission.csv', index=False)
    
    print(f"Optimized submission file created with shape: {submission_df.shape}")
    print("Submission preview:")
    print(submission_df.head())

if __name__ == "__main__":
    main_with_optimization()