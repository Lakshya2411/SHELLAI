import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, X):
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit encoders for categorical variables
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna('missing'))
            self.encoders[col] = le
            
        X_processed = self._transform_features(X)
        self.scaler.fit(X_processed)
        return self

    def transform(self, X):
        X_processed = self._transform_features(X)
        return self.scaler.transform(X_processed)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _transform_features(self, X):
        X_new = X.copy()
        
        # Handle missing values
        for col in self.numeric_cols:
            X_new[col] = X_new[col].fillna(X_new[col].median())
            
        # Encode categorical variables
        for col in self.categorical_cols:
            X_new[col] = self.encoders[col].transform(X_new[col].astype(str).fillna('missing'))
            
        # Create polynomial features
        if len(self.numeric_cols) > 0:
            poly_features = self.poly.fit_transform(X_new[self.numeric_cols])
            feature_names = [f"{self.numeric_cols[i]}_{j}" for i in range(len(self.numeric_cols)) 
                           for j in range(i+1)]
            poly_df = pd.DataFrame(poly_features[:, len(self.numeric_cols):], 
                                 columns=feature_names,
                                 index=X_new.index)
            X_new = pd.concat([X_new, poly_df], axis=1)
            
        # Create statistical features
        for col in self.numeric_cols:
            X_new[f'{col}_log'] = np.log1p(np.abs(X_new[col]))
            X_new[f'{col}_sqrt'] = np.sqrt(np.abs(X_new[col]))
            
        # Create interaction features between all numeric columns
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                X_new[f'{col1}_x_{col2}'] = X_new[col1] * X_new[col2]
                X_new[f'{col1}_div_{col2}'] = X_new[col1] / (X_new[col2] + 1e-6)
        
        # Convert to numpy array while preserving all numeric columns
        return X_new.select_dtypes(include=[np.number]).values

class StackingEnsemble:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.fitted_base_models = []
        
    def fit(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{len(self.base_models)}...")
            oof_preds = np.zeros(X.shape[0])
            
            # If model is RandomizedSearchCV, fit it first
            if isinstance(model, RandomizedSearchCV):
                model.fit(X, y)
                best_model = model.best_estimator_
            else:
                best_model = model
                
            # Generate out-of-fold predictions
            for train_idx, val_idx in kf.split(X):
                if isinstance(best_model, Pipeline):
                    model_clone = Pipeline([
                        (name, clone(transform)) 
                        for name, transform in best_model.steps
                    ])
                else:
                    model_clone = clone(best_model)
                
                model_clone.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = model_clone.predict(X[val_idx])
            
            meta_features[:, i] = oof_preds
            
            # Fit final model on whole dataset
            if isinstance(best_model, Pipeline):
                final_model = Pipeline([
                    (name, clone(transform)) 
                    for name, transform in best_model.steps
                ])
            else:
                final_model = clone(best_model)
                
            final_model.fit(X, y)
            self.fitted_base_models.append(final_model)
            
        self.meta_model.fit(meta_features, y)
        return self
    def predict(self, X):
        base_preds = np.column_stack([model.predict(X) for model in self.fitted_base_models])
        return self.meta_model.predict(base_preds)

def get_optimized_models():
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    lgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 70]
    }
    
    return [
        RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=10, cv=5, random_state=42),
        RandomizedSearchCV(GradientBoostingRegressor(random_state=42), gb_params, n_iter=10, cv=5, random_state=42),
        RandomizedSearchCV(XGBRegressor(random_state=42), xgb_params, n_iter=10, cv=5, random_state=42),
        RandomizedSearchCV(LGBMRegressor(random_state=42), lgb_params, n_iter=10, cv=5, random_state=42),
        Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))]),
        ElasticNet(random_state=42)
    ]

def main():
    # Load data
    df = pd.read_csv('train.csv')
    print("Available columns:", df.columns.tolist())
    
    # Assuming the last column is the target
    target_col = df.columns[-1]
    print(f"Using '{target_col}' as target column")
    
    # Remove any rows with NaN in target column
    df = df.dropna(subset=[target_col])
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Advanced feature engineering - convert to numpy arrays
    fe = AdvancedFeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)
    X_test_proc = fe.transform(X_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Processed features: {X_train_proc.shape[1]}")
    
    # Get optimized base models
    base_models = get_optimized_models()
    
    # Meta model with regularization
    meta_model = Ridge(alpha=0.5)
    
    # Create and train stacking ensemble
    print("\nTraining stacking ensemble...")
    stacking = StackingEnsemble(base_models, meta_model, n_folds=5)
    stacking.fit(X_train_proc, y_train)  # y_train is already numpy array
    
    # Make predictions
    y_pred = stacking.predict(X_test_proc)
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMean Absolute Percentage Error: {mape:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    # Save predictions for test set
    test_df = pd.read_csv('test.csv')
    test_proc = fe.transform(test_df)
    test_predictions = stacking.predict(test_proc)
    
    # Save predictions
    submission = pd.DataFrame({'Id': range(len(test_predictions)), 'Prediction': test_predictions})
    submission.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to 'predictions.csv'")

if __name__ == "__main__":
    main()
