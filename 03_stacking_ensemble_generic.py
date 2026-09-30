import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.utils.multiclass import type_of_target
import warnings
warnings.filterwarnings('ignore')

class SimpleFeatureEngineer:
    """Simple feature engineering class"""
    
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.categorical_cols = []
        
    def fit(self, X):
        # Identify column types
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit encoders for categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna('missing'))
            self.encoders[col] = le
        
        # Fit scaler
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
        
        # Fill missing values
        for col in self.numeric_cols:
            X_new[col] = X_new[col].fillna(X_new[col].median())
        
        # Encode categorical variables
        for col in self.categorical_cols:
            X_new[col] = self.encoders[col].transform(X_new[col].astype(str).fillna('missing'))
        
        # Create simple interaction features (only for numeric columns)
        if len(self.numeric_cols) >= 2:
            for i in range(min(3, len(self.numeric_cols))):
                for j in range(i+1, min(4, len(self.numeric_cols))):
                    col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                    X_new[f'{col1}_x_{col2}'] = X_new[col1] * X_new[col2]
        
        # Create polynomial features for first few numeric columns
        for col in self.numeric_cols[:3]:
            X_new[f'{col}_squared'] = X_new[col] ** 2
        
        return X_new.select_dtypes(include=[np.number])

class StackingEnsemble:
    """Simple Stacking Ensemble that works for both classification and regression"""
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.fitted_base_models = []
        self.problem_type = None
        
    def fit(self, X, y):
        # Determine problem type
        self.problem_type = self._get_problem_type(y)
        print(f"Problem type: {self.problem_type}")
        
        # Create cross-validation folds
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Initialize meta features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        print("Training base models...")
        for i, (name, model) in enumerate(self.base_models):
            print(f"  Training {name}...")
            oof_predictions = np.zeros(X.shape[0])
            
            # Cross-validation predictions
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                
                # Clone and train model
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Get predictions
                val_preds = model_clone.predict(X_val_fold)
                oof_predictions[val_idx] = val_preds
            
            meta_features[:, i] = oof_predictions
            
            # Train final model on full dataset
            final_model = self._clone_model(model)
            final_model.fit(X, y)
            self.fitted_base_models.append((name, final_model))
        
        # Train meta model
        print("Training meta model...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        # Get predictions from base models
        base_predictions = np.zeros((X.shape[0], len(self.fitted_base_models)))
        
        for i, (name, model) in enumerate(self.fitted_base_models):
            base_predictions[:, i] = model.predict(X)
        
        # Get final prediction from meta model
        return self.meta_model.predict(base_predictions)
    
    def _get_problem_type(self, y):
        """Determine if this is classification or regression"""
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass']:
            return 'classification'
        else:
            return 'regression'
    
    def _clone_model(self, model):
        """Create a copy of the model with same parameters"""
        return model.__class__(**model.get_params())

def create_models(problem_type):
    """Create appropriate models based on problem type"""
    if problem_type == 'classification':
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(kernel='rbf', random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('svm', SVR(kernel='rbf')),
            ('knn', KNeighborsRegressor(n_neighbors=5))
        ]
        meta_model = LinearRegression()
    
    return base_models, meta_model

def evaluate_model(y_true, y_pred, problem_type):
    """Evaluate model performance"""
    if problem_type == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        if len(np.unique(y_true)) == 2:  # Binary classification
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
        
        return accuracy
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return r2

def main():
    # Load or create data
    print("Loading data...")
    try:
        df = pd.read_csv('train.csv')
        print(f"Loaded data shape: {df.shape}")
    except FileNotFoundError:
        print("train.csv not found. Creating sample data...")
        np.random.seed(42)
        n_samples = 1000
        
        # Create mixed data (could be classification or regression)
        df = pd.DataFrame({
            'num_feat1': np.random.normal(0, 1, n_samples),
            'num_feat2': np.random.normal(5, 2, n_samples),
            'num_feat3': np.random.exponential(1, n_samples),
            'cat_feat1': np.random.choice(['A', 'B', 'C'], n_samples),
            'cat_feat2': np.random.choice(['X', 'Y'], n_samples),
            'target': np.random.choice([0, 1], n_samples)  # Binary classification
        })
        
        # Uncomment next line for regression example
        # df['target'] = df['num_feat1'] * 2 + df['num_feat2'] * 0.5 + np.random.normal(0, 0.1, n_samples)
        
        print(f"Sample data created with shape: {df.shape}")
    
    # Prepare features and target
    print(f"Columns: {list(df.columns)}")
    
    # Assume last column is target (modify as needed)
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target stats: min={y.min():.2f}, max={y.max():.2f}, unique={y.nunique()}")
    
    # Clean target if needed
    if y.isnull().sum() > 0:
        print(f"Removing {y.isnull().sum()} missing target values")
        mask = ~y.isnull()
        X = X.loc[mask]
        y = y.loc[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature engineering
    print("\nApplying feature engineering...")
    fe = SimpleFeatureEngineer()
    X_train_processed = fe.fit_transform(X_train)
    X_test_processed = fe.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Processed features: {X_train_processed.shape[1]}")
    
    # Determine problem type and create models
    problem_type = 'classification' if y.nunique() <= 20 and y.dtype in ['int64', 'object'] else 'regression'
    base_models, meta_model = create_models(problem_type)
    
    # Create and train stacking ensemble
    print(f"\nCreating stacking ensemble for {problem_type}...")
    stacking = StackingEnsemble(base_models, meta_model, n_folds=5)
    stacking.fit(X_train_processed, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = stacking.predict(X_test_processed)
    
    # Evaluate ensemble
    print(f"\nStacking Ensemble Results:")
    ensemble_score = evaluate_model(y_test, y_pred, problem_type)
    
    # Compare with individual models
    print(f"\nIndividual Model Comparison:")
    for name, model in base_models:
        model_clone = stacking._clone_model(model)
        model_clone.fit(X_train_processed, y_train)
        individual_pred = model_clone.predict(X_test_processed)
        
        print(f"\n{name}:")
        individual_score = evaluate_model(y_test, individual_pred, problem_type)
    
    # Cross-validation
    print(f"\nCross-validation results:")
    cv_scores = cross_val_score(stacking, X_train_processed, y_train, cv=5)
    print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print(f"\nStacking ensemble completed successfully!")
    print(f"Problem type: {problem_type}")
    print(f"Best method: Stacking Ensemble")

if __name__ == "__main__":
    main()