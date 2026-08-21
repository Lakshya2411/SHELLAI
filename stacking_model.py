import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Assume target is the last column
X = train.iloc[:, :-1]
y = train.iloc[:, -1]
X_test = test.copy()

# Basic feature engineering
# 1. Handle missing values
# 2. Encode categoricals
# 3. Create a new feature: sum of numeric columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Ensure all numeric_features exist in X_test
for col in numeric_features:
    if col not in X_test.columns:
        X_test[col] = np.nan
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

# Add a simple new feature: sum of numeric columns
X['num_sum'] = X[numeric_features].sum(axis=1)
X_test['num_sum'] = X_test[numeric_features].sum(axis=1)
numeric_features.append('num_sum')

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ('ridge', Ridge())
]

# Meta-model
meta_model = LinearRegression()

# Stacking regressor
stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1
)

# Full pipeline
model = Pipeline([
    ('pre', preprocessor),
    ('stack', stacking)
])

# Fit model
model.fit(X, y)

# Predict on test set
preds = model.predict(X_test)

# Output predictions
output = pd.DataFrame({'prediction': preds})
output.to_csv('predictions.csv', index=False)

print('Predictions saved to predictions.csv') 