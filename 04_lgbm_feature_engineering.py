import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """
    Engineers sophisticated features to capture complex interactions, crucial for a high score.
    """
    print("Starting advanced feature engineering...")
    features_df = df.copy()

    # --- Interaction Features ---
    for i in range(1, 6):
        for j in range(i + 1, 6):
            frac_i = f'Component{i}_fraction'
            frac_j = f'Component{j}_fraction'
            features_df[f'frac_{i}_x_{j}'] = features_df[frac_i] * features_df[frac_j]

    # --- Weighted Property Features ---
    for prop_num in range(1, 11):
        weighted_prop_name = f'Weighted_Property_{prop_num}'
        features_df[weighted_prop_name] = 0
        for comp_num in range(1, 6):
            frac_col = f'Component{comp_num}_fraction'
            prop_col = f'Component{comp_num}_Property{prop_num}'
            if frac_col in features_df.columns and prop_col in features_df.columns:
                features_df[weighted_prop_name] += features_df[frac_col] * features_df[prop_col]
    
    # --- Statistical Features ---
    all_prop_cols = features_df.filter(regex='_Property').columns
    for comp_num in range(1, 6):
        comp_prop_cols = [col for col in all_prop_cols if f'Component{comp_num}_' in col]
        if comp_prop_cols:
            features_df[f'Comp{comp_num}_props_mean'] = features_df[comp_prop_cols].mean(axis=1)
            features_df[f'Comp{comp_num}_props_std'] = features_df[comp_prop_cols].std(axis=1)

    print(f"Feature engineering complete. Total features: {features_df.shape[1]}")
    return features_df

def calculate_leaderboard_score(mape_cost):
    """
    Calculates the final leaderboard score based on the formula from the problem statement.
    """
    reference_cost = 2.72 # Using the public leaderboard reference cost
    score = max(10, (100 - (90 * mape_cost / reference_cost)))
    return score

def main():
    """
    Main function to execute the full ML pipeline.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found. Please place it in the same directory.")
        return
    print("Data loaded successfully.")

    # --- 2. Define Features (X) and Targets (y) ---
    y = df.filter(regex='BlendProperty')
    X = df.filter(regex='Component')

    # --- 3. Create Advanced Features ---
    X_engineered = create_advanced_features(X)
    
    # --- 4. Split Data ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_engineered, y, test_size=0.15, random_state=42
    )
    print(f"\nData split into training and validation sets.")

    # --- 5. Define the High-Performance Model ---
    lgbm = lgb.LGBMRegressor(
        objective='mape',
        n_estimators=1500,  # Reduced slightly as early stopping is removed
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(lgbm)

    # --- 6. Train the Model ---
    print("\nTraining the model... This may take a few moments.")
    
    # ** THE FIX IS HERE **
    # The .fit() method for MultiOutputRegressor does not accept extra parameters.
    # We remove eval_set, eval_metric, and callbacks from this call.
    model.fit(X_train, y_train)

    # --- 7. Evaluate on Validation Set ---
    print("\nModel training complete. Evaluating performance...")
    predictions = model.predict(X_val)
    
    mape_score = mean_absolute_percentage_error(y_val, predictions)
    leaderboard_score = calculate_leaderboard_score(mape_score)
    
    print("\n" + "="*55)
    print(f"✅ Validation MAPE: {mape_score:.6f}")
    print(f"🚀 Estimated Leaderboard Score: {leaderboard_score:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()