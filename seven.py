import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import gc

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings('ignore')

def create_even_more_advanced_features(df):
    """
    Engineers a highly advanced set of features with complex interactions.
    """
    print("Starting more advanced feature engineering...")
    features_df = df.copy()
    
    # --- 1. Weighted Property Features (Your original strong feature set) ---
    for prop_num in range(1, 11):
        weighted_prop_name = f'Weighted_Property_{prop_num}'
        features_df[weighted_prop_name] = 0
        for comp_num in range(1, 6):
            frac_col = f'Component{comp_num}_fraction'
            prop_col = f'Component{comp_num}_Property{prop_num}'
            features_df[weighted_prop_name] += features_df[frac_col] * features_df[prop_col]
            
    # --- 2. Interaction and Ratio Features ---
    for i in range(1, 6):
        for j in range(i + 1, 6):
            frac_i = f'Component{i}_fraction'
            frac_j = f'Component{j}_fraction'
            
            # Ratio of fractions to capture dominance
            features_df[f'Comp{i}_vs_{j}_frac_ratio'] = features_df[frac_i] / (features_df[frac_j] + 1e-6)
            
            for prop_num in range(1, 11):
                prop_i = f'Component{i}_Property{prop_num}'
                prop_j = f'Component{j}_Property{prop_num}'
                
                # Difference and ratio of properties
                features_df[f'Comp{i}_{j}_prop{prop_num}_diff'] = features_df[prop_i] - features_df[prop_j]
                features_df[f'Comp{i}_{j}_prop{prop_num}_ratio'] = features_df[prop_i] / (features_df[prop_j] + 1e-6)

    # --- 3. Aggregate features across all components ---
    for prop_num in range(1, 11):
        prop_cols = [f'Component{c}_Property{prop_num}' for c in range(1, 6)]
        features_df[f'Overall_Property{prop_num}_mean'] = features_df[prop_cols].mean(axis=1)
        features_df[f'Overall_Property{prop_num}_std'] = features_df[prop_cols].std(axis=1)
        
    # --- 4. Impute NaNs and Handle Duplicates ---
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df.fillna(features_df.median(), inplace=True)
    features_df = features_df.loc[:,~features_df.columns.duplicated()]

    print(f"Feature engineering complete. Total unique features: {features_df.shape[1]}")
    return features_df

def calculate_leaderboard_score(mape_cost):
    """
    Calculates the final leaderboard score based on the hackathon's formula.
    """
    reference_cost = 2.72
    return max(10, (100 - (90 * mape_cost / reference_cost)))

def main():
    """
    Main function to execute the full, stable, high-performance ML pipeline.
    """
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: 'train.csv' not found. Please place it in the project directory.")
        return

    # 1. Prepare Data
    y = df.filter(regex='BlendProperty').astype(np.float32)
    X = df.filter(regex='Component')

    # 2. Engineer Features
    X_engineered = create_even_more_advanced_features(X)

    # --- NEW: A SINGLE, ROBUST SET OF HYPERPARAMETERS ---
    # Using a single, well-tuned parameter set is more efficient and robust
    # than running a new search for each of the 10 targets.
    optimized_lgbm_params = {
        'objective': 'mape',
        'metric': 'mape',
        'n_estimators': 5000,  # Increased capacity, controlled by early stopping
        'learning_rate': 0.01, # A smaller learning rate often yields better results
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 40,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    # 3. Train Final Model for Each Target with 10-Fold CV
    y_numpy = y.values
    oof_preds_all_targets = np.zeros(y_numpy.shape)
    
    for i in range(y.shape[1]):
        target_name = y.columns[i]
        print("\n" + "="*55)
        print(f"🎯 Training for Target: {target_name} ({i+1}/10)")
        print("="*55)
        
        current_y = y_numpy[:, i]
        
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_engineered)):
            print(f"--- Processing Fold {fold+1}/10 ---")
            X_train, X_val = X_engineered.iloc[train_index], X_engineered.iloc[val_index]
            y_train, y_val = current_y[train_index], current_y[val_index]

            model = lgb.LGBMRegressor(**optimized_lgbm_params)
            
            # Using new callback format for recent versions of LightGBM
            callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
            
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='mape',
                      callbacks=callbacks)
            
            oof_preds_all_targets[val_index, i] = model.predict(X_val)
        
        # --- NEW: Clean up memory after training for each target ---
        gc.collect()

    # 4. Evaluate Final Overall Performance
    overall_mape_score = mean_absolute_percentage_error(y_numpy, oof_preds_all_targets)
    leaderboard_score = calculate_leaderboard_score(overall_mape_score)

    print("\n" + "="*55)
    print("🏆 FINAL OVERALL MODEL PERFORMANCE (VALIDATION)")
    print(f"✅ Overall Out-of-Fold MAPE: {overall_mape_score:.6f}")
    print(f"🚀 Final Estimated Leaderboard Score: {leaderboard_score:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()