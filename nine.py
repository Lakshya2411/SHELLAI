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
    # --- Load Data ---
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Please place train.csv, test.csv, and sample_submission.csv in the project directory.")
        return

    # 1. Prepare Data
    y = train_df.filter(regex='BlendProperty').astype(np.float32)
    # The 'ID' column is not a feature, so we drop it from the training and test sets if it exists.
    X = train_df.filter(regex='Component')
    X_test = test_df.filter(regex='Component')

    # 2. Engineer Features for both train and test sets
    X_engineered = create_even_more_advanced_features(X)
    X_test_engineered = create_even_more_advanced_features(X_test)
    
    # Align columns after feature engineering - crucial for consistent predictions
    train_cols = X_engineered.columns
    test_cols = X_test_engineered.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_engineered[c] = 0
        
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_engineered[c] = 0
        
    X_test_engineered = X_test_engineered[train_cols]


    # --- Hyperparameters ---
    optimized_lgbm_params = {
        'objective': 'mape',
        'metric': 'mape',
        'n_estimators': 5000,
        'learning_rate': 0.01,
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
    
    # 3. Train Model and Generate Predictions
    y_numpy = y.values
    oof_preds_all_targets = np.zeros(y_numpy.shape)
    test_preds_all_targets = np.zeros((len(test_df), y.shape[1]))
    
    for i in range(y.shape[1]):
        target_name = y.columns[i]
        print("\n" + "="*55)
        print(f"🎯 Training for Target: {target_name} ({i+1}/10)")
        print("="*55)
        
        current_y = y_numpy[:, i]
        
        # --- Cross-validation for OOF score ---
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(kf.split(X_engineered)):
            print(f"--- Processing Fold {fold+1}/10 ---")
            X_train, X_val = X_engineered.iloc[train_index], X_engineered.iloc[val_index]
            y_train, y_val = current_y[train_index], current_y[val_index]

            model = lgb.LGBMRegressor(**optimized_lgbm_params)
            callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
            
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='mape',
                      callbacks=callbacks)
            
            oof_preds_all_targets[val_index, i] = model.predict(X_val)
        
        # --- Train final model on all data and predict on test set ---
        print(f"\n--- Training final model for {target_name} and predicting on test data ---")
        final_model = lgb.LGBMRegressor(**optimized_lgbm_params)
        final_model.fit(X_engineered, current_y) 
        test_preds_all_targets[:, i] = final_model.predict(X_test_engineered)
        
        gc.collect()

    # 4. Evaluate OOF Performance
    overall_mape_score = mean_absolute_percentage_error(y_numpy, oof_preds_all_targets)
    leaderboard_score = calculate_leaderboard_score(overall_mape_score)

    print("\n" + "="*55)
    print("🏆 FINAL OVERALL MODEL PERFORMANCE (VALIDATION)")
    print(f"✅ Overall Out-of-Fold MAPE: {overall_mape_score:.6f}")
    print(f"🚀 Final Estimated Leaderboard Score: {leaderboard_score:.4f}")
    print("="*55)
    
    # --- 5. Create and save the submission file ---
    submission_df = pd.DataFrame(test_preds_all_targets, columns=y.columns)
    
    # FIX: The column in sample_submission.csv is 'ID', not 'id'.
    # It's also safer to take the ID from the original test_df to ensure alignment.
    submission_df['ID'] = test_df['ID']
    
    # Reorder columns to match the exact format of sample_submission.csv
    submission_df = submission_df[sample_submission.columns]
    
    # Ensure no negative predictions
    for col in submission_df.columns:
        # FIX: The column is 'ID', not 'id'.
        if col != 'ID':
            submission_df[col] = submission_df[col].clip(0)

    submission_df.to_csv('submission.csv', index=False)
    print("\n✅ Predictions saved to submission.csv")

if __name__ == '__main__':
    main()
