import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
# FIX: This import is correct for modern XGBoost and is now required
from xgboost.callback import EarlyStopping
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
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

    # --- 1. Weighted Property Features ---
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
            features_df[f'Comp{i}_vs_{j}_frac_ratio'] = features_df[frac_i] / (features_df[frac_j] + 1e-6)
            for prop_num in range(1, 11):
                prop_i = f'Component{i}_Property{prop_num}'
                prop_j = f'Component{j}_Property{prop_num}'
                features_df[f'Comp{i}_{j}_prop{prop_num}_diff'] = features_df[prop_i] - features_df[prop_j]
                features_df[f'Comp{i}_{j}_prop{prop_num}_ratio'] = features_df[prop_i] / (features_df[prop_j] + 1e-6)

    # --- 3. Aggregate features across all components ---
    for prop_num in range(1, 11):
        prop_cols = [f'Component{c}_Property{prop_num}' for c in range(1, 6)]
        features_df[f'Overall_Property{prop_num}_mean'] = features_df[prop_cols].mean(axis=1)
        features_df[f'Overall_Property{prop_num}_std'] = features_df[prop_cols].std(axis=1)
        features_df[f'Overall_Property{prop_num}_skew'] = features_df[prop_cols].skew(axis=1)

    # --- 4. Polynomial Features on key engineered features ---
    key_features = [f'Weighted_Property_{i}' for i in range(1, 11)] + \
                   [f'Overall_Property_{i}_std' for i in range(1, 11)]
    key_features = [f for f in key_features if f in features_df.columns]
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(features_df[key_features])
    poly_df = pd.DataFrame(poly_features,
                           columns=poly.get_feature_names_out(key_features),
                           index=features_df.index)
    
    features_df = pd.concat([features_df, poly_df], axis=1)

    # --- 5. Impute NaNs and Handle Duplicates ---
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df.fillna(0, inplace=True)
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
    
    # 3. Stacking Ensemble Training
    oof_preds_all_targets = np.zeros(y.shape)
    
    for i in range(y.shape[1]):
        target_name = y.columns[i]
        print("\n" + "="*80)
        print(f"🎯 Training for Target: {target_name} ({i+1}/10)")
        print("="*80)
        
        current_y = y[target_name]
        y_transformed = np.log1p(current_y)
        
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        oof_lgb = np.zeros(len(df))
        oof_xgb = np.zeros(len(df))
        oof_cat = np.zeros(len(df))
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_engineered)):
            print(f"--- Processing Fold {fold+1}/10 ---")
            X_train, X_val = X_engineered.iloc[train_index], X_engineered.iloc[val_index]
            y_train, y_val = y_transformed.iloc[train_index], y_transformed.iloc[val_index]

            # --- Base Model 1: LightGBM ---
            lgb_params = {'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000, 'learning_rate': 0.01,
                          'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1,
                          'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
            
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          eval_metric='mae', callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_lgb[val_index] = lgb_model.predict(X_val)

            # --- Base Model 2: XGBoost ---
            xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'mae', 'n_estimators': 2000, 'learning_rate': 0.01,
                          'colsample_bytree': 0.8, 'subsample': 0.8, 'max_depth': 7, 'gamma': 0.1,
                          'lambda': 1, 'alpha': 0.1, 'n_jobs': -1, 'seed': 42}

            xgb_model = xgb.XGBRegressor(**xgb_params)
            
            # FIX: This is the correct modern syntax for early stopping. It will work after you update your library.
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False,
                          callbacks=[EarlyStopping(rounds=100)])
            oof_xgb[val_index] = xgb_model.predict(X_val)
            
            # --- Base Model 3: CatBoost ---
            cat_params = {'loss_function': 'MAE', 'eval_metric': 'MAE', 'iterations': 2000, 'learning_rate': 0.02,
                          'depth': 7, 'l2_leaf_reg': 3, 'random_strength': 1, 'bagging_temperature': 1,
                          'verbose': 0, 'random_seed': 42}

            cat_model = CatBoostRegressor(**cat_params)
            cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=100, use_best_model=True)
            oof_cat[val_index] = cat_model.predict(X_val)

            gc.collect()

        # --- Stacking Layer (Meta-Model) ---
        print("\n--- Training Meta-Model ---")
        stack_X = pd.DataFrame({'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat})
        
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(stack_X, y_transformed)

        final_oof_preds_transformed = meta_model.predict(stack_X)
        
        final_oof_preds = np.expm1(final_oof_preds_transformed)
        final_oof_preds[final_oof_preds < 0] = 0
        
        oof_preds_all_targets[:, i] = final_oof_preds
        
        target_mape = mean_absolute_percentage_error(current_y, final_oof_preds)
        print(f"✅ MAPE for {target_name}: {target_mape:.6f}")
        gc.collect()

    # 4. Evaluate Final Overall Performance
    overall_mape_score = mean_absolute_percentage_error(y, oof_preds_all_targets)
    leaderboard_score = calculate_leaderboard_score(overall_mape_score)

    print("\n" + "="*80)
    print("🏆 FINAL OVERALL STACKED MODEL PERFORMANCE (VALIDATION)")
    print(f"✅ Overall Out-of-Fold MAPE: {overall_mape_score:.6f}")
    print(f"🚀 Final Estimated Leaderboard Score: {leaderboard_score:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()