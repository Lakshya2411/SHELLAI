import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna  # --- NEW: For hyperparameter tuning ---
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import warnings

# Suppress unnecessary warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- NEW: More advanced feature engineering ---
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
            
    # --- 2. NEW: Interaction and Ratio Features ---
    # These features help capture non-linear relationships between component properties.
    for i in range(1, 6):
        for j in range(i + 1, 6):
            frac_i = f'Component{i}_fraction'
            frac_j = f'Component{j}_fraction'
            
            # Ratio of fractions to capture dominance
            # Adding a small epsilon to avoid division by zero
            features_df[f'Comp{i}_vs_{j}_frac_ratio'] = features_df[frac_i] / (features_df[frac_j] + 1e-6)
            
            for prop_num in range(1, 11):
                prop_i = f'Component{i}_Property{prop_num}'
                prop_j = f'Component{j}_Property{prop_num}'
                
                # Difference and ratio of properties
                features_df[f'Comp{i}_{j}_prop{prop_num}_diff'] = features_df[prop_i] - features_df[prop_j]
                features_df[f'Comp{i}_{j}_prop{prop_num}_ratio'] = features_df[prop_i] / (features_df[prop_j] + 1e-6)

    # --- 3. NEW: Aggregate features across all components ---
    # These summarize the "profile" of the entire blend.
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

# --- NEW: Optuna objective function for hyperparameter tuning ---
def objective(trial, X, y):
    """
    The objective function that Optuna will minimize.
    """
    # Define the search space for hyperparameters
    params = {
        'objective': 'mape',
        'metric': 'mape',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    # Use cross-validation to get a robust score for this set of parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='mape',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        oof_preds[val_index] = model.predict(X_val)
        
    return mean_absolute_percentage_error(y, oof_preds)


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

    # --- STRATEGY CHANGE: Model-per-Target ---
    # We will train one independent model for each of the 10 targets.
    
    y_numpy = y.values
    oof_preds_all_targets = np.zeros(y_numpy.shape)
    
    # Loop through each of the 10 target variables
    for i in range(y.shape[1]):
        target_name = y.columns[i]
        print("\n" + "="*55)
        print(f"🎯 Training for Target: {target_name} ({i+1}/10)")
        print("="*55)
        
        current_y = y_numpy[:, i]
        
        # --- 3. Hyperparameter Tuning with Optuna (for each target) ---
        # For a real competition, increase n_trials to 50 or 100 for better results.
        # This will take longer but is essential for a top score.
        print("🔍 Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_engineered, current_y), n_trials=25) # Using 25 trials as a balance
        
        best_params = study.best_params
        print(f"✅ Best MAPE for {target_name}: {study.best_value:.6f}")
        print(f"🔧 Best Hyperparameters found: {best_params}")
        
        # 4. Train Final Model for the Target with Best Parameters and CV
        final_params = {
            'objective': 'mape',
            'metric': 'mape',
            'n_estimators': 2000, # Using a high number with early stopping
            'learning_rate': best_params['learning_rate'],
            'num_leaves': best_params['num_leaves'],
            'max_depth': best_params['max_depth'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        print("\nTraining final model with 10-Fold Cross-Validation...")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_engineered)):
            print(f"--- Processing Fold {fold+1}/10 ---")
            X_train, X_val = X_engineered.iloc[train_index], X_engineered.iloc[val_index]
            y_train, y_val = current_y[train_index], current_y[val_index]

            model = lgb.LGBMRegressor(**final_params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='mape',
                      callbacks=[lgb.early_stopping(150, verbose=False)])
            
            oof_preds_all_targets[val_index, i] = model.predict(X_val)

    # 5. Evaluate Final Overall Performance
    overall_mape_score = mean_absolute_percentage_error(y_numpy, oof_preds_all_targets)
    leaderboard_score = calculate_leaderboard_score(overall_mape_score)

    print("\n" + "="*55)
    print("🏆 FINAL OVERALL MODEL PERFORMANCE (VALIDATION)")
    print(f"✅ Overall Out-of-Fold MAPE: {overall_mape_score:.6f}")
    print(f"🚀 Final Estimated Leaderboard Score: {leaderboard_score:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()