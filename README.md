# Shell.ai Hackathon — Fuel Blend Property Prediction

My experiments for the Shell.ai Hackathon challenge on predicting the properties of fuel blends.

## The problem

Each training row describes a blend of **5 components**: the fraction of each component in the blend, plus 10 measured properties of each component (55 input features). The task is to predict **10 properties of the final blend** (`BlendProperty1`–`BlendProperty10`), which makes it a **multi-output regression** problem.

- Training data: 2,000 blends · Test data: 500 blends
- Metric: MAPE (mean absolute percentage error), converted to a leaderboard score with `max(10, 100 − 90 × MAPE / reference)`

## How the approach evolved

The scripts are numbered in the order I worked on them.

| # | Script | What changed |
|---|---|---|
| 01 | `01_multimodel_ensemble_optuna.py` | First attempt: ensemble of XGBoost, LightGBM, Random Forest and linear models, with Optuna tuning per model |
| 02 | `02_stacking_prototype_synthetic.py` | Stacking ensemble prototyped on synthetic data to get the mechanics right |
| 03 | `03_stacking_ensemble_generic.py` | Generalised stacking class (works for classification or regression) run on the real data |
| 04 | `04_lgbm_feature_engineering.py` | Switched to LightGBM with a MAPE objective and engineered blend features |
| 05 | `05_lgbm_10fold_cv.py` | Replaced the single train/validation split with 10-fold cross-validation |
| 06 | `06_lgbm_optuna_tuning.py` | Richer feature engineering + Optuna hyperparameter search |
| 07 | `07_lgbm_early_stopping.py` | More trees controlled by early stopping |
| 08 | `08_lgbm_xgb_catboost_stack.py` | LightGBM + XGBoost + CatBoost combined by a Ridge meta-model trained on out-of-fold predictions |
| 09 | `09_lgbm_final_submission.py` | Final pipeline: out-of-fold MAPE per target, writes `submission.csv` |

`stacking_baseline.py` and `stacking_example.py` are early stacking baselines.

**Key ideas used:** feature engineering on weighted component properties, per-target models, K-fold out-of-fold evaluation, early stopping, Optuna tuning, and model stacking.

## Run it

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
python 09_lgbm_final_submission.py
```

The script prints the cross-validated MAPE and estimated leaderboard score, then writes `submission.csv`.

<!-- Add your final leaderboard score / rank here -->
