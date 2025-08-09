# Features:
# - Configurable outlier removal (IQR or threshold)
# - Train/val/test split with reproducibility
# - Hyperparameter tuning (Optuna if available; fallback to RandomizedSearchCV)
# - Robust evaluation: 10-fold CV, repeated KFold, and test metrics
# - Explainability: SHAP (bar, summary, dependence) + permutation importance
# - PDP (Partial Dependence) plots for top features
# - Saves metrics, parameters, plots, and fold predictions to /outputs
# -------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor

# ---------------------- Configuration ----------------------
INPUT_FILE = 'all-50m.xlsx'
TARGET = 'PM2.5'
SEED = 42

# Candidate features (subset will be taken by intersecting with dataset columns)
CANDIDATE_FEATURES = [
    'Fixed', 'Avg_Visibility', 'Ambient Temperature °C', 'Total - Office - 150', 'Total - Office - 50',
    'Dis-From - fuel - 0', 'Ambient Humidity %RH', 'Avg_Dew_Point', 'Avg_Wind', 'Season',
    'Dis-From - subway - 0'
]

TEST_SIZE = 0.2
CV_SPLITS = 10
REPEATED_KF_REPEATS = 10  # keep reasonable to avoid very long runs

# Outlier handling
OUTLIER_METHOD = 'iqr'      # 'iqr' or 'threshold' or None
IQR_FACTOR = 1.5
PM25_THRESHOLD = 300.0

# Hyperparameter tuning
USE_OPTUNA = True      # will auto-fallback if Optuna not installed
OPTUNA_N_TRIALS = 50   # adjust higher for better search (slower)
RANDOM_SEARCH_ITERS = 60

# Outputs
OUTDIR = Path('outputs_rf')
OUTDIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# ---------------------- Load & Clean -----------------------
data = pd.read_excel(INPUT_FILE)

# Drop obvious all-NaN columns
data = data.dropna(axis=1, how='all')

# Outlier removal
if OUTLIER_METHOD == 'threshold':
    before = len(data)
    data = data[data[TARGET] <= PM25_THRESHOLD]
    print(f'Removed {before - len(data)} rows with {TARGET} > {PM25_THRESHOLD} (threshold).')
elif OUTLIER_METHOD == 'iqr':
    q1, q3 = data[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
    before = len(data)
    data = data[(data[TARGET] >= lo) & (data[TARGET] <= hi)]
    print(f'Removed {before - len(data)} rows outside IQR bounds [{lo:.2f}, {hi:.2f}].')
else:
    print('No outlier removal applied.')

# Feature selection by intersection
FEATURES = [c for c in CANDIDATE_FEATURES if c in data.columns]
missing = [c for c in CANDIDATE_FEATURES if c not in data.columns]
if missing:
    print('Skipping missing features:', missing)

# Fill NaNs
data = data.fillna(0)

X = data[FEATURES]
y = data[TARGET]

# ---------------------- Split ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# ---------------------- Tuning -----------------------------
best_params = None
best_estimator = None

def evaluate_cv(params, X_, y_, cv=5):
    model = RandomForestRegressor(random_state=SEED, **params)
    scores = cross_val_score(model, X_, y_, cv=cv, scoring='r2', n_jobs=-1)
    return scores.mean()

used_optuna = False
if USE_OPTUNA:
    try:
        import optuna

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 40),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
            }
            return evaluate_cv(params, X_train, y_train, cv=5)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        used_optuna = True
        print('Optuna best params:', best_params)
    except Exception as e:
        print('Optuna not used (missing or failed). Falling back to RandomizedSearchCV.', e)

if not used_optuna:
    from scipy.stats import randint
    from scipy.stats import uniform
    param_dist = {
        'n_estimators': randint(200, 1500),
        'max_depth': randint(3, 40),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 8),
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    rs = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=SEED),
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITERS,
        scoring='r2',
        cv=5,
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )
    rs.fit(X_train, y_train)
    best_params = rs.best_params_
    print('RandomizedSearch best params:', best_params)

best_estimator = RandomForestRegressor(random_state=SEED, **best_params)
best_estimator.fit(X_train, y_train)

# ---------------------- Evaluation -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

y_pred_test = best_estimator.predict(X_test)
metrics = {
    'test_r2': float(r2_score(y_test, y_pred_test)),
    'test_rmse': rmse(y_test, y_pred_test)
}
print('Test R2:', metrics['test_r2'])
print('Test RMSE:', metrics['test_rmse'])

# 10-fold CV
kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(best_estimator, X, y, cv=kf, scoring='r2', n_jobs=-1)
metrics['cv10_mean_r2'] = float(cv_scores.mean())
metrics['cv10_std_r2'] = float(cv_scores.std())
print(f'10-fold CV R2 mean={metrics["cv10_mean_r2"]:.4f} std={metrics["cv10_std_r2"]:.4f}')

# Repeated KFold
rkf = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=REPEATED_KF_REPEATS, random_state=SEED)
rkf_r2 = []
rkf_rmse = []

for tr, te in rkf.split(X):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    best_estimator.fit(X_tr, y_tr)
    y_pr = best_estimator.predict(X_te)
    rkf_r2.append(r2_score(y_te, y_pr))
    rkf_rmse.append(rmse(y_te, y_pr))

metrics['rkf_mean_r2'] = float(np.mean(rkf_r2))
metrics['rkf_std_r2'] = float(np.std(rkf_r2))
metrics['rkf_mean_rmse'] = float(np.mean(rkf_rmse))
metrics['rkf_std_rmse'] = float(np.std(rkf_rmse))
print(f'Repeated KFold R2 mean={metrics["rkf_mean_r2"]:.4f} std={metrics["rkf_std_r2"]:.4f}')

# Save metrics
(Path(OUTDIR) / 'metrics.json').write_text(json.dumps(metrics, indent=2))

# Save feature importances
fi = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': best_estimator.feature_importances_
}).sort_values('Importance', ascending=False)
fi.to_csv(OUTDIR / 'feature_importances.csv', index=False)
top_features = fi['Feature'].head(min(6, len(fi))).tolist()

# ---------------------- SHAP -------------------------------
try:
    import shap
    explainer = shap.TreeExplainer(best_estimator)
    shap_values = explainer.shap_values(X_test)

    # Bar plot
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'shap_bar.png', bbox_inches='tight')
    plt.close()

    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'shap_summary.png', bbox_inches='tight')
    plt.close()

    # Dependence plots for top features
    # For RF + shap_values, index columns by integer position
    for f in top_features:
        try:
            col_idx = list(X_test.columns).index(f)
            shap.dependence_plot(col_idx, shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(OUTDIR / f'shap_dependence_{f}.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print('SHAP dependence failed for', f, e)
except Exception as e:
    print('SHAP failed or not installed:', e)

# ---------------- Permutation Importance -------------------
try:
    pi = permutation_importance(best_estimator, X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1)
    pi_df = pd.DataFrame({'Feature': FEATURES, 'PI_Mean': pi.importances_mean, 'PI_Std': pi.importances_std})\
        .sort_values('PI_Mean', ascending=False)
    pi_df.to_csv(OUTDIR / 'permutation_importance.csv', index=False)
except Exception as e:
    print('Permutation importance failed:', e)

# ---------------------- PDP plots --------------------------
try:
    for f in top_features[:3]:
        fig, ax = plt.subplots(figsize=(6,4))
        PartialDependenceDisplay.from_estimator(best_estimator, X, [f], ax=ax)
        plt.tight_layout()
        fig.savefig(OUTDIR / f'pdp_{f}.png', bbox_inches='tight')
        plt.close(fig)
except Exception as e:
    print('PDP failed:', e)

# ---------------- Save fold predictions --------------------
fold_results = []
for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    best_estimator.fit(X_tr, y_tr)
    y_pr = best_estimator.predict(X_te)
    df_fold = pd.DataFrame({
        'Actual_PM2.5': y_te.values,
        'Predicted_PM2.5': y_pr
    })
    # Include optional columns if present
    for extra in ['Route', 'Points_id', 'Season']:
        if extra in data.columns:
            df_fold[extra] = data.iloc[test_idx][extra].values
    fold_results.append(df_fold)

cv_results = pd.concat(fold_results, axis=0)
cv_results.to_excel(OUTDIR / 'rf_10fold_predictions.xlsx', index=False)

# ---------------- Final save -------------------------------
conf = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'seed': SEED,
    'features': FEATURES,
    'outlier_method': OUTLIER_METHOD,
    'iqr_factor': IQR_FACTOR,
    'pm25_threshold': PM25_THRESHOLD,
    'used_optuna': used_optuna,
    'best_params': best_params
}
(Path(OUTDIR) / 'run_config.json').write_text(json.dumps(conf, indent=2))

print('All done. Outputs saved to', OUTDIR.resolve())
