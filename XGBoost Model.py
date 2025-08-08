
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_excel('all-50m.xlsx')

# Remove outliers
e = 300
print(f"Rows with PM2.5 > {e}:", (data["PM2.5"] > e).sum())
data = data[data["PM2.5"] <= e]

# Define features and target
features = [
    'Fixed', "Avg_Visibility", "Ambient Temperature °C", "Total - Office - 150", "Total - Office - 50",
    "Dis-From - fuel - 0", "Ambient Humidity %RH", "Avg_Dew_Point", "Avg_Wind", 'Season',
    "Dis-From - subway - 0"
]
features = [col for col in features if col in data.columns]
target = "PM2.5"

# Fill missing values
data = data.fillna(0)
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("80/20 Split - R2:", r2_score(y_test, y_pred))
print("80/20 Split - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Grid search tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Best Model R2:", r2_score(y_test, y_pred_best))
print("Best Model RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))

# SHAP Analysis
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

shap.plots.bar(shap_values, show=False)
plt.savefig("xgb_shap_feature_importance_bar.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("xgb_shap_summary_plot.png", bbox_inches="tight")
plt.close()
print("SHAP plots saved.")

# 10-Fold CV with predictions
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_results = []
actual = []
predicted = []

for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_tr, y_tr)
    y_pr = model.predict(X_te)
    df_result = pd.DataFrame({
        'Route': data.iloc[test_idx]['Route'].values,
        'Points_id': data.iloc[test_idx]['Points_id'].values,
        'Season': data.iloc[test_idx]['Season'].values,
        'Actual_PM2.5': y_te.values,
        'Predicted_PM2.5': y_pr
    })
    all_results.append(df_result)
    actual.extend(y_te)
    predicted.extend(y_pr)

cv_results = pd.concat(all_results, axis=0)
cv_results.to_excel("XGB_10fold_results.xlsx", index=False)

plt.figure(figsize=(10,6))
plt.scatter(actual, predicted, alpha=0.6, color='b')
plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("XGBoost - Actual vs Predicted (10-Fold CV)")
plt.legend()
plt.grid(True)
plt.show()

# Feature importance
importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top10 = importance_df.head(10)
print("Top 10 Features:")
print(top10)
top10.to_excel("XGB_feature_importance.xlsx", index=False)

# Repeated K-Fold Cross-Validation
rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)
all_r2 = []
all_rmse = []

for train_idx, test_idx in rkf.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_tr, y_tr)
    y_pr = model.predict(X_te)
    all_r2.append(r2_score(y_te, y_pr))
    all_rmse.append(np.sqrt(mean_squared_error(y_te, y_pr)))

print(f"XGBoost Repeated K-Fold - Mean R2: {np.mean(all_r2):.4f}, Std: {np.std(all_r2):.4f}")
print(f"XGBoost Repeated K-Fold - Mean RMSE: {np.mean(all_rmse):.4f}, Std: {np.std(all_rmse):.4f}")
