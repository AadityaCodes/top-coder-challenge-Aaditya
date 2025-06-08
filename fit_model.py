import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# Load public cases
with open("public_cases.json", "r") as f:
    public_cases = json.load(f)

# Prepare data
X = [[case["input"]["trip_duration_days"], 
      case["input"]["miles_traveled"], 
      case["input"]["total_receipts_amount"]] for case in public_cases]
y = [case["expected_output"] for case in public_cases]

# Define model and hyperparameter grid
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search.fit(X, y)

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "rf_model.joblib")

# Print best parameters and score for reference
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best MAE: {-grid_search.best_score_:.2f}")