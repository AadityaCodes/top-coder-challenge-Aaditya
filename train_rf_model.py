import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_text
import joblib
import numpy as np

# Load public cases
with open("public_cases.json", "r") as f:
    public_cases = json.load(f)

# Prepare data
X = [[case["input"]["trip_duration_days"], 
      case["input"]["miles_traveled"], 
      case["input"]["total_receipts_amount"]] for case in public_cases]
y = [case["expected_output"] for case in public_cases]

# Define hyperparameter distribution
param_dist = {
    "n_estimators": [100, 200, 300, 500, 600],
    "max_depth": [3, 5, 7, 8, 10, 12, 15, 20, None],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "min_samples_split": [2, 5, 8, 10],
    "max_features": ["auto", "sqrt", 0.5, 0.7, 1.0]
}

# Initialize model
rf = RandomForestRegressor(random_state=42)

# Perform randomized search
random_search = RandomizedSearchCV(
    rf, 
    param_distributions=param_dist, 
    n_iter=100,  # Try 100 combinations
    cv=5, 
    scoring="neg_mean_absolute_error", 
    n_jobs=-1, 
    random_state=42
)
random_search.fit(X, y)

# Save best model
best_model = random_search.best_estimator_
joblib.dump(best_model, "rf_model_2.joblib")

# Print diagnostics
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV MAE: {-random_search.best_score_:.2f}")

# Check training error
y_pred = best_model.predict(X)
train_mae = mean_absolute_error(y, y_pred)
print(f"Training MAE: {train_mae:.2f}")

# Mispredicted cases
errors = np.abs(np.array(y) - y_pred)
mispredicted = [(i, public_cases[i]["input"], y[i], y_pred[i], errors[i]) for i in range(len(y)) if errors[i] > 0.01]
print(f"\nMispredicted cases (error > $0.01): {len(mispredicted)}")
for idx, inputs, true, pred, error in mispredicted[:5]:
    print(f"Case {idx}: Inputs={inputs}, True={true:.2f}, Pred={pred:.2f}, Error={error:.2f}")

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(["trip_duration_days", "miles_traveled", "total_receipts_amount"], best_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Inspect first tree for rule insights
print("\nSample Tree Rules (Tree 0):\n")
tree_rules = export_text(best_model.estimators_[0], feature_names=["trip_duration_days", "miles_traveled", "total_receipts_amount"])
print(tree_rules)