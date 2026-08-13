import joblib
from pathlib import Path

path = Path("models") / "churn_model.joblib"
model = joblib.load(path)

keys = ["boosting_type", "objective", "n_estimators", "learning_rate",
        "num_leaves", "max_depth", "min_child_samples", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "class_weight",
        "random_state"]

params = model.get_params()
for k in keys:
    print(f"{k}: {params.get(k)}")
print("total trees:", model.booster_.num_trees())