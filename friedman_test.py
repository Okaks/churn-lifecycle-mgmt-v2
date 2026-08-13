import numpy as np
from scipy.stats import friedmanchisquare, rankdata

models = ["LightGBM", "XGBoost", "Random Forest", "Logistic Regression"]
scores = np.array([
    [0.8562, 0.9181, 0.7967, 0.8324, 0.9372],
    [0.8561, 0.9149, 0.7971, 0.8322, 0.9362],
    [0.8526, 0.9053, 0.7931, 0.8276, 0.9236],
    [0.8135, 0.8295, 0.7423, 0.7673, 0.9059],
])

stat, p = friedmanchisquare(*scores)
ranks = np.array([rankdata(-scores[:, j]) for j in range(scores.shape[1])])
mean_ranks = ranks.mean(axis=0)
CD = 2.569 * np.sqrt(4 * 5 / (6 * 5))

print(f"Friedman chi-square = {stat:.4f}, df = 3, p = {p:.4f}")
for m, r in zip(models, mean_ranks):
    print(f"{m:22s} mean rank {r:.2f}")
print(f"Nemenyi critical difference = {CD:.3f}")

for i in range(4):
    for j in range(i + 1, 4):
        d = abs(mean_ranks[i] - mean_ranks[j])
        print(f"{models[i]} vs {models[j]}: {d:.2f} "
              f"{'significant' if d > CD else 'not significant'}")