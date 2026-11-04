"""
Workshop: K-Means clustering (Introduccion a la IA, Taller para la Nota IV)

Keep this file in the same folder as datos.csv, then run:
    python kmeans_taller.py

K-Means repeats two steps until nothing changes:
    1. ASSIGN: each sample joins the group of its closest centroid.
    2. UPDATE: each centroid moves to the average of the samples in its group.

One "iteration" means one ASSIGN step plus one UPDATE step.

The answers come from our own implementation, because the workshop asks us to
implement the algorithm and asks for the centroids after specific iterations.
Scikit-learn then runs on the same data to check that the results match.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Print whole tables, with 3 decimals, on a single line.
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.3f}".format)


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
# Look for datos.csv next to this script, wherever you run it from.
data_path = Path(__file__).parent / "datos.csv"

# index_col="ID" makes the ID column the row label, so it isn't used as a number.
data = pd.read_csv(data_path, index_col="ID") #ID identifies each sample (for example, M020), but it isn’t one of the measurements we want K-Means to group samples by. K-Means calculates distances from the feature values, so including an ID as a feature would distort those distances.

# X is a table of plain numbers: 130 rows (samples) x 7 columns (features).
X = data.to_numpy()
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.\n")


# ---------------------------------------------------------------------------
# 2. Choose the starting centroids given by the workshop
# ---------------------------------------------------------------------------
# Row order matters: M020 is C1, M056 is C2 and M116 is C3.
initial_centroids = data.loc[["M020", "M056", "M116"]].to_numpy() #labels are names attached to rows or columns
#.iloc, which selects by numeric position, .loc selects by label.


# ---------------------------------------------------------------------------
# 3. K-Means implemented step by step
# ---------------------------------------------------------------------------
def closest_centroid(samples, centroids):
    """Return, for each sample, the position (0, 1 or 2) of its closest centroid."""
    # Add an empty dimension so NumPy compares every sample with every centroid.
    # differences has the shape (samples, centroids, features).
    differences = samples[:, np.newaxis, :] - centroids

    # Euclidean distance: the square root of the sum of squared differences.
    distances = np.sqrt((differences ** 2).sum(axis=2))

    # argmin picks the smallest distance in each row.
    return distances.argmin(axis=1)


def kmeans(samples, start_centroids, max_iterations=100):
    """Run K-Means and return the final labels and the centroids of every iteration."""
    centroids = start_centroids.copy()

    # history[0] holds the starting centroids, history[1] the centroids after
    # iteration 1, history[2] the centroids after iteration 2, and so on.
    history = [centroids.copy()]

    for _ in range(max_iterations):
        # Step 1, ASSIGN: find the group of each sample.
        labels = closest_centroid(samples, centroids)

        # Step 2, UPDATE: average the samples in each group.
        new_centroids = centroids.copy()
        for group in range(len(centroids)):
            members = samples[labels == group]
            if len(members) > 0:  # An empty group keeps its old centroid.
                new_centroids[group] = members.mean(axis=0)

        history.append(new_centroids.copy())

        # If no centroid moved, the assignments can't change again: we're done.
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, history


labels, history = kmeans(X, initial_centroids)
final_centroids = history[-1]

# The last iteration only confirms that nothing moved.
print(f"Stopped after {len(history) - 1} iterations "
      f"(the last one confirmed that the centroids had stopped moving).\n")


# ---------------------------------------------------------------------------
# 4. Answer the workshop questions
# ---------------------------------------------------------------------------
feature_names = [f"F{i}" for i in range(1, X.shape[1] + 1)]
group_names = ["Group 1", "Group 2", "Group 3"]


def show_centroids(centroids, names=group_names):
    """Turn raw centroid numbers into a labeled table."""
    return pd.DataFrame(centroids, index=names, columns=feature_names)


print("QUESTION 1: How many samples are in each group?")
for group, name in enumerate(group_names):
    print(f"  {name}: {(labels == group).sum()} samples")

print("\nQUESTION 2: Final centroids")
print(show_centroids(final_centroids))

print("\nQUESTION 3: Third centroid after one iteration")
# history[1] = after iteration 1; [2] = the third centroid (counting starts at 0).
print(show_centroids(history[1][[2]], ["C3"]))

print("\nQUESTION 4: First centroid after two iterations")
# history[2] = after iteration 2; [0] = the first centroid.
print(show_centroids(history[2][[0]], ["C1"]))

print("\nQUESTION 5: Groups for the new samples")
new_samples = pd.DataFrame(
    [
        [-2.3, -2.1, 0.2, 1.4, -1.2, 0.7, 2.1],
        [-3.0, -1.8, -0.3, 1.8, -0.8, 0.3, 1.7],
        [2.3, -0.8, 2.4, -1.2, 1.7, 2.1, -0.4],
        [1.7, -1.3, 2.8, -1.7, 1.3, 1.8, -0.8],
        [0.2, 2.7, -1.8, 2.2, 2.4, -1.2, 0.8],
        [-0.4, 2.2, -2.3, 1.7, 2.8, -1.7, 1.3],
    ],
    index=["N001", "N002", "N003", "N004", "N005", "N006"],
    columns=feature_names,
)
# New samples only need the ASSIGN step. The final centroids don't move.
new_labels = closest_centroid(new_samples.to_numpy(), final_centroids)
for sample_id, group in zip(new_samples.index, new_labels):
    print(f"  {sample_id}: {group_names[group]}")


# ---------------------------------------------------------------------------
# 5. Check everything with scikit-learn
# ---------------------------------------------------------------------------
def sklearn_kmeans(max_iter):
    """Run scikit-learn's K-Means from the workshop's starting centroids."""
    # init=initial_centroids : use our centroids instead of random ones
    # n_init=1               : start only once, since the start is fixed
    # algorithm="lloyd"      : the classic version of the algorithm used above
    # tol=0                  : stop only when the result is exactly stable
    model = KMeans(
        n_clusters=3,
        init=initial_centroids,
        n_init=1,
        algorithm="lloyd",
        max_iter=max_iter,
        tol=0,
    )
    return model.fit(X)


full_model = sklearn_kmeans(max_iter=100)

checks = {
    "Same group for every sample": np.array_equal(labels, full_model.labels_),
    "Same final centroids": np.allclose(final_centroids, full_model.cluster_centers_),
    "Same centroids after 1 iteration": np.allclose(
        history[1], sklearn_kmeans(max_iter=1).cluster_centers_
    ),
    "Same centroids after 2 iterations": np.allclose(
        history[2], sklearn_kmeans(max_iter=2).cluster_centers_
    ),
    "Same groups for the new samples": np.array_equal(
        new_labels, full_model.predict(new_samples.to_numpy())
    ),
}

print("\nCHECK AGAINST SCIKIT-LEARN")
for description, passed in checks.items():
    print(f"  {description}: {'OK' if passed else 'DIFFERENT'}")