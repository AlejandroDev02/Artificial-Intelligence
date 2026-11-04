"""
Reusable K-Means tool: uses the elbow method, answers workshop-style questions,
and checks its own results.

HOW TO USE IT
    1. Put this file in the same folder as your CSV.
    2. Change the values in the SETTINGS section below. Nothing else needs editing.
    3. Run:  python kmeans_tool.py

Required: pip install numpy pandas scikit-learn
Optional: pip install matplotlib   (also saves the elbow chart as an image)
"""

import csv
import re
import sys
import warnings
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans, kmeans_plusplus
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import silhouette_score
except ImportError as error:
    sys.exit(f"Missing library: {error.name}\n"
             "Install what's needed with:  pip install numpy pandas scikit-learn")


# ===========================================================================
# SETTINGS: change these values for each problem
# ===========================================================================

# --- Your data ---------------------------------------------------------------
CSV_FILE = "datos.csv"      # File name (same folder as this script) or full path.
HAS_HEADER = "auto"         # "auto", True or False: is the first row column names?
ID_COLUMN = "auto"          # "auto", None (no IDs), a column name, or a column
                            # number where 1 is the first column.
SCALE_FEATURES = False      # True when features use very different units, such as
                            # age and salary. Answers stay in the original units.

# --- Number of groups ----------------------------------------------------------
NUM_CLUSTERS = 3            # A whole number, or "elbow" to use the elbow's suggestion.
MAX_K_TO_TEST = 10          # The elbow method tries k = 1, 2, ... up to this value.
                            # It can suggest at most this value minus 1.

# --- Starting centroids --------------------------------------------------------
# Option A: name samples. Their order sets the group numbers. Use row numbers,
#           starting at 1, if the file has no IDs.
# Option B: give coordinates, such as [[1.0, 2.0], [3.0, 4.0]].
# Option C: None. The script tries several starts and keeps the best result.
INITIAL_CENTROIDS = ["M020", "M056", "M116"]
RANDOM_SEED = 42            # Option C gives repeatable results with the same seed.
NUM_RESTARTS = 10           # How many starts Option C and the elbow method try.

# --- Questions to answer ---------------------------------------------------------
# (centroid number, iteration): "third centroid after one iteration" is (3, 1).
CENTROIDS_TO_SHOW = [(3, 1), (1, 2)]

# New samples to place in groups, as {"name": [one value per feature]}. Use {} for none.
NEW_SAMPLES = {
    "N001": [-2.3, -2.1, 0.2, 1.4, -1.2, 0.7, 2.1],
    "N002": [-3.0, -1.8, -0.3, 1.8, -0.8, 0.3, 1.7],
    "N003": [2.3, -0.8, 2.4, -1.2, 1.7, 2.1, -0.4],
    "N004": [1.7, -1.3, 2.8, -1.7, 1.3, 1.8, -0.8],
    "N005": [0.2, 2.7, -1.8, 2.2, 2.4, -1.2, 0.8],
    "N006": [-0.4, 2.2, -2.3, 1.7, 2.8, -1.7, 1.3],
}

# --- Optional comparison with answers you already have -----------------------------
# Type answers from your teacher or a hand calculation, and the script marks each
# one MATCH or DIFFERENT. Use None or {} for anything you can't compare.
# These are the Workshop 4 answers. CHANGE OR CLEAR THEM WHEN YOU CHANGE THE DATA.
EXPECTED_GROUP_SIZES = [30, 60, 40]     # Samples in Group 1, Group 2, ...
EXPECTED_FINAL_CENTROIDS = [
    [-2.477, -1.982, -0.096, 1.527, -1.147, 0.450, 2.019],
    [2.075, -0.997, 2.501, -1.389, 1.433, 2.015, -0.774],
    [-0.073, 2.393, -2.039, 1.696, 2.548, -1.425, 1.090],
]
EXPECTED_CENTROIDS_AFTER = {            # {(centroid number, iteration): [values]}
    (3, 1): [-0.073, 2.393, -2.039, 1.696, 2.548, -1.425, 1.090],
    (1, 2): [-2.477, -1.982, -0.096, 1.527, -1.147, 0.450, 2.019],
}
EXPECTED_NEW_SAMPLE_GROUPS = {          # {"name": group number}
    "N001": 1, "N002": 1, "N003": 2, "N004": 2, "N005": 3, "N006": 3,
}
ANSWER_TOLERANCE = 0.001    # Largest allowed difference when comparing typed numbers.

# --- Output files -----------------------------------------------------------------
ELBOW_CHART_FILE = "elbow.png"  # Saved only if matplotlib is installed. None to skip.
GROUPS_OUTPUT_FILE = None       # For example "groups.csv" to save each sample's group.
MAX_ITERATIONS = 300            # Safety limit for one K-Means run.

# ===========================================================================
# END OF SETTINGS
# ===========================================================================


# ---------------------------------------------------------------------------
# DATA READING MODULE
# ---------------------------------------------------------------------------
MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none", "-", "?"}
ID_NAME = re.compile(
    r"^(id|ids|identifier|identificaci(o|\u00f3)n|index|sample|sample[ _-]?id|muestra"
    r"|c(o|\u00f3)digo|code|key)$"
    r"|[ _-]id$|^id[ _-]",
    re.IGNORECASE,
)
MIN_NUMERIC_SHARE = 0.8     # A column counts as numeric if 80% of its values are numbers.
ELBOW_CLEAR_RATIO = 3       # Below this ratio, the elbow is reported as weak.


def stop(message):
    """End the program with a readable message instead of a long Python error."""
    sys.exit(f"\nERROR: {message}")


def read_text(path):
    """Read a text file. UTF-8 is the most common encoding, cp1252 is older Excel."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    stop(f"Couldn't read the text encoding of {path.name}. Save it as CSV UTF-8.")


def parse_rows(lines, separator):
    """Split lines into cells, ignoring blank cells beyond the real last column."""
    rows = list(csv.reader(lines, delimiter=separator))
    width = 0
    for row in rows:
        filled = [position for position, cell in enumerate(row) if cell.strip()]
        width = max(width, filled[-1] + 1 if filled else 0)
    return [row[:width] for row in rows]


def split_rows(text):
    """Find the separator and split the file into rows and cells."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        stop("The CSV file is empty.")

    # Semicolons, tabs and pipes are tried before commas. Spanish Excel writes
    # "1,5;2,5", where the commas are decimal marks rather than separators.
    for separator in (";", "\t", "|", ","):
        rows = parse_rows(lines, separator)
        widths = {len(row) for row in rows}
        if len(widths) == 1 and widths != {1}:
            return rows, separator

    # Rows of different lengths: report the first row that doesn't fit.
    for separator in (";", "\t", "|", ","):
        rows = parse_rows(lines, separator)
        if len(rows[0]) > 1:
            for number, row in enumerate(rows, start=1):
                if len(row) != len(rows[0]):
                    stop(f"Row {number} has {len(row)} columns, but row 1 has "
                         f"{len(rows[0])}. Every row needs the same number.")
    return [[line] for line in lines], ","     # The file has a single column.


def to_number(cell, separator):
    """Turn a text cell into a number. Missing or non-numeric cells become NaN."""
    cell = cell.strip()
    if cell.lower() in MISSING_MARKERS:
        return float("nan")
    if separator != ",":
        cell = cell.replace(",", ".")
    try:
        number = float(cell)
    except ValueError:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def is_missing(cell):
    return cell.strip().lower() in MISSING_MARKERS


def looks_numeric(cells, separator):
    """True if nearly all non-missing cells in this column are numbers."""
    filled = [cell for cell in cells if not is_missing(cell)]
    if not filled:
        return False
    numeric = sum(not np.isnan(to_number(cell, separator)) for cell in filled)
    return numeric / len(filled) >= MIN_NUMERIC_SHARE


def detect_header(rows, separator):
    """A first row is a header if it has text where the data below has numbers."""
    if len(rows) < 2:
        return False
    first_row, body = rows[0], rows[1:]
    # An ID-style name counts only if the values below don't look the same way,
    # so a first data row such as "id-001, 1.5, 2.5" isn't mistaken for a header.
    if ID_NAME.search(first_row[0].strip()) and not any(
            ID_NAME.search(row[0].strip()) for row in body):
        return True
    for column, cell in enumerate(first_row):
        below = [row[column] for row in body]
        first_is_text = not is_missing(cell) and np.isnan(to_number(cell, separator))
        if first_is_text and looks_numeric(below, separator):
            return True
    return False


def looks_like_row_counter(cells):
    """True for a column such as 1, 2, 3, ... in any order, starting at 0 or 1."""
    try:
        numbers = sorted(int(cell.strip()) for cell in cells)
    except ValueError:
        return False
    start = numbers[0]
    return start in (0, 1) and numbers == list(range(start, start + len(numbers)))


def detect_id_column(names, columns, separator):
    """Return the position of the ID column, or None if there isn't one."""
    # 1. A column with an ID-style name.
    for position, name in enumerate(names):
        if ID_NAME.search(name):
            return position
    # 2. The first text column where every value is different, such as M001, M002.
    for position, cells in enumerate(columns):
        if not looks_numeric(cells, separator) and len(set(cells)) == len(cells):
            return position
    # 3. A first column that just counts the rows.
    if len(columns) > 1 and looks_like_row_counter(columns[0]):
        return 0
    return None


def choose_id_column(setting, names, columns, separator):
    """Apply the ID_COLUMN setting, or detect the column when it's "auto"."""
    if setting is None:
        return None
    if setting == "auto":
        return detect_id_column(names, columns, separator)
    if isinstance(setting, int):
        if 1 <= setting <= len(names):
            return setting - 1
        stop(f"ID_COLUMN = {setting}, but the file has {len(names)} columns.")
    if setting in names:
        return names.index(setting)
    stop(f'ID_COLUMN = "{setting}" was not found. Columns: {", ".join(names)}')


def read_samples(path, has_header="auto", id_column="auto"):
    """
    Read a CSV with or without a header row and an ID column.

    Returns the numeric features as a DataFrame, using sample IDs as row labels,
    and a list of notes describing what was detected.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    if not path.exists():
        stop(f"File not found: {path}")

    rows, separator = split_rows(read_text(path))

    header = detect_header(rows, separator) if has_header == "auto" else bool(has_header)
    body = rows[1:] if header else rows
    if not body:
        stop("The file has a header but no data rows.")

    # Remove columns that are completely empty, often caused by a trailing separator.
    width = len(rows[0])
    keep = [c for c in range(width) if not all(is_missing(row[c]) for row in body)]
    names = [rows[0][c].strip() if header else "" for c in keep]
    columns = [[row[c].strip() for row in body] for c in keep]

    id_position = choose_id_column(id_column, names, columns, separator)
    ids = columns[id_position] if id_position is not None else None
    if ids is None:
        ids = [str(number) for number in range(1, len(body) + 1)]

    # Keep numeric columns as features. Other text columns can't be averaged.
    features, feature_names, skipped = [], [], []
    for position, cells in enumerate(columns):
        if position == id_position:
            continue
        label = names[position] or f"F{len(feature_names) + 1}"
        if looks_numeric(cells, separator):
            features.append([to_number(cell, separator) for cell in cells])
            feature_names.append(label)
        else:
            skipped.append(label)
    if not features:
        stop("No numeric feature columns were found. Check that the file has number "
             "columns and that no title line sits above the data.")

    table = pd.DataFrame(np.array(features).T, index=ids, columns=feature_names)
    table.index.name = "ID"

    # K-Means needs every value, so rows with gaps are removed and reported.
    incomplete = table[table.isna().any(axis=1)].index.tolist()
    table = table.dropna()

    id_text = "none, using row numbers"
    if id_position is not None:
        id_text = names[id_position] or f"column {id_position + 1}"
    separator_name = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[separator]
    notes = [
        f"File: {path.name} ({separator_name}-separated)",
        f"Header row: {'yes' if header else 'no'}",
        f"ID column: {id_text}",
        f"Used: {len(table)} samples, {len(feature_names)} features "
        f"({', '.join(feature_names[:8])}{', ...' if len(feature_names) > 8 else ''})",
    ]
    if skipped:
        notes.append(f"WARNING: skipped non-numeric columns: {', '.join(skipped)}")
    if incomplete:
        shown = ", ".join(incomplete[:10]) + (", ..." if len(incomplete) > 10 else "")
        notes.append(f"WARNING: removed {len(incomplete)} rows with missing values: {shown}")
    if table.index.duplicated().any():
        notes.append("WARNING: some IDs repeat, so naming those samples is ambiguous.")
    for name in feature_names:
        values = table[name]
        if len(values) > 2 and values.is_unique and (values == values.round()).all():
            notes.append(f'WARNING: "{name}" has unique whole numbers and could be an '
                         'ID. If so, set ID_COLUMN to its name.')
    return table, notes


# ---------------------------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------------------------
def squared_distances(samples, centroids):
    """Squared distance from every sample (rows) to every centroid (columns)."""
    result = np.empty((len(samples), len(centroids)))
    for position, centroid in enumerate(centroids):
        result[:, position] = ((samples - centroid) ** 2).sum(axis=1)
    return result


def closest_centroid(samples, centroids):
    """For each sample, return the position of its closest centroid."""
    return squared_distances(samples, centroids).argmin(axis=1)


def inertia_of(samples, centroids, labels):
    """Inertia: the sum of squared distances from samples to their own centroids."""
    return ((samples - centroids[labels]) ** 2).sum()


def run_kmeans(samples, start_centroids, max_iterations=300):
    """
    One K-Means run, written out step by step. history[0] holds the starting
    centroids and history[n] the centroids after iteration n.
    """
    centroids = np.array(start_centroids, dtype=float)
    history = [centroids.copy()]
    inertias = []
    empty_group_seen = False
    tie_seen = False
    converged = False

    for _ in range(max_iterations):
        # Step 1, ASSIGN: each sample joins its closest centroid's group.
        distances = squared_distances(samples, centroids)
        labels = distances.argmin(axis=1)

        # Note exact ties. argmin picks the lower group number, while another
        # program's rounding can send the sample to the other group.
        if len(centroids) > 1:
            two_closest = np.partition(distances, 1, axis=1)
            gap = two_closest[:, 1] - two_closest[:, 0]
            tie_seen = tie_seen or bool(np.any(gap <= 1e-9 * (1 + two_closest[:, 0])))

        # Step 2, UPDATE: each centroid moves to the average of its group.
        new_centroids = centroids.copy()
        for group in range(len(centroids)):
            members = samples[labels == group]
            if len(members) > 0:
                new_centroids[group] = members.mean(axis=0)
            else:
                empty_group_seen = True     # An empty group keeps its centroid.

        history.append(new_centroids.copy())
        inertias.append(inertia_of(samples, new_centroids, labels))

        if np.array_equal(new_centroids, centroids):
            converged = True
            break
        centroids = new_centroids

    return {
        "start": history[0],
        "centroids": history[-1],
        "labels": labels,
        "history": history,
        "inertia": inertias[-1],
        "inertias": inertias,
        "empty_group_seen": empty_group_seen,
        "tie_seen": tie_seen,
        "converged": converged,
    }


def sklearn_run(samples, start, k, max_iter):
    """Run scikit-learn with the same data, start and method as run_kmeans."""
    model = KMeans(n_clusters=k, init=np.array(start), n_init=1,
                   algorithm="lloyd", max_iter=max_iter, tol=0)
    with warnings.catch_warnings():
        # Repeated data points are reported by this script's own checks.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(samples)


def search_best_start(samples, k, seed, restarts, max_iterations):
    """
    Try several k-means++ starts and return the one with the lowest inertia.
    Scikit-learn does this search because it's much faster on large files.
    """
    best, seed_generator = None, np.random.RandomState(seed)
    for _ in range(restarts):
        start_seed = seed_generator.randint(0, 2**31 - 1)
        start, _ = kmeans_plusplus(samples, k, random_state=start_seed)
        model = sklearn_run(samples, start, k, max_iterations)
        if best is None or model.inertia_ < best["inertia"]:
            best = {"start": start, "inertia": model.inertia_, "labels": model.labels_}
    return best


# ---------------------------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------------------------
def find_elbow(k_values, inertias):
    """
    The elbow is where adding one more group stops helping. For each k, compare
    the inertia drop gained by reaching k with the drop gained by the next group.
    The k with the largest ratio is the suggestion.

    Returns the suggested k and its ratio. It can't suggest the first k, which has
    no earlier drop, or the last k, which has no next drop.
    """
    drops = [before - after for before, after in zip(inertias, inertias[1:])]
    best_k, best_ratio = None, -np.inf
    for position in range(len(drops) - 1):
        ratio = drops[position] / max(drops[position + 1], 1e-12)
        if ratio > best_ratio:
            best_k, best_ratio = k_values[position + 1], ratio
    return best_k, best_ratio


def run_elbow(samples, max_k, seed, restarts, max_iterations):
    """Run K-Means for each k and measure how well each number of groups fits."""
    distinct_points = len(np.unique(samples, axis=0))
    top_k = min(max_k, distinct_points - 1, len(samples) - 1)
    if top_k < 3:
        return None

    print(f"Testing k = 1 to {top_k}, with {restarts} starts for each k...\n", flush=True)
    silhouette_sample = min(len(samples), 5000)     # Keeps large files fast.
    rows = []
    for k in range(1, top_k + 1):
        result = search_best_start(samples, k, seed, restarts, max_iterations)
        silhouette = float("nan")
        if k > 1 and len(np.unique(result["labels"])) > 1:
            silhouette = silhouette_score(
                samples, result["labels"],
                sample_size=silhouette_sample, random_state=seed,
            )
        rows.append({
            "k": k, "inertia": result["inertia"], "silhouette": silhouette,
            "start": result["start"],
        })

    k_values = [row["k"] for row in rows]
    inertias = [row["inertia"] for row in rows]
    elbow_k, elbow_ratio = find_elbow(k_values, inertias)
    return {
        "rows": rows,
        "elbow_k": elbow_k,
        "elbow_ratio": elbow_ratio,
        "k_values": k_values,
        "inertias": inertias,
    }


def print_elbow(elbow, chosen_k):
    """Print the elbow results as a table with a text chart."""
    print("k    inertia  drop from previous k  silhouette  chart")
    inertias, k_values = elbow["inertias"], elbow["k_values"]
    largest = max(inertias) or 1
    previous = None
    for row in elbow["rows"]:
        k, inertia = row["k"], row["inertia"]
        drop = "-" if previous is None else f"{previous - inertia:.3f}"
        silhouette = "-" if np.isnan(row["silhouette"]) else f"{row['silhouette']:.3f}"
        bar = "#" * max(1, round(30 * inertia / largest))
        mark = "  <- elbow" if k == elbow["elbow_k"] else ""
        print(f"{k:<2} {inertia:>10.3f}  {drop:>20}  {silhouette:>10}  {bar}{mark}")
        previous = inertia

    usable = [r for r in elbow["rows"] if not np.isnan(r["silhouette"])]
    top = max(usable, key=lambda r: r["silhouette"]) if usable else None
    elbow_k, ratio = elbow["elbow_k"], elbow["elbow_ratio"]
    position = k_values.index(elbow_k)
    drop_here = inertias[position - 1] - inertias[position]
    drop_next = inertias[position] - inertias[position + 1]
    print(f"\nElbow suggestion: k = {elbow_k}")
    print(f"  Going to k = {elbow_k} lowered inertia by {drop_here:.3f}, but going to "
          f"k = {elbow_k + 1} lowered it by only {drop_next:.3f}: {ratio:.1f} times less.")
    if ratio < ELBOW_CLEAR_RATIO:
        print("NOTE: this bend isn't sharp, so the elbow is weak evidence here.")
    if elbow_k == k_values[-2]:
        print("NOTE: this is the largest k the method could test. Try a larger MAX_K_TO_TEST.")
    if top is not None:
        verdict = ("strong" if top["silhouette"] > 0.7 else
                   "reasonable" if top["silhouette"] > 0.5 else
                   "weak" if top["silhouette"] > 0.25 else "little or none")
        print(f"Best silhouette:  k = {top['k']} (score {top['silhouette']:.3f}; "
              f"evidence of real groups: {verdict})")
        if top["k"] != elbow["elbow_k"]:
            print("NOTE: these two measures disagree. Treat k as a judgment call.")
        if top["silhouette"] <= 0.25:
            print("NOTE: this data may not have clear groups, whatever k you choose.")
    if elbow["elbow_k"] != chosen_k:
        print(f"NOTE: this run uses k = {chosen_k}, which comes from your settings.")


def save_elbow_chart(elbow, filename):
    """Save the elbow chart if matplotlib is installed."""
    if not filename:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")       # Saves the file without opening a window.
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chart image skipped because matplotlib isn't installed. "
              "The text chart above shows the same numbers.")
        return

    k_values, inertias = elbow["k_values"], elbow["inertias"]
    elbow_k = elbow["elbow_k"]
    elbow_inertia = inertias[k_values.index(elbow_k)]

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(k_values, inertias, color="#2F6DB5", linewidth=2,
              marker="o", markersize=7, clip_on=False)
    axis.scatter([elbow_k], [elbow_inertia], s=110, color="#2F6DB5",
                 edgecolor="white", linewidth=2, zorder=3, clip_on=False)
    axis.annotate(f"Elbow: k = {elbow_k}", (elbow_k, elbow_inertia),
                  xytext=(10, 14), textcoords="offset points",
                  fontsize=11, color="#222222")
    axis.set_title("Elbow method", loc="left", fontsize=13, color="#222222")
    axis.set_xlabel("Number of groups (k)", color="#555555")
    axis.set_ylabel("Inertia (lower is tighter)", color="#555555")
    axis.set_xticks(k_values)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#555555", length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#BBBBBB")
    figure.tight_layout()

    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"Elbow chart saved as: {path}")


# ---------------------------------------------------------------------------
# RESULT CHECKS
# ---------------------------------------------------------------------------
class CheckList:
    """Collects PASS, FAIL and SKIPPED results and prints them."""

    def __init__(self):
        self.failed = 0
        self.skipped = 0

    def report(self, description, passed, note=""):
        status = "PASS" if passed else "FAIL"
        self.failed += not passed
        print(f"  {status:<8}{description}{'  (' + note + ')' if note else ''}")

    def compare(self, description, same, tie_seen):
        """Like report, but an exact tie explains a difference without being an error."""
        if not same and tie_seen:
            print(f"  {'NOTE':<8}{description}: no, because an exact tie went the other "
                  "way. Both results are valid.")
        else:
            self.report(description, same)

    def skip(self, description, reason):
        self.skipped += 1
        print(f"  {'SKIPPED':<8}{description}  ({reason})")


def rules_checks(checks, samples, result, k):
    """Rules every correct K-Means result must follow, whatever the data."""
    labels, centroids = result["labels"], result["centroids"]

    sizes = np.bincount(labels, minlength=k)
    checks.report("Group sizes add up to the number of samples",
                  sizes.sum() == len(samples), f"{sizes.sum()} of {len(samples)}")

    distances = squared_distances(samples, centroids)
    own = distances[np.arange(len(samples)), labels]
    checks.report("Every sample is in the group of its closest centroid",
                  np.all(own <= distances.min(axis=1) + 1e-9))

    averages_match = all(
        np.allclose(samples[labels == group].mean(axis=0), centroids[group])
        for group in range(k) if (labels == group).any()
    )
    checks.report("Every centroid equals the average of its group", averages_match)

    never_rose = all(b <= a + 1e-9 for a, b in zip(result["inertias"], result["inertias"][1:]))
    checks.report("Inertia never increased between iterations", never_rose)
    checks.report("The algorithm finished before the iteration limit", result["converged"])


def sklearn_checks(checks, samples, new_samples, result, k, requested, elbow):
    """Run scikit-learn from the same starting centroids and compare the results."""
    start, history = result["start"], result["history"]
    label = "Scikit-learn:"
    if result["empty_group_seen"]:
        checks.skip(f"{label} same results",
                    "a group became empty, and scikit-learn handles that differently")
        return

    tied = result["tie_seen"]
    model = sklearn_run(samples, start, k, MAX_ITERATIONS)
    checks.compare(f"{label} same group for every sample",
                   np.array_equal(result["labels"], model.labels_), tied)
    checks.compare(f"{label} same final centroids",
                   np.allclose(result["centroids"], model.cluster_centers_), tied)
    checks.compare(f"{label} same inertia",
                   np.isclose(result["inertia"], model.inertia_), tied)

    for centroid, iteration in requested:
        step = min(iteration, len(history) - 1)
        if step == 0:
            continue    # Iteration 0 is just the starting centroid.
        partial = sklearn_run(samples, start, k, step)
        checks.compare(f"{label} same C{centroid} after iteration {iteration}",
                       np.allclose(history[step][centroid - 1],
                                   partial.cluster_centers_[centroid - 1]), tied)

    if len(new_samples):
        checks.compare(f"{label} same groups for the new samples",
                       np.array_equal(closest_centroid(new_samples, result["centroids"]),
                                      model.predict(new_samples)), tied)

    if elbow is not None:
        # The elbow numbers come from scikit-learn, so our version checks them here.
        runs = [(row, run_kmeans(samples, row["start"], MAX_ITERATIONS)) for row in elbow["rows"]]
        runs = [(row, run) for row, run in runs if not run["empty_group_seen"]]
        same_curve = all(np.isclose(row["inertia"], run["inertia"]) for row, run in runs)
        checks.compare("Hand-written K-Means: same inertia at every k on the elbow chart",
                       same_curve, any(run["tie_seen"] for _, run in runs))


def expected_checks(checks, sizes, final, history, new_groups):
    """Compare the results with answers typed in the SETTINGS section."""
    tolerance = ANSWER_TOLERANCE

    def close_enough(mine, typed):
        return np.shape(mine) == np.shape(typed) and bool(
            np.all(np.abs(np.array(mine) - np.array(typed)) <= tolerance))

    def show(description, matched, typed):
        word = "MATCH" if matched else "DIFFERENT"
        checks.failed += not matched
        print(f"  {word:<10}{description}")
        if not matched:
            print(f"            you typed: {typed}")

    if EXPECTED_GROUP_SIZES:
        show(f"Group sizes {sizes.tolist()}",
             sizes.tolist() == list(EXPECTED_GROUP_SIZES), EXPECTED_GROUP_SIZES)
    if EXPECTED_FINAL_CENTROIDS:
        show("Final centroids", close_enough(final, EXPECTED_FINAL_CENTROIDS),
             EXPECTED_FINAL_CENTROIDS)
    for (centroid, iteration), typed in (EXPECTED_CENTROIDS_AFTER or {}).items():
        if not 1 <= centroid <= len(final):
            show(f"C{centroid} after iteration {iteration}: this centroid doesn't exist",
                 False, typed)
            continue
        step = min(iteration, len(history) - 1)
        show(f"C{centroid} after iteration {iteration}",
             close_enough(history[step][centroid - 1], typed), typed)
    for name, typed in (EXPECTED_NEW_SAMPLE_GROUPS or {}).items():
        mine = new_groups.get(name)
        show(f"{name} in Group {mine}", mine == typed, f"Group {typed}")


def hand_check(ids, original, samples, centroids, final_original, labels):
    """Print one worked example you can repeat with a calculator or spreadsheet."""
    sample = samples[0]
    scaled_text = " (in scaled values, the ones K-Means used)" if SCALE_FEATURES else ""
    print(f"Distance from sample {ids[0]} to each final centroid{scaled_text}:")
    distances = []
    for number, centroid in enumerate(centroids, start=1):
        parts = " + ".join(f"({a:.3f} - {b:.3f})^2" for a, b in zip(sample[:3], centroid[:3]))
        more = " + ..." if len(sample) > 3 else ""
        distance = np.sqrt(((sample - centroid) ** 2).sum())
        distances.append(distance)
        print(f"  C{number}: sqrt({parts}{more}) = {distance:.3f}")
    closest = int(np.argmin(distances)) + 1
    print(f"  Closest is C{closest}, so {ids[0]} should be in Group {closest}: "
          f"the script placed it in Group {labels[0] + 1}.")

    group = labels[0]
    values = original[labels == group, 0]
    print(f"\nFirst feature of the Group {group + 1} centroid:")
    print(f"  sum of {len(values)} values = {values.sum():.3f}; "
          f"{values.sum():.3f} / {len(values)} = {values.mean():.3f}; "
          f"the script reports {final_original[group, 0]:.3f}.")


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def heading(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def starting_centroids(data, scaled, scaler):
    """Turn the INITIAL_CENTROIDS setting into coordinates. None means Option C."""
    if INITIAL_CENTROIDS is None:
        return None
    if len(INITIAL_CENTROIDS) == 0:
        stop("INITIAL_CENTROIDS is empty. Use None to let the script choose.")

    start = []
    for item in INITIAL_CENTROIDS:
        if isinstance(item, (list, tuple)):
            if len(item) != data.shape[1]:
                stop(f"The starting centroid {item} needs {data.shape[1]} values.")
            start.append((np.array(item, dtype=float) - scaler["mean"]) / scaler["std"])
        else:
            matches = np.flatnonzero(data.index.astype(str) == str(item))
            if len(matches) == 0:
                stop(f'Starting sample "{item}" was not found. '
                     f"The first IDs are: {', '.join(map(str, data.index[:5]))}")
            start.append(scaled[matches[0]])
    start = np.array(start)
    if len(np.unique(start, axis=0)) < len(start):
        stop("Two starting centroids have identical values. Choose different samples.")
    return start


def main():
    heading("DATA")
    data, notes = read_samples(CSV_FILE, HAS_HEADER, ID_COLUMN)
    for note in notes:
        print(note)
    if len(data) < 2:
        stop("At least 2 complete samples are needed.")

    original = data.to_numpy()
    ids = data.index.astype(str).tolist()
    features = data.columns.tolist()

    # Scaling gives every feature mean 0 and standard deviation 1.
    deviations = original.std(axis=0)
    scaler = {"mean": np.zeros(original.shape[1]), "std": np.ones(original.shape[1])}
    if SCALE_FEATURES:
        scaler = {"mean": original.mean(axis=0),
                  "std": np.where(deviations == 0, 1.0, deviations)}
        print("Scaling: on. Centroids below are converted back to the original units.")
    else:
        nonzero = deviations[deviations > 0]
        if len(nonzero) and nonzero.max() / nonzero.min() > 10:
            print("WARNING: feature spreads differ by more than 10 times. Large-valued "
                  "features will dominate.\n         Consider SCALE_FEATURES = True.")
    samples = (original - scaler["mean"]) / scaler["std"]

    # ----- Elbow method -----
    heading("HOW MANY GROUPS? (ELBOW METHOD)")
    elbow = run_elbow(samples, MAX_K_TO_TEST, RANDOM_SEED, NUM_RESTARTS, MAX_ITERATIONS)
    if elbow is None:
        print("Too few distinct samples for an elbow chart.")

    # ----- Choose k and the starting centroids -----
    start = starting_centroids(data, samples, scaler)
    if start is not None:
        k = len(start)
        if isinstance(NUM_CLUSTERS, int) and NUM_CLUSTERS != k:
            stop(f"NUM_CLUSTERS is {NUM_CLUSTERS}, but INITIAL_CENTROIDS has {k} entries.")
    elif NUM_CLUSTERS == "elbow":
        if elbow is None:
            stop('NUM_CLUSTERS = "elbow" needs more distinct samples. Set a whole number.')
        k = elbow["elbow_k"]
    elif isinstance(NUM_CLUSTERS, int):
        k = NUM_CLUSTERS
    else:
        stop('NUM_CLUSTERS must be a whole number or "elbow".')

    distinct = len(np.unique(samples, axis=0))
    if not 1 <= k <= distinct:
        stop(f"k = {k} isn't possible: the data has {distinct} distinct samples.")

    if elbow is not None:
        print_elbow(elbow, k)
        save_elbow_chart(elbow, ELBOW_CHART_FILE)

    # ----- Run K-Means -----
    if start is not None:
        result = run_kmeans(samples, start, MAX_ITERATIONS)
        start_text = "your starting centroids"
    else:
        rows = {row["k"]: row for row in elbow["rows"]} if elbow else {}
        best = rows.get(k) or search_best_start(
            samples, k, RANDOM_SEED, NUM_RESTARTS, MAX_ITERATIONS)
        result = run_kmeans(samples, best["start"], MAX_ITERATIONS)
        start_text = f"the best of {NUM_RESTARTS} starts (seed {RANDOM_SEED})"

    labels = result["labels"]
    sizes = np.bincount(labels, minlength=k)
    group_names = [f"Group {g}" for g in range(1, k + 1)]
    unscale = lambda c: c * scaler["std"] + scaler["mean"]
    final_original = unscale(result["centroids"])
    history_original = [unscale(h) for h in result["history"]]
    iterations = len(result["history"]) - 1

    heading("RESULTS")
    print(f"k = {k}, using {start_text}. Stopped after {iterations} iteration(s); "
          "the last one confirmed that nothing moved.")
    if not result["converged"]:
        print(f"WARNING: stopped at the limit of {MAX_ITERATIONS} iterations.")
    if result["empty_group_seen"]:
        print("WARNING: a group became empty during the run. Try other starting centroids.")
    if result["tie_seen"]:
        print("NOTE: a sample was exactly tied between two centroids during the run. It "
              "went to the\n      lower-numbered group, but another program may choose differently.")

    print("\nSamples in each group:")
    for name, size in zip(group_names, sizes):
        print(f"  {name}: {size}")

    print("\nFinal centroids:")
    print(pd.DataFrame(final_original, index=group_names, columns=features).to_string())

    requested = []
    for centroid, iteration in CENTROIDS_TO_SHOW:
        if not 1 <= centroid <= k:
            print(f"\nC{centroid} doesn't exist because there are only {k} centroids.")
            continue
        if iteration < 0:
            print(f"\nIteration {iteration} doesn't exist. Use 0 for the start.")
            continue
        requested.append((centroid, iteration))
        step = min(iteration, iterations)
        print(f"\nCentroid {centroid} after iteration {iteration}:")
        if iteration > iterations:
            print(f"  (The run stopped at iteration {iterations}, so the centroid stays here.)")
        row = pd.DataFrame([history_original[step][centroid - 1]],
                           index=[f"C{centroid}"], columns=features)
        print(row.to_string())

    new_groups, new_scaled = {}, np.empty((0, original.shape[1]))
    if NEW_SAMPLES:
        for name, values in NEW_SAMPLES.items():
            if len(values) != original.shape[1]:
                stop(f"New sample {name} has {len(values)} values, "
                     f"but the data has {original.shape[1]} features.")
        new_original = np.array(list(NEW_SAMPLES.values()), dtype=float)
        new_scaled = (new_original - scaler["mean"]) / scaler["std"]
        new_labels = closest_centroid(new_scaled, result["centroids"])
        print("\nGroups for the new samples:")
        for name, group in zip(NEW_SAMPLES, new_labels):
            new_groups[name] = int(group) + 1
            print(f"  {name}: Group {group + 1}")

    if GROUPS_OUTPUT_FILE:
        path = Path(GROUPS_OUTPUT_FILE)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        pd.DataFrame({"ID": ids, "Group": labels + 1}).to_csv(path, index=False)
        print(f"\nSample groups saved as: {path}")

    # ----- Checks -----
    heading("CHECKS")
    checks = CheckList()
    print("1. Rules every correct K-Means result must follow:")
    rules_checks(checks, samples, result, k)

    print("\n2. Same calculation with scikit-learn:")
    sklearn_checks(checks, samples, new_scaled, result, k, requested, elbow)

    if start is not None and elbow is not None and k in elbow["k_values"]:
        best = elbow["inertias"][elbow["k_values"].index(k)]
        if result["inertia"] > best * 1.001:
            print(f"\n  NOTE: other starts reach a lower inertia ({best:.3f} vs "
                  f"{result['inertia']:.3f}). Your start is still valid, but it\n"
                  "  stopped at a less tight grouping, which K-Means can do.")
        else:
            print("\n  NOTE: your start reached the tightest grouping found by the restarts.")

    has_expected = (EXPECTED_GROUP_SIZES or EXPECTED_FINAL_CENTROIDS
                    or EXPECTED_CENTROIDS_AFTER or EXPECTED_NEW_SAMPLE_GROUPS)
    if has_expected:
        print(f"\n3. Comparison with your typed answers (allowed difference: {ANSWER_TOLERANCE}):")
        expected_checks(checks, sizes, final_original, history_original, new_groups)
    else:
        print("\n3. Comparison with your typed answers: none entered in SETTINGS.")

    print("\n4. Worked example to repeat by hand:")
    hand_check(ids, original, samples, result["centroids"], final_original, labels)

    print()
    if checks.failed:
        print(f"RESULT: {checks.failed} item(s) need attention. See FAIL or DIFFERENT above.")
    elif checks.skipped:
        print(f"RESULT: every check that ran passed, but {checks.skipped} was skipped.")
    else:
        print("RESULT: all checks passed.")


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    main()