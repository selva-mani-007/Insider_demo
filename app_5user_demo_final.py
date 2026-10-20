"""CERT r4.2 — five-user Extended Isolation Forest explanation dashboard.

Use the exact model/scaler/23-feature list exported by the final Kaggle pipeline.
This app does not retrain or modify the model.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")
st.set_page_config(page_title="CERT r4.2 | Isolation Forest Explorer", layout="wide")

st.title("CERT r4.2 — Extended Isolation Forest: 5-User Explainability")
st.caption("Uses the exported final model, its fitted StandardScaler, and the same 23 features. No retraining occurs in this dashboard.")

# Put the Kaggle-exported files in this folder, or upload them in the sidebar.
ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "model": ROOT / "isolation_forest_23features_extended.pkl",
    "scaler": ROOT / "isolation_forest_23features_scaler.pkl",
    "features": ROOT / "extended_feature_columns.txt",
    "train": ROOT / "train_windows.csv",
    "test": ROOT / "test_windows_with_extended_scores.csv",
    "users": ROOT / "extended_user_results.csv",
}

BASELINE_FEATURES = [
    "logon_count", "logon_ah", "logon_we", "unique_pcs", "logon_ah_ratio",
    "usb_connects", "usb_ah", "usb_we", "file_count", "file_ah", "file_we",
    "email_count", "email_ah", "email_size_total", "http_count", "job_visits",
    "threat_visits", "http_ah", "unique_domains",
]
EXTENSION_FEATURES = ["exfil_overlap_days", "usb_per_logon", "ah_combined_score", "job_usb_interaction"]
FALLBACK_FEATURES = BASELINE_FEATURES + EXTENSION_FEATURES

st.sidebar.header("1. Load final Kaggle exports")
st.sidebar.write("If the files are beside this script, the app loads them automatically. Otherwise upload each file below.")

def load_bytes_or_upload(label, key, path, type_list):
    uploaded = st.sidebar.file_uploader(label, type=type_list, key=key)
    if uploaded is not None:
        return uploaded
    if path.exists():
        return path
    return None

model_src = load_bytes_or_upload("Extended model (.pkl)", "model", DEFAULT_FILES["model"], ["pkl"])
scaler_src = load_bytes_or_upload("Extended scaler (.pkl)", "scaler", DEFAULT_FILES["scaler"], ["pkl"])
features_src = load_bytes_or_upload("Feature columns (.txt)", "features", DEFAULT_FILES["features"], ["txt"])
train_src = load_bytes_or_upload("Normal training windows (.csv)", "train", DEFAULT_FILES["train"], ["csv"])
test_src = load_bytes_or_upload("Test windows with extended scores (.csv)", "test", DEFAULT_FILES["test"], ["csv"])
users_src = load_bytes_or_upload("User-level results (.csv, optional)", "users", DEFAULT_FILES["users"], ["csv"])

missing = [name for name, src in [("model", model_src), ("scaler", scaler_src), ("train windows", train_src), ("test windows", test_src)] if src is None]
if missing:
    st.info("Upload or place these required exports in the same folder as app_5user_demo_final.py: " + ", ".join(missing) + ".")
    st.stop()

def read_joblib(src):
    if hasattr(src, "read"):
        src.seek(0)
        return joblib.load(src)
    return joblib.load(src)

def read_csv(src):
    if hasattr(src, "read"):
        src.seek(0)
        return pd.read_csv(src, low_memory=False)
    return pd.read_csv(src, low_memory=False)

def read_features(src):
    if src is None:
        return FALLBACK_FEATURES
    if hasattr(src, "read"):
        src.seek(0)
        return [line.strip() for line in src.read().decode("utf-8").splitlines() if line.strip()]
    return [line.strip() for line in Path(src).read_text(encoding="utf-8").splitlines() if line.strip()]

try:
    model = read_joblib(model_src)
    scaler = read_joblib(scaler_src)
    feature_cols = read_features(features_src)
    train_df = read_csv(train_src)
    test_df = read_csv(test_src)
    user_df = read_csv(users_src) if users_src is not None else pd.DataFrame()
except Exception as exc:
    st.error(f"Could not load an export: {exc}")
    st.stop()

if len(feature_cols) != 23:
    st.error(f"Expected the final pipeline's 23 features, but loaded {len(feature_cols)}. Check extended_feature_columns.txt.")
    st.stop()
missing_train = [c for c in feature_cols if c not in train_df.columns]
missing_test = [c for c in feature_cols if c not in test_df.columns]
if missing_train or missing_test:
    st.error(f"Feature columns missing. Training: {missing_train}; test: {missing_test}")
    st.stop()
for col in ["user", "window_start", "window_end"]:
    if col not in test_df.columns:
        st.error(f"test CSV is missing required column: {col}")
        st.stop()

# Use stored predictions/scores when present; otherwise reproduce inference with the saved model/scaler.
X_test = test_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
if "ext_score" not in test_df.columns or "ext_predicted" not in test_df.columns:
    X_scaled_all = scaler.transform(X_test)
    test_df["ext_score"] = -model.decision_function(X_scaled_all)
    test_df["ext_predicted"] = (model.predict(X_scaled_all) == -1).astype(int)
else:
    # Use the exported final-model scores; calculate only if values are missing.
    if test_df["ext_score"].isna().any() or test_df["ext_predicted"].isna().any():
        X_scaled_all = scaler.transform(X_test)
        scores = -model.decision_function(X_scaled_all)
        preds = (model.predict(X_scaled_all) == -1).astype(int)
        test_df["ext_score"] = test_df["ext_score"].fillna(pd.Series(scores, index=test_df.index))
        test_df["ext_predicted"] = test_df["ext_predicted"].fillna(pd.Series(preds, index=test_df.index))

test_df["label"] = pd.to_numeric(test_df.get("label", 0), errors="coerce").fillna(0).astype(int)
test_df["ext_predicted"] = pd.to_numeric(test_df["ext_predicted"], errors="coerce").fillna(0).astype(int)
test_df["ext_score"] = pd.to_numeric(test_df["ext_score"], errors="coerce").fillna(0.0)

# One row per user for the selector. Label is ground truth for selecting known malicious users,
# not a feature passed into the model.
user_summary = test_df.groupby("user", as_index=False).agg(
    true_label=("label", "max"),
    max_ext_score=("ext_score", "max"),
    flagged_windows=("ext_predicted", "sum"),
    total_windows=("ext_predicted", "count"),
)
user_summary["any_window_flagged"] = (user_summary["flagged_windows"] > 0).astype(int)
malicious_users = user_summary[user_summary["true_label"] == 1].sort_values("max_ext_score", ascending=False)["user"].astype(str).tolist()
normal_users = user_summary[user_summary["true_label"] == 0].sort_values("max_ext_score", ascending=False)["user"].astype(str).tolist()

st.sidebar.header("2. Choose five sample users")
selection_mode = st.sidebar.radio("Sample set", ["Five known malicious users", "Five normal users", "Choose users manually"], index=0)
if selection_mode == "Five known malicious users":
    choices = malicious_users[:5]
elif selection_mode == "Five normal users":
    choices = normal_users[:5]
else:
    choices = st.sidebar.multiselect("Select up to five users", user_summary["user"].astype(str).tolist(), default=malicious_users[:min(5, len(malicious_users))], max_selections=5)
if not choices:
    st.warning("No users available for this selection.")
    st.stop()
if len(choices) < 5:
    st.warning(f"Only {len(choices)} users were available in this category.")

summary_view = user_summary[user_summary["user"].astype(str).isin(choices)].copy()
summary_view["user"] = summary_view["user"].astype(str)
summary_view["label_text"] = summary_view["true_label"].map({0: "Normal (ground truth)", 1: "Malicious (ground truth)"})
summary_view["prediction_text"] = summary_view["any_window_flagged"].map({0: "No window flagged", 1: "At least one window flagged"})

st.subheader("At-a-glance results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users selected", len(choices))
c2.metric("23-feature input", "Confirmed")
c3.metric("Model trees", len(getattr(model, "estimators_", [])))
c4.metric("Flagged selected users", int(summary_view["any_window_flagged"].sum()))
st.dataframe(summary_view[["user", "label_text", "max_ext_score", "flagged_windows", "total_windows", "prediction_text"]].rename(columns={"label_text":"Ground truth", "max_ext_score":"Max window anomaly score", "flagged_windows":"Flagged windows", "total_windows":"Total windows", "prediction_text":"Model result"}), width="stretch", hide_index=True)

st.caption("The model was trained on normal-user windows only. Known malicious users are selected from the labelled test windows, not from the normal-only training set. ‘Malicious’ below refers to dataset ground truth; the model prediction is shown separately.")

# Sample window: highest anomaly score for the selected user. Use exact 23 features and saved scaler.
user_choice = st.selectbox("Inspect user", choices)
uw = test_df[test_df["user"].astype(str) == str(user_choice)].copy()
if uw.empty:
    st.error("No test windows found for this user.")
    st.stop()
uw = uw.sort_values("ext_score", ascending=False)
window_labels = [f"{r.window_start} → {r.window_end} | score={float(r.ext_score):.4f} | {'FLAGGED' if int(r.ext_predicted) else 'not flagged'}" for r in uw.itertuples()]
selected_label = st.selectbox("Inspect window (highest anomaly score shown first)", window_labels)
selected_idx = window_labels.index(selected_label)
sample = uw.iloc[selected_idx]
raw_x = sample[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy().reshape(1, -1)
scaled_x = scaler.transform(raw_x)[0]
score = float(-model.decision_function(scaler.transform(raw_x))[0])
pred = int(model.predict(scaler.transform(raw_x))[0] == -1)

st.markdown("### Selected window decision")
a, b, c, d = st.columns(4)
a.metric("User", str(user_choice))
b.metric("Anomaly score", f"{score:.5f}")
c.metric("Model prediction", "ANOMALY" if pred else "NORMAL")
d.metric("Ground-truth label", "MALICIOUS" if int(sample.get("label", 0)) == 1 else "NORMAL")
st.write(f"**Window:** {sample['window_start']} to {sample['window_end']}  ")
st.write("Higher `-decision_function` values indicate more anomalous behavior in this pipeline. Prediction uses the saved model's own `predict()` threshold.")

# Feature evidence: compare this window with normal-only training windows.
st.markdown("### 1. Feature evidence — observed values vs normal training behavior")
train_features = train_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
means = train_features.mean()
stds = train_features.std(ddof=0).replace(0, np.nan)
raw_values = pd.Series(raw_x[0], index=feature_cols)
z = ((raw_values - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0)
feat_df = pd.DataFrame({"Feature": feature_cols, "Observed window value": raw_values.values, "Normal training mean": means.values, "Normal training std": stds.fillna(0).values, "Deviation (z-score)": z.values})
feat_df["Absolute deviation"] = feat_df["Deviation (z-score)"].abs()
feat_df = feat_df.sort_values("Absolute deviation", ascending=False)
fig_feat = px.bar(feat_df.head(12).sort_values("Deviation (z-score)"), x="Deviation (z-score)", y="Feature", orientation="h", title="Largest standardized deviations from normal training windows")
fig_feat.add_vline(x=0, line_dash="dash")
st.plotly_chart(fig_feat, width="stretch")
st.dataframe(feat_df.rename(columns={"Observed window value":"Observed", "Normal training mean":"Training mean", "Normal training std":"Training std", "Deviation (z-score)":"Z-score"})[["Feature", "Observed", "Training mean", "Training std", "Z-score"]], width="stretch", hide_index=True)
st.caption("Feature deviations are context, not a causal explanation by themselves. The tree-path view below shows the actual splits this saved Isolation Forest tree used for this window.")

# Select a tree and walk it directly using the actual scaled sample, avoiding CSR path ordering issues.
st.markdown("### 2. Decision-tree path — why this sample reached an isolated leaf")
estimators = getattr(model, "estimators_", None)
if not estimators:
    st.error("The loaded model does not expose sklearn IsolationForest estimators_.")
    st.stop()
tree_num = st.slider("Tree number", min_value=1, max_value=len(estimators), value=1) - 1
est = estimators[tree_num]
tree = est.tree_
# IsolationForest estimators are fitted on a subset of features; map local tree feature index to original 23-feature index.
feature_map = getattr(model, "estimators_features_", None)
if feature_map is not None:
    fmap = np.asarray(feature_map[tree_num], dtype=int)
else:
    fmap = np.arange(len(feature_cols), dtype=int)
node = 0
path = []
max_steps = tree.node_count + 1
for _ in range(max_steps):
    if tree.feature[node] < 0:
        path.append({"node": int(node), "feature": "LEAF", "threshold": np.nan, "scaled_value": np.nan, "observed_value": np.nan, "branch": "Terminal leaf — path ends here", "next_node": None})
        break
    local_idx = int(tree.feature[node])
    original_idx = int(fmap[local_idx])
    fname = feature_cols[original_idx]
    threshold = float(tree.threshold[node])
    value = float(scaled_x[original_idx])
    observed_value = float(raw_x[0, original_idx])
    go_left = value <= threshold
    next_node = int(tree.children_left[node] if go_left else tree.children_right[node])
    path.append({"node": int(node), "feature": fname, "threshold": threshold, "scaled_value": value, "observed_value": observed_value, "branch": f"{value:.4f} {'≤' if go_left else '>'} {threshold:.4f} → {'left' if go_left else 'right'}", "next_node": next_node})
    node = next_node
path_df = pd.DataFrame(path)

# Draw the FULL branching tree, in the style of tree_explorer.py, with the
# selected sample route highlighted. Limit displayed depth for readability.
st.markdown("#### Full tree view — selected sample path highlighted")
max_tree_depth = st.slider("Maximum displayed tree depth", min_value=2, max_value=8, value=5, step=1)
path_nodes = {int(r["node"]) for _, r in path_df.iterrows()}
path_edges = set()
for _, r in path_df.iterrows():
    if pd.notna(r["next_node"]):
        path_edges.add((int(r["node"]), int(r["next_node"])))

# Determine the displayed subtree, stopping at max_tree_depth.
positions = {}
labels = {}
hover = {}
visible_edges = []
leaf_counter = [0]
def layout_subtree(nid, depth):
    nid = int(nid)
    is_leaf = tree.feature[nid] < 0
    if is_leaf or depth >= max_tree_depth:
        x = leaf_counter[0]
        leaf_counter[0] += 1
    else:
        left = int(tree.children_left[nid])
        right = int(tree.children_right[nid])
        lx = layout_subtree(left, depth + 1)
        rx = layout_subtree(right, depth + 1)
        x = (lx + rx) / 2
        visible_edges.extend([(nid, left), (nid, right)])
    positions[nid] = (x, -depth)
    if is_leaf:
        labels[nid] = f"Leaf {nid}"
        hover[nid] = f"Leaf node {nid}<br>Depth: {depth}<br>Training samples: {int(tree.n_node_samples[nid])}"
    elif depth >= max_tree_depth:
        labels[nid] = f"… node {nid}"
        hover[nid] = f"Subtree clipped at display depth {max_tree_depth}<br>Node: {nid}"
    else:
        local_f = int(tree.feature[nid])
        orig_f = int(fmap[local_f])
        fname = feature_cols[orig_f]
        thr = float(tree.threshold[nid])
        val = float(scaled_x[orig_f])
        direction = "left (≤)" if val <= thr else "right (>)"
        labels[nid] = f"{fname}<br>≤ {thr:.3g}"
        hover[nid] = (f"Node {nid}: {fname}<br>Threshold (scaled): {thr:.6g}"
                      f"<br>Sample raw value: {float(raw_x[0, orig_f]):.6g}"
                      f"<br>Sample scaled value: {val:.6g}<br>Selected sample goes {direction}")
    return x
layout_subtree(0, 0)
fig_path = go.Figure()
# Branches first, so node markers render above them.
for parent, child in visible_edges:
    x0, y0 = positions[parent]
    x1, y1 = positions[child]
    on_path = (parent, child) in path_edges
    fig_path.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
        line=dict(color="#ff9800" if on_path else "#69717d", width=4 if on_path else 1.2),
        hoverinfo="skip", showlegend=False))
for nid, (x, y) in positions.items():
    is_leaf = tree.feature[nid] < 0 or nid not in {p for p, _ in visible_edges}
    on_path = nid in path_nodes
    color = "#ff9800" if on_path and not (tree.feature[nid] < 0) else ("#ffadad" if tree.feature[nid] < 0 else "#60a5fa")
    if on_path and tree.feature[nid] < 0:
        color = "#ffadad"
    fig_path.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text",
        marker=dict(size=24 if on_path else 15, color=color, line=dict(color="#3f4650", width=1.5)),
        text=[labels[nid]], textposition="top center", textfont=dict(size=9 if len(positions) > 35 else 10, color="#f3f4f6"),
        hovertext=[hover[nid]], hovertemplate="%{hovertext}<extra></extra>", showlegend=False))
fig_path.update_layout(title=f"Tree {tree_num + 1} — selected sample path highlighted",
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#f3f4f6"),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    height=max(520, min(1000, 170 + max_tree_depth * 115)),
    margin=dict(l=20, r=20, t=70, b=20), showlegend=False)
st.plotly_chart(fig_path, width="stretch")
st.caption("Orange nodes and branches show the selected sample's route; blue nodes are other split nodes; pink nodes are leaves. Increase the displayed depth to inspect more of the tree. Hover over nodes for thresholds and sample values.")
st.markdown("**How to read this path:** each highlighted node is visited by the selected window. At each split, the tree compares the scaled value with the threshold; `≤` follows the left branch and `>` follows the right branch. The original observed value is included for interpretation, but the split itself uses the scaled value.")
st.dataframe(path_df.rename(columns={"node":"Node ID", "feature":"Split feature", "threshold":"Threshold (scaled space)", "scaled_value":"Value used by split (scaled)", "observed_value":"Observed value (original units)", "branch":"Decision taken", "next_node":"Next node"})[["Node ID", "Split feature", "Observed value (original units)", "Value used by split (scaled)", "Threshold (scaled space)", "Decision taken", "Next node"]], width="stretch", hide_index=True)
st.success(f"Selected window prediction: {'ANOMALY' if pred else 'NORMAL'} | anomaly score: {score:.5f} | ground truth: {'MALICIOUS' if int(sample.get('label', 0)) == 1 else 'NORMAL'}. This tree path explains one tree’s route; the final prediction comes from the whole forest.")

# Aggregate path evidence across a sample of trees; report the actual per-tree path length.
st.markdown("### 3. Forest-level isolation evidence")
num_trees = len(estimators)
all_path_lengths = []
all_leaf_depths = []
for tree_i, est_i in enumerate(estimators):
    tr = est_i.tree_
    fmap_i = np.asarray(feature_map[tree_i], dtype=int) if feature_map is not None else np.arange(len(feature_cols), dtype=int)
    n = 0
    depth = 0
    while tr.feature[n] >= 0 and depth < tr.node_count:
        loc = int(tr.feature[n]); orig = int(fmap_i[loc])
        if scaled_x[orig] <= float(tr.threshold[n]):
            n = int(tr.children_left[n])
        else:
            n = int(tr.children_right[n])
        depth += 1
    all_path_lengths.append(depth)
    all_leaf_depths.append(n)
length_df = pd.DataFrame({"Tree": np.arange(1, num_trees + 1), "Decision depth to leaf": all_path_lengths})
fig_len = px.histogram(length_df, x="Decision depth to leaf", nbins=max(5, min(25, int(np.sqrt(num_trees)))), title="Path lengths across the forest for the selected window")
fig_len.add_vline(x=float(np.mean(all_path_lengths)), line_dash="dash", annotation_text=f"Mean depth = {np.mean(all_path_lengths):.2f}")
st.plotly_chart(fig_len, width="stretch")
st.write(f"Across **{num_trees} trees**, this sample's mean decision depth to a leaf is **{np.mean(all_path_lengths):.2f}** (min {min(all_path_lengths)}, max {max(all_path_lengths)}). Isolation Forest combines path-length-based tree results into its anomaly score; this chart is explanatory and is not a replacement for the model's exact scoring implementation.")

st.markdown("### 4. Human-readable evidence summary")
large_devs = feat_df.head(5)
if pred == 1:
    st.warning("The saved model flags this window as anomalous. The most unusual feature values relative to normal-only training windows are listed below; the selected tree path shows the concrete split decisions for one tree.")
else:
    st.success("The saved model does not flag this window as anomalous, although individual features may differ from normal training behavior.")
for _, row in large_devs.iterrows():
    direction = "above" if row["Deviation (z-score)"] > 0 else "below"
    st.write(f"- **{row['Feature']}** = {row['Observed window value']:.4g}; normal-training mean = {row['Normal training mean']:.4g}; {abs(row['Deviation (z-score)']):.2f} standard deviations {direction} the training mean.")
st.caption("This is an interpretable evidence summary, not a claim that any one feature caused the alert. Isolation Forest is unsupervised; the ground-truth label is displayed for evaluation only and is never passed to the model.")

with st.expander("Show the exact 23 raw feature values used"):
    st.dataframe(pd.DataFrame({"Feature": feature_cols, "Raw value": raw_x[0], "Scaled value": scaled_x}), width="stretch", hide_index=True)

st.markdown("---")
st.caption("Model files should come from the same Kaggle run: isolation_forest_23features_extended.pkl, isolation_forest_23features_scaler.pkl, extended_feature_columns.txt, train_windows.csv, and test_windows_with_extended_scores.csv. Do not substitute a scaler from another model run.")

# ─────────────────────────────────────────────────────────────────────────────
# PLAIN-LANGUAGE EXPLAINABILITY SECTION
# Paste this block into your final app JUST BEFORE the last st.markdown("---")
# footer line. It reads variables already set above it in your app:
#   user_choice, pred, score, feat_df, raw_x, feature_cols, sample, all_path_lengths
# Nothing else is changed.
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔍 What Does This Mean? (Plain English Explanation)")
st.caption(
    "This section explains the result in simple words — "
    "no machine learning knowledge needed to understand it."
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _severity(z):
    """Convert absolute z-score to a plain severity word."""
    if z >= 6:   return "extremely unusual"
    if z >= 4:   return "very unusual"
    if z >= 2.5: return "notably unusual"
    if z >= 1.5: return "slightly unusual"
    return "close to normal"

def _icon(z):
    if z >= 6:   return "🚨"
    if z >= 4:   return "⚠️"
    if z >= 2.5: return "🔶"
    if z >= 1.5: return "🔸"
    return "✅"

def _direction_plain(z):
    return "much more than normal" if z > 0 else "much less than normal"

# ── build plain-English sentences for the top deviating features ─────────────

PLAIN_DESCRIPTIONS = {
    "usb_we":             ("used a USB drive on weekends",
                           "plugging in a USB drive on weekends is uncommon for normal employees"),
    "logon_we":           ("logged into the system on weekends",
                           "most employees do not log in on weekends"),
    "usb_ah":             ("used a USB drive outside normal working hours (before 8 AM or after 6 PM)",
                           "connecting USB drives outside working hours is a data-theft risk indicator"),
    "file_ah":            ("accessed files outside normal working hours",
                           "after-hours file access can indicate unauthorized data copying"),
    "logon_ah":           ("logged in outside normal working hours",
                           "after-hours logins are unusual for most employees"),
    "job_visits":         ("visited job-hunting or recruitment websites",
                           "employees looking for a new job sometimes copy data before leaving"),
    "job_usb_interaction":("visited job websites AND used a USB drive in the same week",
                           "combining job-site browsing with USB activity in the same period is a classic pre-departure data-theft pattern"),
    "usb_connects":       ("made many USB drive connections",
                           "a high number of USB connections can mean data is being copied to external storage"),
    "exfil_overlap_days": ("had days where USB use and external-website visits happened on the same day",
                           "doing both on the same day could mean: browse for a place to send data, then copy it to a USB"),
    "email_size_total":   ("sent or received unusually large emails",
                           "very large emails can be used to send sensitive files outside the company"),
    "email_count":        ("sent or received far more emails than normal",
                           "a spike in email activity can indicate data being sent out"),
    "threat_visits":      ("visited websites flagged as high-risk (e.g. wikileaks, pastebin)",
                           "visiting these sites at work is very rare for normal employees"),
    "http_count":         ("browsed the internet far more than average",
                           "unusually high web activity can mean research for an attack or distraction"),
    "unique_domains":     ("visited many more different websites than normal",
                           "a large number of distinct websites can indicate reconnaissance or data-sharing activity"),
    "unique_pcs":         ("used an unusually high or low number of computers",
                           "using many computers could indicate moving between systems to avoid detection"),
    "usb_per_logon":      ("had a high ratio of USB use per login session",
                           "connecting USB drives on almost every login is not typical employee behaviour"),
    "logon_count":        ("logged in many more times than usual",
                           "an unusually high login count can mean long or repeated sessions"),
    "logon_ah_ratio":     ("had a high proportion of after-hours logins",
                           "when most of someone's logins happen after hours, it is a timing red flag"),
    "ah_combined_score":  ("showed after-hours activity across multiple data sources at the same time",
                           "when logins, USB, files, emails and web browsing all happen after hours together, it is a strong combined signal"),
    "file_count":         ("accessed an unusual number of files",
                           "accessing far more files than normal can mean bulk data collection"),
    "file_we":            ("accessed files on weekends",
                           "weekend file access is uncommon and can indicate unauthorised activity"),
    "email_ah":           ("sent emails outside normal working hours",
                           "emailing after hours — especially with attachments — is a data-exfiltration warning sign"),
    "http_ah":            ("browsed the web outside normal working hours",
                           "after-hours web browsing combined with other signals raises the risk level"),
}

top_evidence = feat_df.head(5)   # already sorted by absolute deviation in your app

plain_bullets = []
for _, row in top_evidence.iterrows():
    fname = row["Feature"]
    z     = float(row["Deviation (z-score)"])
    abs_z = abs(z)
    obs   = float(row["Observed window value"])
    mean  = float(row["Normal training mean"])

    what, why = PLAIN_DESCRIPTIONS.get(
        fname,
        (f"had an unusual value for '{fname}'",
         f"this feature was {_severity(abs_z)} compared with normal employees")
    )

    direction = "more" if z > 0 else "fewer/less"
    times_str = ""
    if mean > 0.01 and obs > 0:
        ratio = obs / mean
        if ratio >= 2:
            times_str = f" (about {ratio:.0f}× the normal amount)"

    plain_bullets.append({
        "icon":  _icon(abs_z),
        "label": fname,
        "severity": _severity(abs_z),
        "what":  what,
        "why":   why,
        "obs":   obs,
        "mean":  mean,
        "z":     z,
        "times": times_str,
    })

# ── verdict card ─────────────────────────────────────────────────────────────

mean_depth  = float(np.mean(all_path_lengths))
is_anomaly  = pred == 1
is_malicious = int(sample.get("label", 0)) == 1

if is_anomaly:
    verdict_color = "#7f1d1d"   # dark red
    verdict_icon  = "🚨"
    verdict_title = "The system flagged this user as SUSPICIOUS"
    verdict_intro = (
        f"The Isolation Forest algorithm decided that **{user_choice}**'s behaviour "
        f"during this 7-day period was **very different from what normal employees do**. "
        f"Think of it like this: imagine you train a detector by showing it thousands of "
        f"examples of normal office behaviour. When it sees something that does not fit "
        f"that pattern — it raises an alarm. That is exactly what happened here."
    )
else:
    verdict_color = "#14532d"   # dark green
    verdict_icon  = "✅"
    verdict_title = "The system considers this user NORMAL"
    verdict_intro = (
        f"The Isolation Forest algorithm decided that **{user_choice}**'s behaviour "
        f"during this 7-day period was **similar to what normal employees do**. "
        f"Even if some individual numbers look a little high or low, they were not "
        f"unusual enough — in combination — to trigger an alert."
    )

st.markdown(
    f"""
    <div style="background:{verdict_color};border-radius:12px;padding:24px 28px;margin-bottom:18px;">
        <h2 style="color:white;margin:0 0 8px 0;">{verdict_icon} {verdict_title}</h2>
        <p style="color:#f3f4f6;font-size:16px;margin:0;">{verdict_intro}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── how the machine decides — one-paragraph plain explanation ─────────────────

st.markdown("#### 🤖 How Did the Machine Reach This Decision?")
st.markdown(
    f"""
Imagine you have a big bucket of data about how normal employees behave at work every week —
how many times they log in, whether they use a USB drive, how many emails they send, 
which websites they visit, and so on. 

The **Isolation Forest** algorithm learned what "normal" looks like from **830 normal employees** 
over 18 months. Then, when it sees a new 7-day window of activity, it tries to 
**separate (isolate) that person from the crowd** using a series of random questions like:
*"Did this person log in more than X times? Did they use a USB drive more than Y times?"*

If a person blends in with the normal crowd, it takes **many questions** to isolate them — 
they are normal. If a person stands out immediately and gets isolated in just a **few questions**, 
they are flagged as anomalous.

For **{user_choice}** in this window, the algorithm needed an average of only 
**{mean_depth:.1f} questions** across 200 decision trees to isolate them 
({'which is short — a sign of anomaly' if mean_depth < 5 else 'which is within the normal range'}).
The resulting anomaly score was **{score:.4f}** 
({'above the threshold → FLAGGED' if is_anomaly else 'below the threshold → not flagged'}).
""",
    unsafe_allow_html=False,
)

# ── top signals in plain English ─────────────────────────────────────────────

st.markdown("#### 📋 The Top Signals — What Stood Out?")
st.caption(
    "These are the behaviours that were most different from a normal employee during this week. "
    "Each one by itself might not be alarming — but together they build a picture."
)

for b in plain_bullets:
    z_display = f"{abs(b['z']):.1f}× more extreme than normal" if abs(b['z']) > 1 else "close to normal"
    st.markdown(
        f"""
        <div style="background:#1e293b;border-left:5px solid {'#ef4444' if abs(b['z'])>=2.5 else '#f59e0b' if abs(b['z'])>=1.5 else '#22c55e'};
             border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:10px;">
            <b style="font-size:16px;">{b['icon']} {b['what'].capitalize()}{b['times']}</b><br>
            <span style="color:#94a3b8;font-size:14px;">
                <b>Why it matters:</b> {b['why'].capitalize()}.<br>
                <b>How unusual:</b> This is <b>{b['severity']}</b> ({z_display}).
                Observed value: <b>{b['obs']:.1f}</b> vs typical employee: <b>{b['mean']:.2f}</b>.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── the big picture narrative ─────────────────────────────────────────────────

st.markdown("#### 🧩 Putting It All Together")

# Build a narrative based on which signals fired
signal_groups = {
    "after_hours": [b for b in plain_bullets if b["label"] in ("logon_ah","usb_ah","file_ah","email_ah","http_ah","logon_ah_ratio","ah_combined_score") and abs(b["z"]) >= 1.5],
    "weekend":     [b for b in plain_bullets if b["label"] in ("logon_we","usb_we","file_we") and abs(b["z"]) >= 1.5],
    "usb":         [b for b in plain_bullets if b["label"] in ("usb_connects","usb_per_logon","usb_we","usb_ah") and abs(b["z"]) >= 1.5],
    "job":         [b for b in plain_bullets if b["label"] in ("job_visits","job_usb_interaction") and abs(b["z"]) >= 1.5],
    "overlap":     [b for b in plain_bullets if b["label"] == "exfil_overlap_days" and abs(b["z"]) >= 1.5],
    "email":       [b for b in plain_bullets if b["label"] in ("email_size_total","email_count","email_ah") and abs(b["z"]) >= 1.5],
    "threat":      [b for b in plain_bullets if b["label"] == "threat_visits" and abs(b["z"]) >= 1.5],
}

story_parts = []

if signal_groups["threat"]:
    story_parts.append(
        "🔴 **High-risk website visits:** This employee visited websites that are almost never "
        "accessed by normal staff (like Wikileaks or Pastebin). This alone is a strong warning signal."
    )

if signal_groups["job"] and signal_groups["usb"]:
    story_parts.append(
        "🔴 **Classic pre-departure pattern:** The employee was browsing job-hunting websites "
        "AND using USB drives in the same week. This combination is one of the most well-known "
        "patterns of an employee who is planning to leave and copy company data before going."
    )
elif signal_groups["job"]:
    story_parts.append(
        "🟡 **Job-site browsing:** The employee visited job-hunting or recruitment websites "
        "significantly more than normal. Employees actively job-searching sometimes take company "
        "data with them when they leave."
    )
elif signal_groups["usb"]:
    story_parts.append(
        "🟡 **Unusual USB activity:** The employee connected USB drives far more than normal. "
        "This could be innocent, but it is also a common way of copying files out of a company."
    )

if signal_groups["overlap"]:
    story_parts.append(
        "🔴 **Same-day USB + external website:** On multiple days, the employee both plugged in "
        "a USB drive AND visited external websites. A possible explanation: research where to send "
        "the data, then copy it to the USB. This co-occurrence is a strong exfiltration signal."
    )

if signal_groups["after_hours"] and signal_groups["weekend"]:
    story_parts.append(
        "🟠 **Off-hours and weekend activity:** This employee was active both outside working hours "
        "(late at night / early morning) AND on weekends. Normal employees rarely access company "
        "systems at these times, so this pattern raises concern."
    )
elif signal_groups["after_hours"]:
    story_parts.append(
        "🟡 **After-hours activity:** The employee logged in, used files, or sent emails outside "
        "normal working hours more than a typical employee would. This alone is not conclusive, "
        "but in combination with other signals it adds to the risk picture."
    )
elif signal_groups["weekend"]:
    story_parts.append(
        "🟡 **Weekend activity:** The employee was active on weekends — logging in or using USB "
        "drives on days when most employees do not work."
    )

if signal_groups["email"]:
    story_parts.append(
        "🟡 **Unusual email activity:** The employee sent significantly larger or more emails than "
        "normal. Large email attachments are a common way of sending sensitive documents outside "
        "the organisation."
    )

if not story_parts:
    if is_anomaly:
        story_parts.append(
            "🟠 The individual signals may each seem small, but the Isolation Forest algorithm "
            "detected that the overall combination of this employee's behaviour was unusual compared "
            "to the 830 normal employees it learned from. "
            "No single action caused the flag — it was the unusual pattern as a whole."
        )
    else:
        story_parts.append(
            "✅ None of this employee's behaviours stood out significantly from what normal "
            "employees do. The overall pattern matched the normal training data closely enough "
            "that the algorithm did not raise an alarm."
        )

for part in story_parts:
    st.markdown(part)

# ── final plain verdict ───────────────────────────────────────────────────────

st.markdown("#### ⚖️ Final Verdict (in simple terms)")

if is_anomaly and is_malicious:
    final_text = (
        f"✅ **The system got it right.** {user_choice} is a confirmed malicious insider "
        f"in the dataset, and the algorithm flagged them. The combination of unusual behaviours "
        f"listed above is what made this person stand out from the crowd of normal employees."
    )
    box_color = "#14532d"
elif is_anomaly and not is_malicious:
    final_text = (
        f"⚠️ **False alarm.** {user_choice} is actually a normal user in the dataset, "
        f"but the algorithm flagged them. This can happen when a normal employee has an "
        f"unusually busy week — for example, working late on a big project or moving files "
        f"between computers. This is called a False Positive."
    )
    box_color = "#78350f"
elif not is_anomaly and not is_malicious:
    final_text = (
        f"✅ **Correctly not flagged.** {user_choice} is a normal user and the algorithm "
        f"correctly identified their behaviour as normal. Their activity during this week "
        f"closely matched what the system learned from hundreds of normal employees."
    )
    box_color = "#14532d"
else:
    final_text = (
        f"❌ **Missed detection.** {user_choice} is a confirmed malicious insider "
        f"but the algorithm did not flag them in this window. This is called a False Negative. "
        f"It can happen when the malicious activity in a particular week was subtle enough "
        f"to blend in with normal behaviour, or when the attack window did not fall within "
        f"this specific 7-day period."
    )
    box_color = "#7f1d1d"

st.markdown(
    f'<div style="background:{box_color};border-radius:10px;padding:18px 22px;">'
    f'<p style="color:#f3f4f6;font-size:15px;margin:0;">{final_text}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

st.caption(
    "⚠️ Important: This is an academic research demonstration on the CERT r4.2 dataset. "
    "An anomaly flag is a review signal — not proof of wrongdoing. "
    "In a real investigation, a human analyst would always verify the findings before any action is taken."
)