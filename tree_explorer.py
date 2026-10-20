
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 1. ARTIFACT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_names.joblib"
TEST_PATH = ARTIFACT_DIR / "test_windows.csv"


# =========================================================
# 2. LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model_artifacts():
    required = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH,
        TEST_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing artifact files:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = list(joblib.load(FEATURES_PATH))
    test_df = pd.read_csv(TEST_PATH)

    missing_features = [
        f for f in feature_names if f not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features in test_windows.csv: "
            + ", ".join(missing_features)
        )

    return model, scaler, feature_names, test_df


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def find_column(df, candidates):
    """Find a column without case sensitivity."""
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def make_window_labels(df):
    start_col = find_column(
        df,
        ["window_start", "start_date", "window_start_date", "date"],
    )

    end_col = find_column(
        df,
        ["window_end", "end_date", "window_end_date"],
    )

    labels = []

    for idx, row in df.iterrows():
        parts = [f"Row {idx}"]

        if start_col:
            parts.append(str(row[start_col]))

        if end_col:
            parts.append(str(row[end_col]))

        labels.append(" | ".join(parts))

    return labels


# =========================================================
# 4. CALCULATE TREE LAYOUT
# =========================================================

def calculate_tree_layout(tree):
    """Calculate coordinates for every node in a tree."""

    left = tree.children_left
    right = tree.children_right

    positions = {}
    depths = {}
    next_leaf_x = [0]

    def visit(node, depth):
        depths[node] = depth

        is_leaf = left[node] == right[node]

        if is_leaf:
            x = next_leaf_x[0]
            next_leaf_x[0] += 1
        else:
            left_x = visit(left[node], depth + 1)
            right_x = visit(right[node], depth + 1)
            x = (left_x + right_x) / 2

        positions[node] = (x, -depth)
        return x

    visit(0, 0)

    return positions, depths


# =========================================================
# 5. GET THE SELECTED SAMPLE'S PATH THROUGH A TREE
# =========================================================

def get_sample_path(model, X_scaled, tree_index, row_position):
    estimator = model.estimators_[tree_index]
    tree = estimator.tree_

    # Isolation Forest trees may use a subset of the features.
    selected_features = model.estimators_features_[tree_index]

    sample_for_tree = X_scaled[
        row_position, selected_features
    ].reshape(1, -1)

    path_matrix = estimator.decision_path(sample_for_tree)

    path_nodes = path_matrix.indices.tolist()

    path_edges = list(
        zip(path_nodes[:-1], path_nodes[1:])
    )

    return (
        estimator,
        tree,
        selected_features,
        path_nodes,
        path_edges,
    )


# =========================================================
# 6. DRAW ONE TREE
# =========================================================

def draw_tree(
    model,
    X_scaled,
    feature_names,
    tree_index,
    row_position,
    max_depth=5,
):
    (
        _,
        tree,
        selected_features,
        path_nodes,
        path_edges,
    ) = get_sample_path(
        model, X_scaled, tree_index, row_position
    )

    positions, depths = calculate_tree_layout(tree)

    left = tree.children_left
    right = tree.children_right

    visible_nodes = {
        node
        for node, depth in depths.items()
        if depth <= max_depth
    }

    path_edge_set = set(path_edges)

    fig = go.Figure()

    normal_x, normal_y = [], []
    path_x, path_y = [], []

    # Draw branches.
    for parent in visible_nodes:
        for child in (left[parent], right[parent]):

            if child < 0 or child not in visible_nodes:
                continue

            x0, y0 = positions[parent]
            x1, y1 = positions[child]

            if (parent, child) in path_edge_set:
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])
            else:
                normal_x.extend([x0, x1, None])
                normal_y.extend([y0, y1, None])

    fig.add_trace(
        go.Scatter(
            x=normal_x,
            y=normal_y,
            mode="lines",
            line=dict(color="#9CA3AF", width=1),
            hoverinfo="skip",
            name="Other branches",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            line=dict(color="#F59E0B", width=4),
            hoverinfo="skip",
            name="Selected sample path",
        )
    )

    node_x, node_y = [], []
    node_text, hover_text, node_colors = [], [], []

    path_node_set = set(path_nodes)

    for node in sorted(visible_nodes):

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)

        is_leaf = left[node] == right[node]

        if is_leaf:
            label = f"Leaf {node}"

            hover = (
                f"Leaf node: {node}<br>"
                f"Tree depth: {depths[node]}<br>"
                f"Training samples: "
                f"{tree.n_node_samples[node]}"
            )

        else:
            local_feature = int(tree.feature[node])

            original_index = int(
                selected_features[local_feature]
            )

            feature_name = feature_names[original_index]

            threshold = float(tree.threshold[node])

            sample_value = float(
                X_scaled[row_position, original_index]
            )

            went_left = sample_value <= threshold

            branch = (
                "LEFT (≤ threshold)"
                if went_left
                else "RIGHT (> threshold)"
            )

            label = (
                f"{feature_name}<br>"
                f"≤ {threshold:.3f}?"
            )

            hover = (
                f"Node: {node}<br>"
                f"Feature: {feature_name}<br>"
                f"Threshold: {threshold:.6f}<br>"
                f"Sample value: {sample_value:.6f}<br>"
                f"Branch taken: {branch}<br>"
                f"Tree depth: {depths[node]}<br>"
                f"Training samples: "
                f"{tree.n_node_samples[node]}"
            )

        node_text.append(label)
        hover_text.append(hover)

        if node in path_node_set:
            node_colors.append("#F59E0B")
        elif is_leaf:
            node_colors.append("#D1D5DB")
        else:
            node_colors.append("#60A5FA")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=16,
                color=node_colors,
                line=dict(color="#374151", width=1),
            ),
            name="Tree nodes",
        )
    )

    fig.update_layout(
        title=f"Tree {tree_index} — selected sample path",
        height=650,
        showlegend=True,
        hovermode="closest",
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, title="Tree depth"),
    )

    fig.add_annotation(
        text=(
            "Orange = selected path | "
            "Blue = split node | Grey = leaf"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
        align="left",
    )

    return fig


# =========================================================
# 7. DECISION PATH TABLE
# =========================================================

def make_path_table(
    model,
    X_scaled,
    feature_names,
    tree_index,
    row_position,
):
    (
        _,
        tree,
        selected_features,
        path_nodes,
        _,
    ) = get_sample_path(
        model, X_scaled, tree_index, row_position
    )

    records = []

    for step, node in enumerate(path_nodes):

        is_leaf = (
            tree.children_left[node]
            == tree.children_right[node]
        )

        if is_leaf:
            records.append({
                "Step": step + 1,
                "Node": int(node),
                "Feature": "LEAF",
                "Sample value": np.nan,
                "Threshold": np.nan,
                "Decision": "Reached leaf",
                # FIX: None instead of the string "—".
                "Next node": None,
            })
            continue

        local_feature = int(tree.feature[node])

        original_index = int(
            selected_features[local_feature]
        )

        feature_name = feature_names[original_index]

        value = float(
            X_scaled[row_position, original_index]
        )

        threshold = float(tree.threshold[node])

        went_left = value <= threshold

        next_node = (
            tree.children_left[node]
            if went_left
            else tree.children_right[node]
        )

        decision = (
            f"{value:.4f} ≤ {threshold:.4f} → LEFT"
            if went_left
            else f"{value:.4f} > {threshold:.4f} → RIGHT"
        )

        records.append({
            "Step": step + 1,
            "Node": int(node),
            "Feature": feature_name,
            "Sample value": round(value, 6),
            "Threshold": round(threshold, 6),
            "Decision": decision,
            "Next node": int(next_node),
        })

    result = pd.DataFrame(records)

    # Ensure Arrow sees the Next node column as nullable integers.
    if not result.empty:
        result["Next node"] = pd.array(
            result["Next node"],
            dtype="Int64",
        )

    return result


# =========================================================
# 8. MAIN STREAMLIT TREE EXPLORER
# =========================================================

def render_tree_explorer():

    st.header("🌳 Isolation Forest Tree Explorer")

    try:
        model, scaler, feature_names, test_df = (
            load_model_artifacts()
        )

    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")

        st.info(
            "Ensure the artifacts folder contains "
            "isolation_forest.joblib, scaler.joblib, "
            "feature_names.joblib, and test_windows.csv."
        )
        return

    if test_df.empty:
        st.warning("test_windows.csv contains no rows.")
        return

    # -----------------------------------------------------
    # Select user
    # -----------------------------------------------------

    user_col = find_column(
        test_df,
        ["user", "user_id", "employee", "employee_id"],
    )

    if user_col:
        users = sorted(
            test_df[user_col].dropna().unique().tolist()
        )

        if not users:
            st.warning("No valid user IDs found.")
            return

        selected_user = st.selectbox(
            "Select user",
            users,
        )

        candidate_indices = test_df.index[
            test_df[user_col] == selected_user
        ].tolist()

    else:
        st.warning(
            "No user ID column found. Showing all windows."
        )
        candidate_indices = test_df.index.tolist()

    if not candidate_indices:
        st.warning("No test windows found for this selection.")
        return

    # -----------------------------------------------------
    # Select window
    # -----------------------------------------------------

    labels = make_window_labels(test_df)
    label_by_index = dict(zip(test_df.index, labels))

    score_col = find_column(
        test_df,
        ["ext_score", "anomaly_score", "score"],
    )

    default_index = candidate_indices[0]

    if score_col:
        numeric_scores = pd.to_numeric(
            test_df[score_col],
            errors="coerce",
        )

        valid_indices = [
            idx
            for idx in candidate_indices
            if pd.notna(numeric_scores.loc[idx])
        ]

        if valid_indices:
            default_index = max(
                valid_indices,
                key=lambda idx: numeric_scores.loc[idx],
            )

    selected_index = st.selectbox(
        "Select test window",
        options=candidate_indices,
        index=candidate_indices.index(default_index),
        format_func=lambda idx: label_by_index[idx],
    )

    # -----------------------------------------------------
    # Prepare features
    # -----------------------------------------------------

    X_raw = test_df[feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X_raw.isna().any().any():
        bad_cols = X_raw.columns[
            X_raw.isna().any()
        ].tolist()

        st.error(
            "Missing or non-numeric feature values in: "
            + ", ".join(bad_cols)
        )

        st.info(
            "Use the same preprocessing and imputation "
            "as during model training."
        )
        return

    try:
        X_scaled = scaler.transform(X_raw.to_numpy())

    except Exception as exc:
        st.error(f"Could not scale the test features: {exc}")
        return

    # Convert dataframe index to array position.
    row_position = test_df.index.get_loc(selected_index)

    if not isinstance(row_position, (int, np.integer)):
        st.error(
            "The test CSV has duplicate index labels. "
            "Reset the CSV index and reload the data."
        )
        return

    # -----------------------------------------------------
    # Final forest prediction
    # -----------------------------------------------------

    model_scores = -model.decision_function(X_scaled)

    model_predictions = (
        model.predict(X_scaled) == -1
    ).astype(int)

    selected_score = float(
        model_scores[row_position]
    )

    selected_prediction = int(
        model_predictions[row_position]
    )

    # Show exported values when present and valid.
    if score_col:
        saved_score = pd.to_numeric(
            pd.Series([
                test_df.loc[selected_index, score_col]
            ]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(saved_score):
            selected_score = float(saved_score)

    pred_col = find_column(
        test_df,
        [
            "ext_predicted",
            "predicted",
            "prediction",
            "is_anomaly",
        ],
    )

    if pred_col:
        saved_prediction = pd.to_numeric(
            pd.Series([
                test_df.loc[selected_index, pred_col]
            ]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(saved_prediction):
            selected_prediction = int(saved_prediction)

    st.subheader("Final model decision")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Anomaly score",
        f"{selected_score:.6f}",
    )

    c2.metric(
        "Prediction",
        "ANOMALOUS" if selected_prediction == 1 else "NORMAL",
    )

    c3.metric(
        "Number of trees",
        len(model.estimators_),
    )

    st.caption(
        "The anomaly score is not a probability. "
        "For the exported IsolationForest configured with "
        "contamination='auto', a positive -decision_function "
        "score indicates an anomaly."
    )

    # -----------------------------------------------------
    # Select tree and display its diagram
    # -----------------------------------------------------

    st.subheader("Explore an individual tree")

    tree_count = len(model.estimators_)

    if tree_count == 0:
        st.error("The model contains no trees.")
        return

    tree_index = st.slider(
        "Tree index",
        min_value=0,
        max_value=tree_count - 1,
        value=0,
        step=1,
    )

    tree_depths = [
        max(calculate_tree_layout(est.tree_)[1].values())
        for est in model.estimators_
    ]

    max_available_depth = max(
        1, min(10, int(max(tree_depths)))
    )

    max_depth = st.slider(
        "Maximum displayed depth",
        min_value=1,
        max_value=max_available_depth,
        value=min(5, max_available_depth),
        step=1,
    )

    fig = draw_tree(
        model,
        X_scaled,
        feature_names,
        tree_index,
        row_position,
        max_depth,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # -----------------------------------------------------
    # Selected tree path
    # -----------------------------------------------------

    st.markdown("#### Selected sample's decision path")

    path_df = make_path_table(
        model,
        X_scaled,
        feature_names,
        tree_index,
        row_position,
    )

    st.dataframe(
        path_df,
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # Summary of all trees
    # -----------------------------------------------------

    st.subheader(
        "All trees: selected sample path summary"
    )

    all_tree_rows = []

    for idx in range(tree_count):

        _, tree, _, nodes, _ = get_sample_path(
            model,
            X_scaled,
            idx,
            row_position,
        )

        leaf_node = nodes[-1]

        all_tree_rows.append({
            "Tree": idx,
            "Path splits": len(nodes) - 1,
            "Leaf node": int(leaf_node),
            "Leaf training samples": int(
                tree.n_node_samples[leaf_node]
            ),
        })

    all_tree_df = pd.DataFrame(all_tree_rows)

    left, right = st.columns(2)

    left.metric(
        "Average splits",
        f"{all_tree_df['Path splits'].mean():.2f}",
    )

    right.metric(
        "Shortest / longest path",
        f"{all_tree_df['Path splits'].min()} / "
        f"{all_tree_df['Path splits'].max()}",
    )

    st.dataframe(
        all_tree_df,
        width="stretch",
        hide_index=True,
        height=350,
    )

    st.download_button(
        "Download all-tree path summary",
        data=all_tree_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="isolation_forest_tree_paths.csv",
        mime="text/csv",
    )
