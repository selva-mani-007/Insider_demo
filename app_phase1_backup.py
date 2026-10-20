import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Insider Threat Detection", page_icon="🛡️", layout="wide")

FEATURES = [
    'logon_count','logon_ah','logon_we','unique_pcs','logon_ah_ratio',
    'usb_connects','usb_ah','usb_we','file_count','file_ah','file_we',
    'email_count','email_ah','email_size_total','http_count','job_visits',
    'threat_visits','http_ah','unique_domains','exfil_overlap_days',
    'usb_per_logon','ah_combined_score','job_usb_interaction'
]
FRIENDLY = {
    'logon_count':'Logon activity','logon_ah':'After-hours logons','logon_we':'Weekend logons',
    'unique_pcs':'Number of distinct PCs','logon_ah_ratio':'After-hours logon ratio',
    'usb_connects':'USB connections','usb_ah':'After-hours USB activity','usb_we':'Weekend USB activity',
    'file_count':'File activity','file_ah':'After-hours file activity','file_we':'Weekend file activity',
    'email_count':'Email volume','email_ah':'After-hours email activity','email_size_total':'Total email size',
    'http_count':'Web activity','job_visits':'Job-site visits','threat_visits':'Visits to configured threat-related sites',
    'http_ah':'After-hours web activity','unique_domains':'Distinct web domains',
    'exfil_overlap_days':'Days with USB and configured external-site activity',
    'usb_per_logon':'USB connections per logon','ah_combined_score':'Combined after-hours activity ratio',
    'job_usb_interaction':'Job-site visits × USB connections'
}

st.title("🛡️ Insider Threat Detection System")
st.caption("CERT r4.2 · Extended Isolation Forest · 23 behavioural features")
st.info("This demo explains the results of the existing experiment. An anomaly is a review signal, not proof of malicious intent.")

with st.sidebar:
    st.header("Load experiment outputs")
    st.write("Upload the CSV outputs exported from your completed Kaggle run.")
    train_file = st.file_uploader("Normal training windows CSV", type="csv", key="train")
    windows_file = st.file_uploader("Test windows with extended scores CSV", type="csv", key="windows")
    users_file = st.file_uploader("Extended user-level results CSV", type="csv", key="users")
    st.divider()
    st.caption("No prediction is fabricated when required files or columns are missing.")

if not (train_file and windows_file and users_file):
    st.subheader("Demo setup")
    st.markdown("Upload these three files to enable user-level investigation:")
    st.markdown("1. `train_windows.csv` — normal training features.\n2. `test_windows_with_extended_scores.csv` — test windows with `ext_score` and `ext_predicted`.\n3. `extended_user_results.csv` — one row per user with the extended-model decision.")
    st.markdown("**Important:** Your current notebook saves the extended model and scaler, but it does not save the final extended-scored test windows and user-level results under these exact filenames yet. Add the export snippet provided with this demo to the end of the notebook and rerun only that final export cell after the model variables are still in memory.")
    st.stop()

try:
    train_df = pd.read_csv(train_file, low_memory=False)
    windows_df = pd.read_csv(windows_file, low_memory=False)
    users_df = pd.read_csv(users_file, low_memory=False)
except Exception as e:
    st.error(f"Could not read uploaded CSV: {e}")
    st.stop()

required_user_cols = {'user','max_ext_score','user_pred'}
required_window_cols = {'user','ext_score','ext_predicted','window_start','window_end'}
missing = required_user_cols - set(users_df.columns)
missing_windows = required_window_cols - set(windows_df.columns)
missing_features = set(FEATURES) - set(windows_df.columns)
if missing or missing_windows or missing_features:
    st.error("The uploaded files do not match the expected extended-model output schema.")
    if missing: st.write("Missing user-result columns:", sorted(missing))
    if missing_windows: st.write("Missing window columns:", sorted(missing_windows))
    if missing_features: st.write("Missing feature columns:", sorted(missing_features))
    st.stop()

users_df['user'] = users_df['user'].astype(str)
windows_df['user'] = windows_df['user'].astype(str)
users_df['user_pred'] = pd.to_numeric(users_df['user_pred'], errors='coerce').fillna(0).astype(int)
windows_df['ext_predicted'] = pd.to_numeric(windows_df['ext_predicted'], errors='coerce').fillna(0).astype(int)

left, right = st.columns([1, 2])
with left:
    chosen_user = st.selectbox("Select a user", sorted(users_df['user'].unique()))
user_row = users_df.loc[users_df['user'] == chosen_user].iloc[0]
user_windows = windows_df.loc[windows_df['user'] == chosen_user].copy()
pred = int(user_row['user_pred'])
score = float(user_row['max_ext_score'])

with right:
    st.subheader(f"Investigation: {chosen_user}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model decision", "ANOMALOUS" if pred == 1 else "NORMAL")
    c2.metric("Maximum anomaly score", f"{score:.4f}")
    c3.metric("Flagged windows", f"{int(user_windows['ext_predicted'].sum())} / {len(user_windows)}")

if user_windows.empty:
    st.warning("No behavioural windows were found for this user in the uploaded results.")
    st.stop()

st.subheader("Behaviour over time")
plot_df = user_windows.copy()
plot_df['window_start'] = pd.to_datetime(plot_df['window_start'], errors='coerce')
plot_df = plot_df.sort_values('window_start').dropna(subset=['window_start'])
if not plot_df.empty:
    st.line_chart(plot_df.set_index('window_start')[['ext_score']])
    st.dataframe(plot_df[['window_start','window_end','ext_score','ext_predicted']].rename(columns={
        'window_start':'Window start','window_end':'Window end','ext_score':'Anomaly score','ext_predicted':'Flagged by model'
    }), use_container_width=True, hide_index=True)

st.subheader("Evidence-based explanation")
valid_features = [c for c in FEATURES if c in train_df.columns and c in user_windows.columns]
if not valid_features:
    st.warning("No compatible feature columns were found for evidence comparison.")
else:
    # Reference statistics come only from the normal-only model training windows.
    normal_ref = train_df[valid_features].apply(pd.to_numeric, errors='coerce')
    ref_mean = normal_ref.mean()
    ref_std = normal_ref.std().replace(0, np.nan)
    # Use the user's highest-scoring window for an interpretable snapshot.
    selected_window = user_windows.loc[pd.to_numeric(user_windows['ext_score'], errors='coerce').idxmax()]
    observed = pd.to_numeric(selected_window[valid_features], errors='coerce')
    z = ((observed - ref_mean) / ref_std).replace([np.inf, -np.inf], np.nan).fillna(0)
    evidence = pd.DataFrame({
        'Feature': [FRIENDLY.get(c,c) for c in valid_features],
        'Observed value': [observed[c] for c in valid_features],
        'Normal-training mean': [ref_mean[c] for c in valid_features],
        'Absolute standardized difference': [abs(z[c]) for c in valid_features],
        '_key': valid_features,
    }).sort_values('Absolute standardized difference', ascending=False)
    top = evidence.head(6).copy()
    st.caption("The table ranks features by absolute standardized difference from normal training windows. These are descriptive indicators, not exact feature attributions from Isolation Forest.")
    st.dataframe(top.drop(columns=['_key']).round(3), use_container_width=True, hide_index=True)

    st.subheader("Investigation report")
    label = "flagged as anomalous" if pred == 1 else "classified as normal"
    lines = [
        f"### User {chosen_user}: {label}",
        f"The Extended Isolation Forest's user-level decision is **{'ANOMALOUS' if pred == 1 else 'NORMAL'}**.",
        f"The maximum observed window anomaly score is **{score:.4f}**. Higher scores indicate more unusual behaviour within this experiment; the score is not a probability.",
        f"Across {len(user_windows)} available windows, {int(user_windows['ext_predicted'].sum())} were flagged by the model.",
        "**Largest differences from normal training behaviour in the highest-scoring window:**"
    ]
    for _, row in top.head(4).iterrows():
        lines.append(f"- {row['Feature']}: observed {row['Observed value']:.3f}; normal-training mean {row['Normal-training mean']:.3f}; absolute standardized difference {row['Absolute standardized difference']:.2f}.")
    lines.append("**Interpretation caveat:** These differences provide context for review; they do not prove intent and are not a causal explanation of the Isolation Forest decision. Review the underlying activity and applicable policy before drawing conclusions.")
    report = "\n\n".join(lines)
    st.markdown(report)
    st.download_button("Download investigation report (.md)", report, file_name=f"investigation_{chosen_user}.md", mime="text/markdown")

st.subheader("Evaluation context")
if {'true_label','user_pred'}.issubset(users_df.columns):
    y_true = pd.to_numeric(users_df['true_label'], errors='coerce').fillna(0).astype(int)
    y_pred = users_df['user_pred'].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    a,b,c,d = st.columns(4)
    a.metric("True positives", tp); b.metric("False positives", fp)
    c.metric("True negatives", tn); d.metric("False negatives", fn)
    st.caption("Ground-truth labels are for evaluating this CERT test set only. In a real deployment, they would not be available at prediction time.")
else:
    st.caption("To show the confusion matrix, include the `true_label` column in the user-level results CSV.")
