import streamlit as st

import pandas as pd

import numpy as np

import requests



st.set_page_config(page_title="Insider Threat Detection", page_icon="🛡️", layout="wide")


GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)
REQUEST_TIMEOUT_SECONDS = 120



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




# Feature definitions supplied to the local LLM. Where exact notebook formulas
# are not fully confirmed, the wording explicitly tells the model not to guess.
FEATURE_DEFINITIONS = {
    'logon_count': 'Count of logon records in the analysis window.',
    'logon_ah': 'Count of logon records marked as after-hours by the notebook time rule.',
    'logon_we': 'Count of logon records marked as weekend by the notebook time rule.',
    'unique_pcs': 'Number of distinct PCs associated with logon activity in the window.',
    'logon_ah_ratio': 'Ratio of after-hours logons to logons, as defined by the feature-engineering code.',
    'usb_connects': 'Count of device records whose activity is classified as connect.',
    'usb_ah': 'Count of device activity records marked as after-hours by the notebook time rule.',
    'usb_we': 'Count of device activity records marked as weekend by the notebook time rule; verify the notebook aggregation before treating these as connect-only events.',
    'file_count': 'Count of file activity records in the window; exact event types depend on source preprocessing.',
    'file_ah': 'Count of file activity records marked as after-hours by the notebook time rule; not a count of unique files unless preprocessing establishes that.',
    'file_we': 'Count of file activity records marked as weekend by the notebook time rule.',
    'email_count': 'Count of email records in the window.',
    'email_ah': 'Count of email records marked as after-hours by the notebook time rule.',
    'email_size_total': 'Aggregated email size for the window; confirm source units and aggregation in the notebook.',
    'http_count': 'Count of web/HTTP activity records in the window.',
    'job_visits': 'Count of visits classified by the notebook as job-site visits.',
    'threat_visits': 'Count of visits classified by the notebook as threat-related-site visits; this configured category is not proof of harmful intent.',
    'http_ah': 'Count of web/HTTP activity records marked as after-hours by the notebook time rule.',
    'unique_domains': 'Number of distinct web domains observed in the window.',
    'exfil_overlap_days': 'Number of days in the window with at least one USB connection and at least one activity record classified as external-site activity. Co-occurrence alone does not prove data exfiltration.',
    'usb_per_logon': 'USB connection count divided by logon count, using the notebook definition and its zero-denominator handling.',
    'ah_combined_score': 'Combined after-hours activity ratio calculated by the notebook; consult its exact formula before interpreting the scale.',
    'job_usb_interaction': 'Product of job-site visits and USB connections, as named in the notebook feature.',
}

def classify_feature_direction(observed, normal_mean, tolerance=1e-9):
    """Calculate direction deterministically; never ask the LLM to infer it."""
    difference = float(observed) - float(normal_mean)
    if abs(difference) <= tolerance:
        return "approximately equal to"
    return "higher than" if difference > 0 else "lower than"


def generate_gemini_ai_explanation(user_id, decision, max_score, flagged_windows,
                                  total_windows, top_evidence, report_text, temperature=0.1):
    evidence_lines = []
    for _, row in top_evidence.head(6).iterrows():
        observed = pd.to_numeric(pd.Series([row.get("Observed value")]), errors="coerce").iloc[0]
        mean = pd.to_numeric(pd.Series([row.get("Normal-training mean")]), errors="coerce").iloc[0]
        std_diff = pd.to_numeric(pd.Series([row.get("Absolute standardized difference")]), errors="coerce").iloc[0]
        key = str(row.get("_key", ""))
        if pd.isna(observed) or pd.isna(mean):
            continue
        direction = classify_feature_direction(observed, mean)
        definition = FEATURE_DEFINITIONS.get(key, "Definition not verified; do not infer more than the feature name supports.")
        signed_z = std_diff if observed > mean else (-std_diff if observed < mean else 0.0)
        evidence_lines.append(
            f"- Feature: {row['Feature']} (raw name: {key}); definition: {definition} "
            f"Observed={observed:.6g}; normal-training mean={mean:.6g}; direction={direction}; "
            f"raw difference={observed - mean:.6g}; signed standardized difference={signed_z:.3f}; "
            f"absolute standardized difference={std_diff:.3f} standard deviations."
        )
    evidence_text = "\n".join(evidence_lines) or "No valid feature comparisons available."

    prompt = f"""You are a practical cyber-investigation assistant for an academic CERT r4.2 insider-threat project. Your output must add meaningful interpretation beyond the rule-based report: explain in plain language WHY this selected window stands out from normal training behaviour, then give an actionable investigation plan. Do not merely list feature names and values.

TRUST AND ACCURACY RULES
- The model decision, score, and counts are fixed. Do not change them.
- Python-supplied observed values, means, directions and standardized differences are authoritative. Repeat them accurately.
- A standardized difference is (observed - normal mean) / normal standard deviation. Its signed value shows direction; its absolute value shows magnitude. Do not report raw difference as standardized difference.
- An anomaly score is not a probability. Feature deviations are descriptive context, not exact model feature attribution or proof of causation.
- Explain that the model flags the window because its overall pattern is unusual relative to learned normal behaviour; do not claim the listed features individually caused the model decision.
- Do not invent raw events, timestamps, file names, recipients, domains, policies, or investigation findings.
- Give actions that can be performed using available records. Phrase them as checks to perform, not findings already confirmed.
- Avoid copying the rule-based report sentence-for-sentence. Use the numbers as evidence within a clear narrative, not as a bare repeated list.

FIXED RESULT
User: {user_id}; decision: {decision}; max anomaly score: {max_score:.6g}; flagged windows: {flagged_windows}/{total_windows}.

VERIFIED FEATURE EVIDENCE
{evidence_text}

RULE-BASED REPORT FOR CONTEXT ONLY (DO NOT COPY ITS WORDING)
{report_text}

Write these sections, around 300-450 words:
### Why this window was flagged
Write 2-3 substantive paragraphs in accessible language. Explain the most notable pattern by connecting related observations. For example, describe how unusually high weekend logon activity (observed versus normal mean and absolute standardized difference) appears alongside unusually high weekend USB activity and repeated USB/external-site co-occurrence. Explain why that combination deserves review: it suggests the timing and co-occurrence of activities are atypical compared with normal training windows, so an analyst should check whether there is a legitimate explanation. Mention elevated email size/volume or domain count only if present in the supplied evidence. Explain standardized differences in understandable terms (e.g., several standard deviations above the training mean means far from the typical training value in this dataset), without implying probability or statistical significance. Use actual supplied values, directions, and deviations accurately. Clearly state that these are context indicators and not proof of data theft or the direct reason used by the forest.

### What the evidence may mean for an investigation
Give a short, thoughtful interpretation of how the observations relate to one another, not just another bullet list. Distinguish measured facts from possibilities. Do not assume the configured external sites are malicious, and do not assume weekend activity is unauthorized.

### Prioritized investigation actions
Give 3-5 numbered, concrete actions tailored to supplied evidence. State the record to retrieve, fields to compare, and question answered. Examples:
- Weekend logons: retrieve underlying logon records for the selected highest-scoring 7-day window; list timestamps and workstation; compare with approved schedule/on-call/change records.
- Weekend USB activity: retrieve device connect/disconnect records; inspect timestamp, device identifier/serial if recorded, host, and user; verify authorization and that source records support the aggregate count.
- USB/external-site overlap: identify the exact dates counted; compare USB events with configured external-site visits on each date. Do not claim exfiltration without file-transfer evidence.
- Email size/volume: verify source records, size units, aggregation, timestamps, attachment sizes and recipients where available.
- Distinct domains: retrieve the domain list and validate categorization and whether visits were permitted.
Choose only actions supported by the supplied evidence; no generic advice.

### Decision for the analyst
End with one sentence: treat this as a priority for human review, not as a conclusion of misconduct. Do not add a separate limitations section or a section called “Evidence Still Needed.”
"""

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "in your Streamlit app's Secrets settings."
        )

    response = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
            },
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if not response.ok:
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = response.text[:1500]
        raise RuntimeError(
            f"Gemini API returned HTTP {response.status_code}: {error_detail}"
        )
    data = response.json()

    answer = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

    if not answer:
        raise ValueError(
            f"Gemini returned no text. Response: {data}"
        )

    return answer


st.title("🛡️ Insider Threat Detection System")

st.caption("CERT r4.2 · Extended Isolation Forest · 23 behavioural features · Gemini API")

st.info("This demo explains the results of the existing experiment. An anomaly is a review signal, not proof of malicious intent.")



with st.sidebar:

    st.header("Load experiment outputs")

    st.write("Upload the CSV outputs exported from your completed Kaggle run.")

    train_file = st.file_uploader("Normal training windows CSV", type="csv", key="train")

    windows_file = st.file_uploader("Test windows with extended scores CSV", type="csv", key="windows")

    users_file = st.file_uploader("Extended user-level results CSV", type="csv", key="users")

    st.divider()

    st.caption("No prediction is fabricated when required files or columns are missing.")

    st.divider()

    st.caption(f"Gemini model: {GEMINI_MODEL}")

    ai_temperature = st.slider(
        "AI response temperature", min_value=0.0, max_value=0.2, value=0.1, step=0.05,
        help="Lower values make wording more consistent, but do not guarantee factual correctness."
    )

    st.caption("AI explanations are generated by the Google Gemini API. The selected evidence is sent to Google for processing.")



if not (train_file and windows_file and users_file):

    st.subheader("Demo setup")

    st.markdown("Upload these three files to enable user-level investigation:")

    st.markdown("1. `train_windows.csv` — normal training features.\n2. `test_windows_with_extended_scores.csv` — test windows with `ext_score` and `ext_predicted`.\n3. `extended_user_results.csv` — one row per user with the extended-model decision.")

    st.markdown("Use outputs from the same completed 23-feature Extended Isolation Forest run. Do not mix baseline-model outputs with extended-model outputs.")

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

missing_train_features = set(FEATURES) - set(train_df.columns)

if missing or missing_windows or missing_features or missing_train_features:

    st.error("The uploaded files do not match the expected extended-model output schema.")

    if missing: st.write("Missing user-result columns:", sorted(missing))

    if missing_windows: st.write("Missing window columns:", sorted(missing_windows))

    if missing_features: st.write("Missing test-window feature columns:", sorted(missing_features))

    if missing_train_features: st.write("Missing normal-training feature columns:", sorted(missing_train_features))

    st.stop()



users_df['user'] = users_df['user'].astype(str)

windows_df['user'] = windows_df['user'].astype(str)

users_df['user_pred'] = pd.to_numeric(users_df['user_pred'], errors='coerce').fillna(0).astype(int)

windows_df['ext_predicted'] = pd.to_numeric(windows_df['ext_predicted'], errors='coerce').fillna(0).astype(int)

users_df['max_ext_score'] = pd.to_numeric(users_df['max_ext_score'], errors='coerce')



left, right = st.columns([1, 2])

with left:

    chosen_user = st.selectbox("Select a user", sorted(users_df['user'].unique()))

user_row = users_df.loc[users_df['user'] == chosen_user].iloc[0]

user_windows = windows_df.loc[windows_df['user'] == chosen_user].copy()

pred = int(user_row['user_pred'])

score = float(user_row['max_ext_score']) if pd.notna(user_row['max_ext_score']) else float("nan")



with right:

    st.subheader(f"Investigation: {chosen_user}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Model decision", "ANOMALOUS" if pred == 1 else "NORMAL")

    c2.metric("Maximum anomaly score", f"{score:.4f}" if np.isfinite(score) else "Unavailable")

    flagged_count = int(user_windows['ext_predicted'].sum())

    c3.metric("Flagged windows", f"{flagged_count} / {len(user_windows)}")



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

        'window_start':'Window start','window_end':'Window end',

        'ext_score':'Anomaly score','ext_predicted':'Flagged by model'

    }),     width="stretch",
    hide_index=True)



st.subheader("Evidence-based explanation")

valid_features = [c for c in FEATURES if c in train_df.columns and c in user_windows.columns]

top = pd.DataFrame(columns=['Feature','Observed value','Normal-training mean','Absolute standardized difference','Direction vs normal mean','Signed difference','_key'])

if not valid_features:

    st.warning("No compatible feature columns were found for evidence comparison.")

else:

    normal_ref = train_df[valid_features].apply(pd.to_numeric, errors='coerce')

    ref_mean = normal_ref.mean()

    ref_std = normal_ref.std().replace(0, np.nan)

    numeric_scores = pd.to_numeric(user_windows['ext_score'], errors='coerce')

    if numeric_scores.notna().any():

        selected_window = user_windows.loc[numeric_scores.idxmax()]

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

        st.caption("Features are ranked by absolute standardized difference from normal training windows. These are descriptive indicators, not exact feature attributions from Isolation Forest.")
        with st.expander("Feature dictionary used by the AI"):
            st.write("Definitions are based on available notebook notes and feature names. Where the exact formula or preprocessing is not confirmed, the AI is instructed not to guess.")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Feature": k,
                        "Definition supplied to AI": FEATURE_DEFINITIONS.get(
                            k, "Definition not available"
                        ),
                    }
                    for k in valid_features
                ]),
                width="stretch",
                hide_index=True,
            )

        st.dataframe(
    top.drop(columns=["_key"]).round(3),
    width="stretch",
    hide_index=True,
)
    else:

        st.warning("Anomaly scores are not numeric for this user's windows, so the evidence snapshot cannot be selected.")



st.subheader("Investigation report")

label = "flagged as anomalous" if pred == 1 else "classified as normal"

lines = [

    f"### User {chosen_user}: {label}",

    f"The Extended Isolation Forest's user-level decision is **{'ANOMALOUS' if pred == 1 else 'NORMAL'}**.",

    f"The maximum observed window anomaly score is **{score:.4f}**. Higher scores indicate more unusual behaviour within this experiment; the score is not a probability." if np.isfinite(score) else "The maximum anomaly score was unavailable in the uploaded results.",

    f"Across {len(user_windows)} available windows, {flagged_count} were flagged by the model.",

    "**Largest differences from normal training behaviour in the highest-scoring window:**"

]

for _, row in top.head(4).iterrows():

    obs = pd.to_numeric(pd.Series([row['Observed value']]), errors='coerce').iloc[0]

    mean = pd.to_numeric(pd.Series([row['Normal-training mean']]), errors='coerce').iloc[0]

    diff = pd.to_numeric(pd.Series([row['Absolute standardized difference']]), errors='coerce').iloc[0]

    lines.append(f"- {row['Feature']}: observed {obs:.3f}; normal-training mean {mean:.3f}; absolute standardized difference {diff:.2f}.")

lines.append("**Interpretation caveat:** These differences provide context for review; they do not prove intent and are not a causal explanation of the Isolation Forest decision. Review underlying activity and applicable policy.")

report = "\n\n".join(lines)

st.markdown(report)

st.download_button("Download investigation report (.md)", report, file_name=f"investigation_{chosen_user}.md", mime="text/markdown")



st.divider()

st.subheader("🤖 AI Investigation Assistant (Gemini)")
st.caption("Unlike the rule-based report, the AI converts the selected window’s strongest evidence into a prioritized, feature-specific investigation checklist.")

st.caption(f"Uses `{GEMINI_MODEL}` through the Gemini API. The existing model decision and scores are not changed. Selected evidence is sent to Google Gemini to generate the explanation.")

if st.button("Generate AI investigation explanation", type="primary"):

    with st.spinner("Generating explanation with Gemini..."):

        try:

            ai_answer = generate_gemini_ai_explanation(

                user_id=chosen_user,

                decision="ANOMALOUS" if pred == 1 else "NORMAL",

                max_score=score if np.isfinite(score) else float("nan"),

                flagged_windows=flagged_count,

                total_windows=len(user_windows),

                top_evidence=top,

                report_text=report,

            )

            st.session_state[f"gemini_answer_{chosen_user}"] = ai_answer

        except requests.exceptions.Timeout:
            st.error("Gemini took too long to respond. Try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the Gemini API: {e}")
        except Exception as e:
            st.error(f"Could not generate the AI explanation: {e}")


ai_result = st.session_state.get(f"gemini_answer_{chosen_user}")

if ai_result:

    st.markdown(ai_result)

    st.download_button("Download AI explanation (.md)", ai_result, file_name=f"ai_investigation_{chosen_user}.md", mime="text/markdown", key=f"download_gemini_ai_{chosen_user}")



st.subheader("Evaluation context")

if {'true_label','user_pred'}.issubset(users_df.columns):

    y_true = pd.to_numeric(users_df['true_label'], errors='coerce').fillna(0).astype(int)

    y_pred = users_df['user_pred'].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    a,b,c,d = st.columns(4)

    a.metric("True positives", tp)

    b.metric("False positives", fp)

    c.metric("True negatives", tn)

    d.metric("False negatives", fn)

    st.caption("Ground-truth labels are for evaluating this CERT test set only. In a real deployment, they would not be available at prediction time.")

else:

    st.caption("To show the confusion matrix, include the `true_label` column in the user-level results CSV.")

from tree_explorer import render_tree_explorer

st.divider()
render_tree_explorer()