# CERT r4.2 Extended Isolation Forest — Methodology

## Model scope
The Streamlit demo uses the existing Extended Isolation Forest experiment with 23 behavioural features: 19 baseline features and four extension features. It must not substitute the separate 31-feature historical model.

## Windowing
The current FAST notebook configuration uses seven-day windows with a seven-day slide. Therefore, successive windows are consecutive and non-overlapping. The displayed app reads already-computed predictions from CSV files; it does not retrain or re-score data.

## Training and scaling
The model is fitted using training windows from normal users only. A StandardScaler is fitted on the training feature matrix and then applied to test windows. The Streamlit demo does not use the saved model or scaler for its current precomputed-results workflow.

## Anomaly score
The notebook computes the anomaly score as the negative of IsolationForest.decision_function. Larger values therefore indicate more unusual observations within this experiment. The score is not a probability or calibrated risk percentage. The model's prediction threshold, not score sign alone, determines whether a window is flagged.

## User-level decision
A user is flagged if at least one of that user's windows is flagged. The user-level maximum anomaly score is the maximum window score for that user. The dashboard's evidence comparison uses the user's highest-scoring window.

## Evidence table limitations
The dashboard ranks feature values by absolute standardized difference from the means and standard deviations of the normal training windows. This gives descriptive context. It is not an exact Isolation Forest feature attribution, a causal explanation, or proof of malicious intent. A feature can be unusual because its observed value is higher or lower than normal training behaviour.

## Evaluation labels
Ground-truth labels are available in the CERT test data for retrospective evaluation only. They would not be available when predicting an unknown user's behaviour in a real deployment. Do not use a user's ground-truth label to construct the AI explanation.

## Cross-source overlap caveat
exfil_overlap_days captures same-day co-occurrence between USB connection activity and activity on domains classified by configured keyword lists. Co-occurrence does not prove sequence, file transfer, exfiltration, intent, or a causal relationship.

## Investigation guidance
Treat an anomaly as a review signal, not proof of malicious intent. Validate source logs, timestamps, user role, approved business activity, and applicable policy before making a conclusion. State uncertainty and distinguish observed facts from hypotheses.
