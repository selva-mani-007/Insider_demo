CERT r4.2 Insider Threat Demo — Phase 1 + Phase 2

What this app does
- Phase 1: Displays the existing 23-feature Extended Isolation Forest results from CSV files.
- Phase 2: Retrieves passages from local project documentation and asks the OpenAI API to produce a cautious, source-cited investigation explanation.
- The app does not retrain the model or alter existing predictions.

Files
- app.py: Streamlit dashboard plus AI investigation assistant.
- knowledge_base/feature_definitions.md: feature definitions from the notebook's feature engineering.
- knowledge_base/methodology.md: model/window/score interpretation and limitations.
- requirements.txt: Python packages.
- app_phase1_backup.py: copy of the previous Phase 1 app for rollback.

Windows / VS Code setup
1. Extract this folder and open it in VS Code.
2. Open Terminal > New Terminal. Ensure the terminal is in the insider_demo folder.
3. Optional but recommended: create a virtual environment:
   py -m venv .venv
   .venv\Scripts\activate
4. Install packages:
   python -m pip install -r requirements.txt
5. Set your OpenAI API key for the current PowerShell terminal (do not paste it into app.py):
   $env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
6. Optional model override:
   $env:OPENAI_MODEL="gpt-4.1-mini"
7. Start the app:
   python -m streamlit run app.py
8. Upload these three CSV files in the app sidebar:
   - train_windows.csv
   - test_windows_with_extended_scores.csv
   - extended_user_results.csv
9. Select a user, then click “Generate AI investigation explanation”.

API cost and privacy
- AI generation sends the selected user's derived feature values, score/prediction, and retrieved project documentation to the OpenAI API. Do not use sensitive production user data without organizational approval.
- API usage may incur charges. Keep the key private and never commit it to source control.
- If no API key is configured, Phase 1 still works; Phase 2 displays a setup message.

Local retrieval
- Retrieval is local keyword matching over Markdown/TXT files in knowledge_base/. No vector database is required.
- Add project methodology or feature-documentation Markdown/TXT files to knowledge_base/ to expand retrieval.
- Citations identify retrieved source file and heading. These are project-document references, not external citations.

Limitations
- The dashboard consumes precomputed predictions; it does not score new raw CERT logs.
- Standardized feature differences are descriptive, not native Isolation Forest attributions.
- An anomaly is a review signal, not proof of malicious intent.
- The ground-truth labels in the test CSV are used for retrospective evaluation only and are not sent to the AI prompt.
