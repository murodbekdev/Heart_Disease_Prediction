# Heart Disease Prediction — Thesis Demo done by GAFFOROV MURODJON MUKHIDDIN UGLI
# Advisor : Prof. KyooJae Shin 
# Committee Member : Prof. Hee Tae Chung 
# Committee Member : Bum Mo Ryu

This research-only Streamlit application loads the protected CatBoost pipeline, isotonic calibrator and protected threshold. It does not retrain or tune anything.

## Run locally

```bash                                        my sample
python3 -m venv .venv               (heart_disease_streamlit_demo)
source .venv/bin/activate           (souce heart_disease_streamlit_demo/bin/activate)
pip install -r requirements.txt     (pip install freeze > requirements.txt)
streamlit run app.py                (streamlit run app.py)
```

## Deploy

1. Upload the files in this folder to a GitHub repository (prefer private if you do not want to distribute model artifacts).
2. In Streamlit Community Cloud choose **Create app**, select the repository, branch and `app.py`.
3. In advanced settings select Python 3.12 if available, and deploy.
4. Open the generated URL in an incognito browser, run one prediction, and check the SHAP plot.

## Notes

- `input_schema.json` stores input controls and defaults extracted from the protected final-test CSV **solely to reproduce the original app's controls**, not to fit or evaluate the model. These ranges are observed bounds, not clinical reference intervals. For stronger separation, replace with schema from development data.
- Do not upload the protected final-test CSV; it is not needed for live inference.
- The app uses raw uncalibrated scores for the frozen threshold and isotonic-calibrated probabilities for reporting.
- The serialized model should only be loaded from a trusted source. Package compatibility must be tested in the deployment environment.
- The model is research-only, not a clinical diagnostic tool.
