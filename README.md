# Sales Opportunity Scorer

End-to-end binary classification project for the Kaggle dataset **Sales Pipeline Conversion at a SaaS Startup**.

## Project Objective
Predict whether an opportunity converts to **Won** (`1`) vs **Loss** (`0`), compare multiple ML models, explain top drivers with SHAP, and deploy a business-facing Streamlit app for deal prioritization.

## Dataset
- Source: Kaggle dataset "Sales Pipeline Conversion at a SaaS Startup"
- File used: `data/raw/Sales_Pipeline_SaaS_Startup.csv`
- Rows: ~78k
- Features: 12 predictor columns after dropping target
- Target: `Opportunity Status` (mapped to binary: Won=1, Loss=0)

## Recommended Project Structure

```text
sales_opportunity_scorer/
├── app.py
├── train_pipeline.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── data 1.zip
│       └── Sales_Pipeline_SaaS_Startup.csv
├── artifacts/
│   ├── logistic_regression_pipeline.joblib
│   ├── decision_tree_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   ├── mlp_pipeline.joblib
│   ├── fitted_preprocessor.joblib
│   ├── shap_explainer.joblib
│   ├── model_comparison_metrics.csv
│   ├── best_hyperparameters.json
│   ├── classification_reports.json
│   ├── metadata.json
│   └── shap_metadata.json
└── reports/
    └── figures/
        ├── target_distribution.png
        ├── conversion_by_sales_medium.png
        ├── conversion_by_technology.png
        ├── sales_velocity_by_outcome.png
        ├── conversion_by_opportunity_sizing.png
        ├── correlation_heatmap.png
        ├── roc_curves.png
        ├── f1_comparison.png
        ├── mlp_training_history.png
        ├── shap_beeswarm.png
        ├── shap_bar.png
        └── shap_waterfall_single_case.png
```

## Method Overview

### Part 1: Descriptive Analytics
- Dataset summary (rows, columns, feature types, target)
- Target distribution chart + imbalance assessment
- Four business EDA charts:
  - Conversion rate by `B2B Sales Medium`
  - Conversion rate by `Technology Primary`
  - Sales velocity distribution by outcome
  - Conversion rate by `Opportunity Sizing`
- Numeric correlation heatmap

### Part 2: Predictive Analytics
- Train/test split: 70/30 (`random_state=42`, stratified)
- Preprocessing:
  - Missing value imputation (`median` for numeric, `most_frequent` for categorical)
  - One-hot encoding for categorical features
  - Standard scaling for numeric features
- Models:
  1. Logistic Regression
  2. Decision Tree + GridSearchCV (5-fold)
  3. Random Forest + GridSearchCV (5-fold)
  4. XGBoost + GridSearchCV (5-fold, 5 tuned hyperparameters)
  5. MLP with two hidden layers `(128, 64)`
- Test metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- ROC curves for all models
- MLP training-history chart (loss + validation score)
- Best hyperparameters persisted

### Part 2.7: Model Comparison
- Metrics table (`artifacts/model_comparison_metrics.csv`)
- F1 comparison bar chart
- Business trade-off summary included in app

## Final Model Results (Held-out Test Set)

The following metrics are from the current trained run saved in `artifacts/model_comparison_metrics.csv`:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.8218 | 0.5805 | 0.7615 | 0.6588 | 0.8886 |
| Random Forest | 0.8291 | 0.6091 | 0.6795 | 0.6424 | 0.8631 |
| Decision Tree | 0.7677 | 0.4909 | 0.7682 | 0.5990 | 0.8514 |
| MLP (Neural Net) | 0.8406 | 0.7474 | 0.4448 | 0.5577 | 0.8398 |
| Logistic Regression | 0.7097 | 0.4085 | 0.6363 | 0.4976 | 0.7652 |

Summary: **XGBoost** is the best overall model for this run by F1 and AUC-ROC, while Random Forest is a close second on F1 with higher accuracy.

### Part 3: Explainability
- Best tree-based model selected between Random Forest and XGBoost (by test F1)
- SHAP outputs:
  - Beeswarm summary
  - Mean absolute SHAP bar plot
  - Waterfall plot for one high-probability case

### Part 4: Streamlit App
`app.py` provides exactly 4 tabs:
1. Executive Summary
2. Descriptive Analytics
3. Model Performance
4. Explainability & Interactive Prediction

The app loads saved artifacts from disk and **does not retrain**.

## How to Run

```bash
cd /Users/annyommalath/Documents/New\ project/sales_opportunity_scorer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_pipeline.py
streamlit run app.py
```

## Business Interpretation Notes (for submission write-up)
- A positive prediction means a deal has characteristics historically associated with higher conversion.
- Model outputs should support prioritization, not replace judgment.
- SHAP feature effects provide transparent rationale for why a deal is scored higher/lower.

## Known Limitations
This project uses historical opportunity-stage features, and some fields (for example, `Sales Stage Iterations`) may carry partial leakage risk if they are only fully known late in the sales cycle; in production, scoring should be restricted to features available at prediction time. Results can also drift as market behavior and sales process change, so periodic retraining/monitoring is required. Finally, the dataset is imbalanced toward losses, so threshold tuning and business-cost calibration should be validated before operational rollout.

## Manual Verification Checklist
1. Confirm target values are exactly two classes (`Won`, `Loss`) in your CSV version.
2. Check that column names match expected names after newline cleanup (`Technology Primary`, etc.).
3. Confirm Streamlit shows all generated figure files after running training.
4. Validate training runtime on your machine (GridSearch settings may take time on CPU).
5. If package versions differ, verify SHAP plot behavior (especially waterfall rendering).
6. If your instructor requires a notebook, run `train_pipeline.py` once and mirror outputs in a notebook narrative.

## Reproducibility
- `random_state=42` is used consistently.
- Model artifacts and preprocessing pipeline are persisted to `artifacts/`.
- Inference in app is deterministic given selected inputs and saved model weights.
