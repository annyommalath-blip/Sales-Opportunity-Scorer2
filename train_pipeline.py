import json
import os
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Sales_Pipeline_SaaS_Startup.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
FIG_DIR = BASE_DIR / "reports" / "figures"


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Sales_Pipeline_SaaS_Startup.csv under data/raw/."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset shape: {df.shape}")
    print("Columns found:")
    for col in df.columns:
        print(f" - {col}")
    return df


def infer_target_column(df: pd.DataFrame) -> str:
    candidate_targets = [
        "Opportunity Status",
        "opportunity_status",
        "target",
        "converted",
        "is_converted",
        "conversion",
    ]
    for col in candidate_targets:
        if col in df.columns:
            return col

    # Fallback heuristic: binary object column with conversion-like names
    for col in df.columns:
        lower = col.lower()
        if any(tok in lower for tok in ["status", "convert", "won", "outcome"]):
            n_unique = df[col].dropna().nunique()
            if n_unique <= 5:
                return col

    raise ValueError(
        "Could not safely infer target column. Please set target column manually in infer_target_column()."
    )


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Keep original semantics while removing newline artifacts for consistent UI handling.
    clean_map = {c: c.replace("\n", " ").strip() for c in df.columns}
    return df.rename(columns=clean_map)


def make_target_binary(df: pd.DataFrame, target_col: str) -> pd.Series:
    raw = df[target_col].astype(str).str.strip().str.lower()
    won_aliases = {"won", "win", "converted", "yes", "1", "true"}
    y = raw.isin(won_aliases).astype(int)
    if y.nunique() < 2:
        raise ValueError("Target mapping produced one class only. Check target encoding.")
    return y


def summarize_dataset(df: pd.DataFrame, target_col: str, y: pd.Series) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = [c for c in feature_cols if c in numeric_cols]
    categorical_features = [c for c in feature_cols if c not in numeric_cols]

    summary = {
        "n_rows": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "target_column": target_col,
        "positive_class_name": "Won",
        "class_distribution": {
            "Loss (0)": int((y == 0).sum()),
            "Won (1)": int((y == 1).sum()),
        },
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_cols,
    }
    return summary


def plot_target_distribution(y: pd.Series) -> None:
    plt.figure(figsize=(8, 5))
    counts = y.value_counts().sort_index()
    labels = ["Loss (0)", "Won (1)"]
    sns.barplot(x=labels, y=[counts.get(0, 0), counts.get(1, 0)], palette="viridis")
    plt.title("Target Distribution: Opportunity Status")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png", bbox_inches="tight")
    plt.close()


def plot_business_eda(df: pd.DataFrame, y: pd.Series, target_col: str) -> None:
    eda_df = df.copy()
    eda_df["Converted"] = y

    # 1) Conversion rate by sales medium
    medium = (
        eda_df.groupby("B2B Sales Medium", as_index=False)["Converted"]
        .mean()
        .sort_values("Converted", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=medium, y="B2B Sales Medium", x="Converted", palette="crest")
    plt.title("Conversion Rate by B2B Sales Medium")
    plt.xlabel("Conversion Rate")
    plt.ylabel("Sales Medium")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "conversion_by_sales_medium.png", bbox_inches="tight")
    plt.close()

    # 2) Conversion rate by technology
    tech = (
        eda_df.groupby("Technology Primary", as_index=False)["Converted"]
        .mean()
        .sort_values("Converted", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tech, y="Technology Primary", x="Converted", palette="mako")
    plt.title("Conversion Rate by Technology")
    plt.xlabel("Conversion Rate")
    plt.ylabel("Technology")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "conversion_by_technology.png", bbox_inches="tight")
    plt.close()

    # 3) Sales velocity distribution by class
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=eda_df,
        x="Converted",
        y="Sales Velocity",
        palette=["#cc4c02", "#2b8cbe"],
    )
    plt.xticks([0, 1], ["Loss", "Won"])
    plt.title("Sales Velocity by Outcome")
    plt.xlabel("Opportunity Outcome")
    plt.ylabel("Sales Velocity")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sales_velocity_by_outcome.png", bbox_inches="tight")
    plt.close()

    # 4) Conversion rate by opportunity sizing
    sizing = (
        eda_df.groupby("Opportunity Sizing", as_index=False)["Converted"]
        .mean()
        .sort_values("Converted", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sizing, y="Opportunity Sizing", x="Converted", palette="rocket")
    plt.title("Conversion Rate by Opportunity Sizing")
    plt.xlabel("Conversion Rate")
    plt.ylabel("Opportunity Sizing Bucket")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "conversion_by_opportunity_sizing.png", bbox_inches="tight")
    plt.close()

    # Correlation heatmap for numeric features
    numeric_df = eda_df.select_dtypes(include=[np.number]).drop(columns=["Converted"], errors="ignore")
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_model(model_name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict]:
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - start

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without probabilities
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc,
        "prediction_time_sec": elapsed,
    }

    roc_data = {"model": model_name, "fpr": fpr, "tpr": tpr, "auc": roc_auc}
    return metrics, roc_data


def select_positive_shap(shap_values, expected_value):
    # Handle different SHAP return formats across versions and estimators.
    if isinstance(shap_values, list):
        return np.array(shap_values[1]), expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value

    shap_arr = np.array(shap_values)

    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
        base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        return shap_arr[:, :, 1], base

    if shap_arr.ndim == 2:
        base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        return shap_arr, base

    raise ValueError(f"Unsupported SHAP output shape: {shap_arr.shape}")


def plot_roc_curves(roc_curves: list[dict]) -> None:
    plt.figure(figsize=(9, 7))
    for curve in roc_curves:
        plt.plot(curve["fpr"], curve["tpr"], label=f"{curve['model']} (AUC={curve['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curves by Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close()


def plot_f1_comparison(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.sort_values("f1", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="f1", y="model", palette="viridis")
    plt.title("F1 Score Comparison Across Models")
    plt.xlabel("F1 Score")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "f1_comparison.png", bbox_inches="tight")
    plt.close()


def plot_mlp_history(mlp_pipeline: Pipeline) -> None:
    mlp = mlp_pipeline.named_steps["classifier"]
    losses = getattr(mlp, "loss_curve_", [])
    val_scores = getattr(mlp, "validation_scores_", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses, color="#1f77b4")
    axes[0].set_title("MLP Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    if len(val_scores) > 0:
        axes[1].plot(val_scores, color="#2ca02c")
    axes[1].set_title("MLP Validation Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Score")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "mlp_training_history.png", bbox_inches="tight")
    plt.close()


def create_shap_outputs(
    best_tree_model_name: str,
    best_tree_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict:
    preprocessor = best_tree_pipeline.named_steps["preprocessor"]
    tree_model = best_tree_pipeline.named_steps["classifier"]

    # Keep SHAP computationally practical while preserving representative explanations.
    train_sample = X_train.sample(min(3000, len(X_train)), random_state=RANDOM_STATE)
    test_sample = X_test.sample(min(1200, len(X_test)), random_state=RANDOM_STATE)

    X_train_t = preprocessor.transform(train_sample)
    X_test_t = preprocessor.transform(test_sample)

    if sparse.issparse(X_train_t):
        X_train_t = X_train_t.toarray()
    if sparse.issparse(X_test_t):
        X_test_t = X_test_t.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(tree_model)
    shap_values_raw = explainer.shap_values(X_test_t)
    shap_values, base_value = select_positive_shap(shap_values_raw, explainer.expected_value)

    # SHAP beeswarm summary
    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values,
        features=X_test_t,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Beeswarm Summary ({best_tree_model_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    # SHAP mean absolute bar
    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values,
        features=X_test_t,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Mean |Value| ({best_tree_model_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_bar.png", bbox_inches="tight")
    plt.close()

    # SHAP waterfall for one interesting observation (highest predicted probability)
    probs = best_tree_pipeline.predict_proba(test_sample)[:, 1]
    row_idx = int(np.argmax(probs))

    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=base_value,
        data=X_test_t[row_idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall_single_case.png", bbox_inches="tight")
    plt.close()

    importances = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(importances)[::-1][:8]
    top_features_encoded = [str(feature_names[i]) for i in top_idx]

    shap_payload = {
        "best_tree_model_name": best_tree_model_name,
        "feature_names": [str(f) for f in feature_names],
        "base_value": float(base_value),
        "top_features_encoded": top_features_encoded,
    }

    joblib.dump(explainer, ARTIFACT_DIR / "shap_explainer.joblib")
    with open(ARTIFACT_DIR / "shap_metadata.json", "w", encoding="utf-8") as f:
        json.dump(shap_payload, f, indent=2)

    return shap_payload


def main() -> None:
    ensure_dirs()
    configure_style()

    df = load_dataset(DATA_PATH)
    df = sanitize_column_names(df)

    target_col = infer_target_column(df)
    y = make_target_binary(df, target_col)
    X = df.drop(columns=[target_col]).copy()

    # Drop identifier columns that do not carry predictive signal and may leak uniqueness.
    id_like_cols = [c for c in X.columns if "id" in c.lower()]
    X = X.drop(columns=id_like_cols, errors="ignore")

    dataset_summary = summarize_dataset(pd.concat([X, y.rename(target_col)], axis=1), target_col, y)

    print("\nDataset summary")
    print(json.dumps(dataset_summary, indent=2))

    # Part 1: descriptive analytics outputs
    plot_target_distribution(y)
    descriptive_df = pd.concat([X, y.rename(target_col)], axis=1)
    plot_business_eda(descriptive_df, y, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    imbalance_ratio = y_train.value_counts(normalize=True).max()
    use_balanced = imbalance_ratio >= 0.60
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {}
    best_params = {}
    metrics = []
    roc_curves = []

    # 1) Logistic Regression baseline
    logreg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    class_weight="balanced" if use_balanced else None,
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    logreg_pipeline.fit(X_train, y_train)
    logreg_fit_time = time.perf_counter() - t0
    m, r = evaluate_model("Logistic Regression", logreg_pipeline, X_test, y_test)
    m["fit_time_sec"] = logreg_fit_time
    models["logistic_regression"] = logreg_pipeline
    best_params["logistic_regression"] = logreg_pipeline.named_steps["classifier"].get_params()
    metrics.append(m)
    roc_curves.append(r)

    # 2) Decision Tree with GridSearchCV
    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced" if use_balanced else None,
                ),
            ),
        ]
    )
    dt_grid = {
        "classifier__max_depth": [5, 10, 20, None],
        "classifier__min_samples_split": [2, 10, 30],
        "classifier__min_samples_leaf": [1, 5, 15],
        "classifier__criterion": ["gini", "entropy"],
    }
    dt_search = GridSearchCV(
        estimator=dt_pipeline,
        param_grid=dt_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    dt_search.fit(X_train, y_train)
    dt_fit_time = time.perf_counter() - t0
    dt_best = dt_search.best_estimator_
    m, r = evaluate_model("Decision Tree", dt_best, X_test, y_test)
    m["fit_time_sec"] = dt_fit_time
    models["decision_tree"] = dt_best
    best_params["decision_tree"] = dt_search.best_params_
    metrics.append(m)
    roc_curves.append(r)

    # 3) Random Forest with GridSearchCV
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced" if use_balanced else None,
                ),
            ),
        ]
    )
    rf_grid = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 20, 35],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 3],
    }
    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    rf_search.fit(X_train, y_train)
    rf_fit_time = time.perf_counter() - t0
    rf_best = rf_search.best_estimator_
    m, r = evaluate_model("Random Forest", rf_best, X_test, y_test)
    m["fit_time_sec"] = rf_fit_time
    models["random_forest"] = rf_best
    best_params["random_forest"] = rf_search.best_params_
    metrics.append(m)
    roc_curves.append(r)

    # 4) XGBoost with GridSearchCV over >=3 hyperparameters
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight if use_balanced else 1.0,
                ),
            ),
        ]
    )
    xgb_grid = {
        "classifier__n_estimators": [200, 350],
        "classifier__max_depth": [4, 6],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }
    xgb_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    xgb_search.fit(X_train, y_train)
    xgb_fit_time = time.perf_counter() - t0
    xgb_best = xgb_search.best_estimator_
    m, r = evaluate_model("XGBoost", xgb_best, X_test, y_test)
    m["fit_time_sec"] = xgb_fit_time
    models["xgboost"] = xgb_best
    best_params["xgboost"] = xgb_search.best_params_
    metrics.append(m)
    roc_curves.append(r)

    # 5) MLP with at least two hidden layers
    mlp_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    mlp_pipeline.fit(X_train, y_train)
    mlp_fit_time = time.perf_counter() - t0
    m, r = evaluate_model("MLP (Neural Net)", mlp_pipeline, X_test, y_test)
    m["fit_time_sec"] = mlp_fit_time
    models["mlp"] = mlp_pipeline
    best_params["mlp"] = mlp_pipeline.named_steps["classifier"].get_params()
    metrics.append(m)
    roc_curves.append(r)

    plot_mlp_history(mlp_pipeline)
    plot_roc_curves(roc_curves)

    metrics_df = pd.DataFrame(metrics).sort_values("f1", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(ARTIFACT_DIR / "model_comparison_metrics.csv", index=False)
    plot_f1_comparison(metrics_df)

    with open(ARTIFACT_DIR / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)

    for name, model in models.items():
        joblib.dump(model, ARTIFACT_DIR / f"{name}_pipeline.joblib")

    # Save a dedicated fitted preprocessor object from the best model pipeline.
    best_model_key = (
        metrics_df.iloc[0]["model"].lower().replace(" ", "_").replace("(neural_net)", "")
    )
    display_to_key = {
        "logistic regression": "logistic_regression",
        "decision tree": "decision_tree",
        "random forest": "random_forest",
        "xgboost": "xgboost",
        "mlp (neural net)": "mlp",
    }
    best_key = display_to_key[metrics_df.iloc[0]["model"].lower()]
    fitted_preprocessor = models[best_key].named_steps["preprocessor"]
    joblib.dump(fitted_preprocessor, ARTIFACT_DIR / "fitted_preprocessor.joblib")

    # Best tree-based model for explainability: Random Forest vs XGBoost
    tree_metric = metrics_df[metrics_df["model"].isin(["Random Forest", "XGBoost"])].copy()
    best_tree_model_name = tree_metric.sort_values("f1", ascending=False).iloc[0]["model"]
    tree_name_to_key = {"Random Forest": "random_forest", "XGBoost": "xgboost"}
    best_tree_key = tree_name_to_key[best_tree_model_name]
    best_tree_pipeline = models[best_tree_key]

    shap_payload = create_shap_outputs(best_tree_model_name, best_tree_pipeline, X_train, X_test)

    # Persist metadata for Streamlit defaults and form rendering.
    defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            mode = X[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"

    class_balance = y.value_counts(normalize=True).to_dict()
    imbalance_note = (
        "Class imbalance detected; balanced class weights (and scale_pos_weight for XGBoost) were applied."
        if max(class_balance.values()) >= 0.60
        else "No severe class imbalance detected; standard training used."
    )

    metadata = {
        "random_state": RANDOM_STATE,
        "target_column": target_col,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "defaults": defaults,
        "class_distribution": {
            "loss_0": int((y == 0).sum()),
            "won_1": int((y == 1).sum()),
        },
        "imbalance_note": imbalance_note,
        "best_overall_model": metrics_df.iloc[0]["model"],
        "best_tree_model": best_tree_model_name,
        "top_encoded_features_from_shap": shap_payload["top_features_encoded"],
        "manual_input_features": [
            "Sales Velocity",
            "Sales Stage Iterations",
            "Opportunity Size (USD)",
            "B2B Sales Medium",
            "Technology Primary",
            "Opportunity Sizing",
            "Compete Intel",
        ],
    }

    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save detailed model reports
    reports = {}
    for display_name, key in [
        ("Logistic Regression", "logistic_regression"),
        ("Decision Tree", "decision_tree"),
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("MLP (Neural Net)", "mlp"),
    ]:
        model = models[key]
        pred = model.predict(X_test)
        reports[display_name] = classification_report(y_test, pred, output_dict=True)

    with open(ARTIFACT_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print("\nTraining complete. Artifacts saved to:")
    print(f" - {ARTIFACT_DIR}")
    print("Figures saved to:")
    print(f" - {FIG_DIR}")


if __name__ == "__main__":
    main()
