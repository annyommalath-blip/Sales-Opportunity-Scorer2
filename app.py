from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import importlib


def _safe_import(module_name: str, alias: str | None = None):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


plt = _safe_import("matplotlib.pyplot")
joblib = _safe_import("joblib")
np = _safe_import("numpy")
pd = _safe_import("pandas")
px = _safe_import("plotly.express")
go = _safe_import("plotly.graph_objects")
shap = _safe_import("shap")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
FIG_DIR = BASE_DIR / "reports" / "figures"
DATA_PATH = BASE_DIR / "data" / "raw" / "Sales_Pipeline_SaaS_Startup.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "Decision Tree": "decision_tree_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
    "MLP (Neural Net)": "mlp_pipeline.joblib",
}

PALETTE = {
    "bg": "#f4f7fb",
    "card": "#ffffff",
    "ink": "#0f172a",
    "muted": "#475569",
    "brand": "#0b5d57",
    "brand_2": "#0f766e",
    "accent": "#1d4ed8",
    "success": "#15803d",
    "warning": "#b45309",
    "border": "#dbe4ee",
}


def _missing_runtime_packages() -> list[str]:
    missing = []
    checks = {
        "joblib": joblib,
        "numpy": np,
        "pandas": pd,
        "plotly": px,
        "matplotlib": plt,
        "shap": shap,
    }
    for name, mod in checks.items():
        if mod is None:
            missing.append(name)
    return missing


def inject_premium_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

            .stApp {{
                font-family: "Manrope", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: {PALETTE['ink']};
                background:
                    radial-gradient(circle at 10% 0%, #ddeffd 0%, transparent 24%),
                    radial-gradient(circle at 95% 15%, #d8f6f1 0%, transparent 18%),
                    {PALETTE['bg']};
            }}

            .hero {{
                position: relative;
                overflow: hidden;
                background: linear-gradient(135deg, #0f172a 0%, #0a2647 48%, #0b5d57 100%);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 18px;
                padding: 1.3rem 1.4rem;
                margin: 0.35rem 0 1.0rem 0;
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.22);
            }}

            .hero:after {{
                content: "";
                position: absolute;
                top: -110px;
                right: -70px;
                width: 240px;
                height: 240px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, transparent 62%);
            }}

            .hero h1 {{
                margin: 0;
                color: #f8fbff;
                font-size: 1.62rem;
                font-weight: 800;
                letter-spacing: -0.01em;
            }}

            .hero p {{
                margin: 0.45rem 0 0 0;
                color: #e2ecff;
                font-size: 0.98rem;
                line-height: 1.45;
                max-width: 1000px;
            }}

            .kpi-card {{
                background: {PALETTE['card']};
                border: 1px solid {PALETTE['border']};
                border-radius: 14px;
                padding: 0.72rem 0.85rem;
                box-shadow: 0 5px 14px rgba(2, 8, 23, 0.05);
                min-height: 92px;
            }}

            .kpi-label {{
                font-size: 0.78rem;
                color: {PALETTE['muted']};
                margin-bottom: 0.15rem;
            }}

            .kpi-value {{
                font-size: 1.22rem;
                font-weight: 800;
                color: {PALETTE['ink']};
                margin: 0;
                letter-spacing: -0.01em;
            }}

            .section {{
                background: {PALETTE['card']};
                border: 1px solid {PALETTE['border']};
                border-radius: 16px;
                padding: 1rem 1rem 0.9rem 1rem;
                margin-bottom: 0.9rem;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
            }}

            .section h3 {{
                margin: 0;
                font-size: 1.05rem;
                color: {PALETTE['ink']};
                letter-spacing: -0.01em;
            }}

            .section-sub {{
                margin-top: 0.28rem;
                color: {PALETTE['muted']};
                font-size: 0.9rem;
            }}

            .insight {{
                border-left: 4px solid {PALETTE['accent']};
                background: #f8fbff;
                border-radius: 10px;
                border: 1px solid #d9e7ff;
                padding: 0.7rem 0.8rem;
                margin-bottom: 0.62rem;
            }}

            .insight b {{
                color: {PALETTE['ink']};
            }}

            .pill {{
                display: inline-block;
                border-radius: 999px;
                padding: 0.2rem 0.52rem;
                font-size: 0.76rem;
                font-weight: 600;
                margin-right: 0.32rem;
                margin-bottom: 0.26rem;
                border: 1px solid #c5d6e8;
                color: #1e3a5f;
                background: #f3f8ff;
            }}

            .factor-pos {{
                border-color: #b7efcb;
                background: #f2fff6;
                color: #166534;
            }}

            .factor-neg {{
                border-color: #ffd0d0;
                background: #fff5f5;
                color: #991b1b;
            }}

            [data-testid="stTabs"] button {{
                font-weight: 700;
                font-size: 0.9rem;
                padding-top: 0.55rem;
                padding-bottom: 0.55rem;
            }}

            [data-testid="stImage"] img {{
                border-radius: 12px;
                border: 1px solid {PALETTE['border']};
            }}

            [data-testid="stMetricValue"] {{
                font-weight: 700;
            }}

            .subtle-note {{
                color: {PALETTE['muted']};
                font-size: 0.82rem;
                margin-top: 0.2rem;
            }}

            @media (max-width: 900px) {{
                .hero h1 {{font-size: 1.35rem;}}
                .kpi-value {{font-size: 1.1rem;}}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "model_comparison_metrics.csv")


@st.cache_data
def load_raw_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={c: c.replace("\n", " ").strip() for c in df.columns})
    return df


@st.cache_resource
def load_models() -> dict:
    models = {}
    for name, file_name in MODEL_FILES.items():
        path = ARTIFACT_DIR / file_name
        if path.exists():
            models[name] = joblib.load(path)
    return models


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def section_footer() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <p class="kpi-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
            <b>{title}</b><br>{body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_static_figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path.name}")


def normalize_target(df: pd.DataFrame) -> pd.Series:
    if "Opportunity Status" not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    return (df["Opportunity Status"].astype(str).str.strip().str.lower() == "won").astype(int)


def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Outcome Label"].value_counts().reindex(["Loss", "Won"])
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            marker=dict(color=["#64748b", "#0f766e"], line=dict(color="#e2e8f0", width=1)),
        )
    )
    fig.update_layout(
        title="Class Distribution",
        title_x=0,
        yaxis_title="Opportunities",
        xaxis_title="Outcome",
        height=360,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_conversion_by_segment(df: pd.DataFrame, feature: str) -> go.Figure:
    seg = (
        df.groupby(feature, as_index=False)["Converted"]
        .mean()
        .sort_values("Converted", ascending=False)
    )
    fig = px.bar(
        seg,
        x="Converted",
        y=feature,
        orientation="h",
        color="Converted",
        color_continuous_scale=["#bae6fd", "#0f766e"],
        text=seg["Converted"].map(lambda x: f"{x:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"Conversion Rate by {feature}",
        title_x=0,
        xaxis_tickformat=".0%",
        xaxis_title="Conversion Rate",
        yaxis_title="",
        height=430,
        margin=dict(l=10, r=10, t=55, b=20),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_numeric_pattern(df: pd.DataFrame, numeric_feature: str) -> go.Figure:
    fig = px.box(
        df,
        x="Outcome Label",
        y=numeric_feature,
        color="Outcome Label",
        color_discrete_map={"Loss": "#64748b", "Won": "#0f766e"},
        points=False,
    )
    fig.update_layout(
        title=f"{numeric_feature} Distribution by Outcome",
        title_x=0,
        xaxis_title="Outcome",
        yaxis_title=numeric_feature,
        height=420,
        margin=dict(l=10, r=10, t=55, b=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_numeric_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_cols = ["Sales Velocity", "Sales Stage Iterations", "Opportunity Size (USD)"]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Teal",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}<br>corr=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Numeric Correlation Heatmap",
        title_x=0,
        height=420,
        margin=dict(l=10, r=10, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_f1_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    rank = metrics_df.sort_values("f1", ascending=True)
    fig = px.bar(
        rank,
        x="f1",
        y="model",
        orientation="h",
        text=rank["f1"].map(lambda x: f"{x:.3f}"),
        color="f1",
        color_continuous_scale=["#dbeafe", "#1d4ed8"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="F1 Leaderboard",
        title_x=0,
        xaxis_title="F1 Score",
        yaxis_title="",
        height=400,
        margin=dict(l=10, r=10, t=55, b=20),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def gauge_probability(prob: float, title: str = "Win Probability") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob) * 100.0,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, 35], "color": "#fee2e2"},
                    {"range": [35, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.75,
                    "value": float(prob) * 100.0,
                },
            },
        )
    )
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def friendly_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    if "_" in cleaned and cleaned.startswith(("Technology", "City", "B2B", "Client", "Business", "Compete", "Opportunity")):
        return cleaned
    return cleaned


def build_input_row(metadata: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    defaults = metadata["defaults"]
    feature_columns = metadata["feature_columns"]
    numeric_features = set(metadata["numeric_features"])
    manual_features = metadata["manual_input_features"]

    row = {col: defaults[col] for col in feature_columns}

    for col in manual_features:
        if col not in row:
            continue

        if col in numeric_features:
            values = raw_df[col].astype(float)
            min_val = float(np.nanpercentile(values, 1))
            max_val = float(np.nanpercentile(values, 99))
            step = max((max_val - min_val) / 120.0, 1.0)
            row[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=float(np.clip(float(defaults[col]), min_val, max_val)),
                step=float(step),
            )
        else:
            options = sorted(raw_df[col].dropna().astype(str).unique().tolist())
            default = str(defaults[col])
            if default not in options and options:
                default = options[0]
            if not options:
                options = [default]
            row[col] = st.selectbox(
                col,
                options=options,
                index=options.index(default) if default in options else 0,
            )

    return pd.DataFrame([row], columns=feature_columns)


def get_single_prediction_explanation(model_pipeline, input_row: pd.DataFrame):
    pre = model_pipeline.named_steps["preprocessor"]
    clf = model_pipeline.named_steps["classifier"]
    feature_names = pre.get_feature_names_out()

    transformed = pre.transform(input_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(clf)
    shap_values_raw = explainer.shap_values(transformed)

    if isinstance(shap_values_raw, list):
        shap_values = np.array(shap_values_raw[1])[0]
        base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        arr = np.array(shap_values_raw)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            shap_values = arr[0, :, 1]
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            shap_values = arr[0]
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    explanation = shap.Explanation(
        values=shap_values,
        base_values=float(base),
        data=transformed[0],
        feature_names=feature_names,
    )

    shap_series = pd.Series(shap_values, index=feature_names)
    ranked = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(8)
    return explanation, ranked


def main() -> None:
    st.set_page_config(page_title="Sales Opportunity Scorer", page_icon="📈", layout="wide")
    inject_premium_css()

    missing_pkgs = _missing_runtime_packages()
    if missing_pkgs:
        st.error(
            "Missing runtime dependencies on this environment: "
            + ", ".join(missing_pkgs)
        )
        st.code(
            "pip install -r requirements.txt",
            language="bash",
        )
        st.info(
            "If deploying on Streamlit Cloud, confirm `requirements.txt` and `runtime.txt` are both in repo root, then reboot app."
        )
        st.stop()

    required = [
        ARTIFACT_DIR / "metadata.json",
        ARTIFACT_DIR / "model_comparison_metrics.csv",
        ARTIFACT_DIR / "best_hyperparameters.json",
        DATA_PATH,
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("Missing artifacts. Run `python train_pipeline.py` first. Missing: " + ", ".join(missing))
        st.stop()

    metadata = load_json(ARTIFACT_DIR / "metadata.json")
    metrics_df = load_metrics()
    best_params = load_json(ARTIFACT_DIR / "best_hyperparameters.json")
    models = load_models()
    raw_df = load_raw_df()

    with st.sidebar:
        st.markdown("### Workspace Controls")
        audience_mode = st.radio("View mode", options=["Executive", "Technical"], index=0)
        st.markdown("---")
        st.markdown("**Story flow**")
        st.markdown("1. Business context\n2. Data insights\n3. Model winner\n4. Decision support")
        st.markdown("---")
        st.caption("For screenshots, keep browser zoom at 100%.")

    y = normalize_target(raw_df)
    df = raw_df.copy()
    df["Converted"] = y
    df["Outcome Label"] = np.where(df["Converted"] == 1, "Won", "Loss")

    won = int((y == 1).sum())
    loss = int((y == 0).sum())
    total = len(y)
    best_row = metrics_df.sort_values("f1", ascending=False).iloc[0]

    st.markdown(
        """
        <div class="hero">
            <h1>Sales Opportunity Scorer</h1>
            <p>Enterprise-grade decision intelligence for pipeline prioritization: predict conversion likelihood, compare modeling strategies, and explain every recommendation with transparent feature-level drivers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top = st.columns(5)
    with top[0]:
        kpi_card("Total Opportunities", f"{total:,}")
    with top[1]:
        kpi_card("Conversion Rate", f"{won/total:.1%}")
    with top[2]:
        kpi_card("Best Model", str(best_row["model"]))
    with top[3]:
        kpi_card("Best F1", f"{best_row['f1']:.3f}")
    with top[4]:
        kpi_card("Best AUC", f"{best_row['auc_roc']:.3f}")

    tabs = st.tabs(
        [
            "Executive Summary",
            "Descriptive Analytics",
            "Model Performance",
            "Explainability & Interactive Prediction",
        ]
    )

    with tabs[0]:
        section_header(
            "Why This Product Matters",
            "Sales teams need prioritization they can trust. This scorer turns historical opportunity behavior into explainable, action-ready recommendations.",
        )
        left, right = st.columns([1.25, 1])
        with left:
            insight_box(
                "Mission",
                "Improve win-rate efficiency by directing seller attention toward opportunities with the highest expected conversion payoff.",
            )
            insight_box(
                "Business impact",
                "Better prioritization reduces wasted rep capacity, improves forecast confidence, and supports more consistent quarter-end performance.",
            )
        with right:
            st.markdown(
                f'<span class="pill">Won: {won:,} ({won/total:.1%})</span>'
                f'<span class="pill">Loss: {loss:,} ({loss/total:.1%})</span>'
                f'<span class="pill">Target: Opportunity Status</span>'
                f'<span class="pill">Random State: 42</span>',
                unsafe_allow_html=True,
            )
            st.metric("Recommended Production Model", best_row["model"])
            st.metric("Operational F1", f"{best_row['f1']:.3f}")
            st.metric("Discrimination (AUC)", f"{best_row['auc_roc']:.3f}")

        insight_cols = st.columns(3)
        with insight_cols[0]:
            insight_box(
                "Key finding 1",
                "Conversion likelihood varies materially by channel and solution category, suggesting differentiated coverage strategy.",
            )
        with insight_cols[1]:
            insight_box(
                "Key finding 2",
                "Tree-based models provide stronger lift while preserving interpretable decision drivers through SHAP.",
            )
        with insight_cols[2]:
            insight_box(
                "Recommendation",
                "Use the interactive scorer in weekly pipeline reviews to triage at-risk deals and prioritize high-confidence wins.",
            )
        section_footer()

    with tabs[1]:
        section_header(
            "Descriptive Intelligence",
            "Explore class balance, segment performance, and numeric relationships with interactive controls.",
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            segment_feature = st.selectbox(
                "Segment dimension",
                options=["B2B Sales Medium", "Technology Primary", "City", "Opportunity Sizing"],
                index=0,
            )
        with f2:
            numeric_feature = st.selectbox(
                "Numeric behavior",
                options=["Sales Velocity", "Sales Stage Iterations", "Opportunity Size (USD)"],
                index=0,
            )
        with f3:
            channel_filter = st.multiselect(
                "Filter channels",
                options=sorted(df["B2B Sales Medium"].dropna().unique().tolist()),
                default=sorted(df["B2B Sales Medium"].dropna().unique().tolist()),
            )

        filtered = df[df["B2B Sales Medium"].isin(channel_filter)].copy()
        if filtered.empty:
            st.warning("No rows match the selected filters. Reset channel filter.")
            filtered = df.copy()

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_target_distribution(filtered), use_container_width=True)
            st.markdown('<div class="subtle-note">Target understanding: majority-class loss distribution confirms real-world funnel attrition.</div>', unsafe_allow_html=True)
        with c2:
            st.plotly_chart(plot_conversion_by_segment(filtered, segment_feature), use_container_width=True)
            st.markdown('<div class="subtle-note">Segment pattern: conversion differs by selected dimension, guiding channel and solution strategy.</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_numeric_pattern(filtered, numeric_feature), use_container_width=True)
            st.markdown('<div class="subtle-note">Numeric behavior: outcome-linked spread indicates useful signal for predictive modeling.</div>', unsafe_allow_html=True)
        with c4:
            st.plotly_chart(plot_numeric_corr_heatmap(filtered), use_container_width=True)
            st.markdown('<div class="subtle-note">Numeric relationships: strongest linear correlations support robust feature interpretation.</div>', unsafe_allow_html=True)

        if audience_mode == "Technical":
            with st.expander("Technical notes"):
                st.write("Missing values are concentrated in `Compete Intel`; training pipeline imputes and encodes systematically.")
                st.write("Charts here use filtered data only for exploratory interaction, not retraining.")

        section_footer()

    with tabs[2]:
        section_header(
            "Model Evaluation Console",
            "Compare model quality, ranking, and operational trade-offs across the held-out test set.",
        )
        leaderboard = metrics_df.sort_values("f1", ascending=False).reset_index(drop=True)
        leaderboard.index = leaderboard.index + 1
        show_df = leaderboard.copy()
        for col in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            show_df[col] = show_df[col].map(lambda x: round(float(x), 4))
        show_df["fit_time_sec"] = show_df["fit_time_sec"].map(lambda x: round(float(x), 2))

        best = leaderboard.iloc[0]
        cards = st.columns(3)
        with cards[0]:
            kpi_card("Top Accuracy", f"{leaderboard['accuracy'].max():.3f}")
        with cards[1]:
            kpi_card("Top F1", f"{leaderboard['f1'].max():.3f}")
        with cards[2]:
            kpi_card("Top AUC", f"{leaderboard['auc_roc'].max():.3f}")

        st.dataframe(show_df, use_container_width=True)
        st.caption(f"#1 ranked model: {best['model']} (F1={best['f1']:.3f}, AUC={best['auc_roc']:.3f})")

        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(plot_f1_comparison(metrics_df), use_container_width=True)
        with r2:
            show_static_figure(FIG_DIR / "roc_curves.png", "ROC Curves Across All Models")

        insight_box(
            "Executive trade-off",
            "Ensemble models deliver stronger predictive lift, while linear/tree baselines remain easier to communicate. Recommended deployment balances performance with explainability requirements.",
        )

        if audience_mode == "Technical":
            with st.expander("Advanced technical details"):
                st.markdown("**Best hyperparameters**")
                st.json(best_params)
                st.download_button(
                    "Download Metrics CSV",
                    data=show_df.to_csv(index=False).encode("utf-8"),
                    file_name="model_comparison_metrics.csv",
                    mime="text/csv",
                )

        section_footer()

    with tabs[3]:
        section_header(
            "Explainability & Prediction Workstation",
            "Configure opportunity details, score conversion probability, and inspect local feature-level reasoning.",
        )

        if not models:
            st.error("No trained model artifacts found.")
            st.stop()

        left, right = st.columns([1.05, 1.15])

        with left:
            st.markdown("#### Opportunity Input")
            selected_model_name = st.selectbox("Scoring model", options=list(models.keys()))
            model = models[selected_model_name]
            input_row = build_input_row(metadata, raw_df.drop(columns=["Opportunity Status"], errors="ignore"))

        with right:
            probs = model.predict_proba(input_row)[0]
            pred = int(np.argmax(probs))
            pred_label = "Won" if pred == 1 else "Loss"
            confidence = probs[1] if pred == 1 else probs[0]

            st.markdown("#### Prediction Outcome")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Class", pred_label)
            with m2:
                st.metric("Win Probability", f"{probs[1]:.2%}")
            with m3:
                st.metric("Model Confidence", f"{confidence:.2%}")
            st.plotly_chart(gauge_probability(probs[1]), use_container_width=True)

            human_text = (
                "This opportunity is predicted to convert with strong confidence."
                if probs[1] >= 0.65
                else "This opportunity appears moderate-risk and may need targeted intervention."
                if probs[1] >= 0.4
                else "This opportunity is unlikely to convert without meaningful strategy changes."
            )
            insight_box("Decision narrative", human_text)

        st.markdown("#### Global Explainability")
        g1, g2 = st.columns(2)
        with g1:
            show_static_figure(FIG_DIR / "shap_beeswarm.png", "SHAP Beeswarm Summary")
        with g2:
            show_static_figure(FIG_DIR / "shap_bar.png", "SHAP Mean Absolute Impact")

        best_tree_name = metadata.get("best_tree_model", "Random Forest")
        tree_model = models.get(best_tree_name) or models.get("Random Forest") or models.get("XGBoost")

        st.markdown("#### Local Explanation for This Opportunity")
        if tree_model is not None:
            explanation, ranked = get_single_prediction_explanation(tree_model, input_row)

            st.markdown("**Top contributing factors**")
            chip_html = []
            for feat, val in ranked.items():
                cls = "factor-pos" if val >= 0 else "factor-neg"
                direction = "+" if val >= 0 else "-"
                chip_html.append(
                    f'<span class="pill {cls}">{friendly_feature_name(str(feat))}: {direction}{abs(val):.3f}</span>'
                )
            st.markdown("".join(chip_html), unsafe_allow_html=True)

            fig = plt.figure(figsize=(12, 8))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Tree-based model artifact missing; local SHAP explanation unavailable.")

        if audience_mode == "Technical":
            with st.expander("Technical notes: prediction pipeline"):
                st.write("All predictions use persisted preprocessing + model artifacts. No in-app retraining occurs.")
                st.write("Waterfall explanation uses the best tree-based model selected during training.")

        section_footer()

    st.markdown(
        '<div class="subtle-note" style="text-align:center; margin-top: 0.35rem;">Sales Opportunity Scorer | Executive ML Product Experience | Streamlit + scikit-learn + SHAP</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
