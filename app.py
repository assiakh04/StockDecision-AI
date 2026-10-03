
from pathlib import Path
import io
import hashlib
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="StockDecision AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PASTELS = {
    "lavender": "#CFC2F2",
    "purple": "#9A84D8",
    "mint": "#BFE8D5",
    "peach": "#F6CFB6",
    "blue": "#BFDDF4",
    "rose": "#F2C7D8",
    "cream": "#FFF8F1",
    "ink": "#2F3140",
    "muted": "#6E7180",
}

st.markdown(
    """
    <style>
    :root {
        --ink: #2F3140;
        --muted: #6E7180;
        --lavender: #EEE8FB;
        --lavender-2: #DFD5F4;
        --mint: #EAF7F1;
        --peach: #FFF1E7;
        --blue: #EDF6FC;
        --rose: #FBEFF4;
        --cream: #FFF9F4;
        --card: rgba(255,255,255,.74);
        --border: rgba(108,88,145,.15);
        --accent: #947CC8;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(207,194,242,.32), transparent 28%),
            radial-gradient(circle at 100% 10%, rgba(191,232,213,.30), transparent 25%),
            linear-gradient(135deg, #fffaf7 0%, #fbf8ff 48%, #f7fbfa 100%);
        color: var(--ink);
    }

    .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stWidgetLabel"] {
        color: var(--ink) !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(255,255,255,.50);
        backdrop-filter: blur(14px);
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(238,232,251,.98), rgba(248,244,253,.98));
        border-right: 1px solid #DED4EE;
    }

    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    .brand {
        padding: 18px 18px 14px 18px;
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(238,232,251,.94), rgba(234,247,241,.94));
        border: 1px solid rgba(148,124,200,.18);
        margin: 4px 0 18px 0;
    }

    .brand-kicker {
        font-size: .78rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #7A6B9A !important;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .brand-title {
        font-size: 2.05rem;
        line-height: 1.08;
        font-weight: 800;
        color: #33313F !important;
        margin-bottom: 8px;
    }

    .brand-sub {
        color: #6F6A78 !important;
        font-size: .98rem;
        margin: 0;
    }

    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        background: #EDF7F2;
        border: 1px solid #D0E9DD;
        font-size: .82rem;
        color: #49675A !important;
        margin-top: 8px;
    }

    .metric-card {
        height: 100%;
        min-height: 118px;
        padding: 17px 18px;
        border-radius: 20px;
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: 0 10px 30px rgba(90,72,118,.06);
        backdrop-filter: blur(10px);
    }

    .metric-label {
        color: var(--muted) !important;
        font-size: .85rem;
        font-weight: 650;
        margin-bottom: 9px;
    }

    .metric-value {
        color: var(--ink) !important;
        font-size: 2rem;
        line-height: 1;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .metric-foot {
        color: #7B7F8B !important;
        font-size: .78rem;
    }

    .section-card {
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(255,255,255,.68);
        border: 1px solid rgba(105,86,140,.14);
        box-shadow: 0 10px 26px rgba(90,72,118,.045);
        margin-bottom: 12px;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 8px;
        margin: 10px 0 20px 0;
    }

    .pipe-step {
        text-align: center;
        padding: 11px 8px;
        border-radius: 14px;
        font-size: .78rem;
        font-weight: 700;
        border: 1px solid rgba(120,100,155,.14);
        color: #484552 !important;
        min-width: 0;
    }

    .p1 { background:#F2ECFA; }
    .p2 { background:#EDF6FC; }
    .p3 { background:#EAF7F1; }
    .p4 { background:#FFF1E7; }
    .p5 { background:#FBEFF4; }
    .p6 { background:#F0EDFA; }
    .p7 { background:#EBF5F0; }

    .soft-note {
        padding: 13px 15px;
        border-radius: 14px;
        background: #FFF3E9;
        border: 1px solid #F0D9C8;
        color: #5E5149 !important;
        font-size: .9rem;
    }

    .prediction-result {
        padding: 22px;
        border-radius: 24px;
        text-align: center;
        border: 1px solid rgba(87,83,103,.12);
        box-shadow: 0 14px 30px rgba(83,72,108,.07);
    }

    .class-A { background: linear-gradient(135deg,#E7F7EF,#F3FBF7); }
    .class-B { background: linear-gradient(135deg,#FFF0E5,#FFF8F2); }
    .class-C { background: linear-gradient(135deg,#EEE8FA,#F7F3FC); }

    .pred-small {
        font-size:.83rem;
        color:#74727D !important;
        font-weight:700;
        text-transform:uppercase;
        letter-spacing:.08em;
    }

    .pred-class {
        font-size:4.3rem;
        font-weight:850;
        line-height:1;
        color:#302E3A !important;
        margin: 8px 0;
    }

    .pred-text {
        font-size:1rem;
        color:#5E5C67 !important;
    }

    /* Widgets */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.68);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 13px 15px;
    }

    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        background: rgba(255,255,255,.90) !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        border-color: #DDD2EB !important;
        border-radius: 12px !important;
    }

    [data-testid="stForm"] {
        background: rgba(255,255,255,.55);
        border: 1px solid rgba(116,92,154,.15);
        border-radius: 22px;
        padding: 18px;
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 12px !important;
        border: 1px solid #8269B7 !important;
        background: linear-gradient(135deg,#9E87D1,#876FBA) !important;
        color: white !important;
        font-weight: 750 !important;
        min-height: 43px;
    }

    .stButton > button *,
    .stDownloadButton > button *,
    [data-testid="stFormSubmitButton"] > button * {
        color:white !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.55);
        border: 1px dashed #BCAED7;
        padding: 10px;
        border-radius: 16px;
    }

    details[data-testid="stExpander"] {
        background: rgba(255,255,255,.58);
        border: 1px solid rgba(116,92,154,.14);
        border-radius: 14px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(112,92,146,.12);
        border-radius: 15px;
        overflow: hidden;
    }

    @media (max-width: 900px) {
        .pipeline {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .brand-title { font-size:1.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTES
# ============================================================

REQUIRED_COLUMNS = [
    "Risk",
    "Demand fluctuation",
    "Average stock",
    "Daily usage",
    "Unit cost",
    "Lead time",
    "Consignment stock",
    "Unit size",
]

CATEGORICAL = [
    "Risk",
    "Demand fluctuation",
    "Consignment stock",
    "Unit size",
]

RISK_SCORES = {"High": 0.47, "Normal": 0.35, "Low": 0.18}
DEMAND_SCORES = {
    "Increasing": 0.36,
    "Stable": 0.28,
    "Unknown": 0.20,
    "Decreasing": 0.16,
    "Ending": 0.00,
}
CONSIGNMENT_SCORES = {"No": 0.80, "Yes": 0.20}
UNIT_SIZE_SCORES = {"Large": 0.53, "Medium": 0.31, "Small": 0.13}

CRITERIA = [
    "Criticality",
    "Demand",
    "Supply",
    "Unit cost",
    "Unit_size_score",
]


# ============================================================
# CALCULS
# ============================================================

def validate_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes : " + ", ".join(missing)
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Le fichier contient des valeurs manquantes dans les colonnes utilisées."
        )

    allowed = {
        "Risk": set(RISK_SCORES),
        "Demand fluctuation": set(DEMAND_SCORES),
        "Consignment stock": set(CONSIGNMENT_SCORES),
        "Unit size": set(UNIT_SIZE_SCORES),
    }

    for col, valid in allowed.items():
        invalid = set(df[col].astype(str).unique()) - valid
        if invalid:
            raise ValueError(
                f"Modalités non reconnues dans {col} : {sorted(invalid)}"
            )


def x_over_max(s):
    maximum = s.max()
    if maximum == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s / maximum


def preprocess_and_aggregate(raw_df):
    df = raw_df.copy()

    if "ID" not in df.columns:
        df.insert(0, "ID", np.arange(1, len(df) + 1))

    df["Risk_score"] = df["Risk"].map(RISK_SCORES)
    df["Demand_fluctuation_score"] = df["Demand fluctuation"].map(DEMAND_SCORES)
    df["Consignment_score"] = df["Consignment stock"].map(CONSIGNMENT_SCORES)
    df["Unit_size_score"] = df["Unit size"].map(UNIT_SIZE_SCORES)

    df["Daily_usage_norm"] = x_over_max(df["Daily usage"])
    df["Average_stock_norm"] = x_over_max(df["Average stock"])
    df["Lead_time_norm"] = x_over_max(df["Lead time"])

    df["Criticality"] = (
        0.78 * df["Risk_score"]
        + 0.22 * df["Demand_fluctuation_score"]
    )

    df["Demand"] = (
        0.71 * df["Daily_usage_norm"]
        + 0.29 * df["Average_stock_norm"]
    )

    df["Supply"] = (
        0.75 * df["Lead_time_norm"]
        + 0.25 * df["Consignment_score"]
    )

    decision = df[CRITERIA].astype(float).copy()
    return df, decision


def critic_weights(decision):
    R = pd.DataFrame(index=decision.index)

    for col in decision.columns:
        xmin = decision[col].min()
        xmax = decision[col].max()
        R[col] = (
            0.0 if xmax == xmin
            else (decision[col] - xmin) / (xmax - xmin)
        )

    corr = R.corr().fillna(0.0)
    for col in corr.columns:
        corr.loc[col, col] = 1.0

    sigma = R.std(ddof=1)
    conflict = (1 - corr.abs()).sum(axis=1)
    C = sigma * conflict

    if C.sum() == 0:
        raise ValueError("CRITIC ne peut pas déterminer des poids sur ce fichier.")

    weights = C / C.sum()

    details = pd.DataFrame({
        "Critère": decision.columns,
        "Sigma": sigma.values,
        "Conflit": conflict.values,
        "Cj": C.values,
        "Poids": weights.values,
    })

    return weights, R, corr, details


def topsis(decision, weights):
    denom = np.sqrt((decision ** 2).sum(axis=0)).replace(0, 1)
    R = decision / denom

    V = R.mul(weights / weights.sum(), axis=1)

    A_plus = V.max(axis=0)
    A_minus = V.min(axis=0)

    S_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    RC = S_minus / (S_plus + S_minus)

    return {
        "normalized": R,
        "weighted": V,
        "ideal_positive": A_plus,
        "ideal_negative": A_minus,
        "s_plus": S_plus,
        "s_minus": S_minus,
        "score": RC,
    }


def apply_abc(scored_df):
    ranked = (
        scored_df
        .sort_values("TOPSIS_score", ascending=False)
        .reset_index(drop=True)
    )

    ranked["Rang"] = np.arange(1, len(ranked) + 1)
    ranked["Pourcentage_cumule"] = ranked["Rang"] / len(ranked) * 100

    ranked["Classe"] = np.select(
        [
            ranked["Pourcentage_cumule"] <= 20,
            ranked["Pourcentage_cumule"] <= 50,
        ],
        ["A", "B"],
        default="C",
    )

    return ranked


def prepare_ml_matrix(labeled):
    X = pd.get_dummies(
        labeled[REQUIRED_COLUMNS].copy(),
        columns=CATEGORICAL,
        drop_first=False,
        dtype=int,
    )
    y = labeled["Classe"].copy()
    return X, y


@st.cache_resource(show_spinner=False)
def train_models_from_csv(csv_bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    validate_data(raw)

    processed, decision = preprocess_and_aggregate(raw)
    weights, critic_norm, corr, critic_details = critic_weights(decision)

    t_result = topsis(decision, weights)
    scored = processed.copy()
    scored["TOPSIS_score"] = t_result["score"]
    labeled = apply_abc(scored)

    X, y = prepare_ml_matrix(labeled)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    definitions = {
        "Naive Bayes": (GaussianNB(), X_train, X_test, False),
        "SVM": (
            SVC(kernel="rbf", probability=True, random_state=42),
            X_train_scaled,
            X_test_scaled,
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=250, random_state=42),
            X_train,
            X_test,
            False,
        ),
        "ANN": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1500,
                early_stopping=False,
                random_state=42,
            ),
            X_train_scaled,
            X_test_scaled,
            True,
        ),
    }

    metrics = []
    trained = {}

    for name, (model, train_x, test_x, scaled) in definitions.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_x, y_train)

        pred = model.predict(test_x)

        metrics.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "F1 pondéré": f1_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "F1 macro": f1_score(
                y_test, pred, average="macro", zero_division=0
            ),
        })

        trained[name] = (model, scaled)

    metrics_df = (
        pd.DataFrame(metrics)
        .sort_values("F1 macro", ascending=False)
        .reset_index(drop=True)
    )

    best_name = metrics_df.loc[0, "Modèle"]
    _, best_scaled = trained[best_name]

    # Refit du modèle retenu sur toute la base labellisée
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(X)

    if best_name == "Naive Bayes":
        final_model = GaussianNB()
        final_model.fit(X, y)
        final_scaled = False

    elif best_name == "SVM":
        final_model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        )
        final_model.fit(X_full_scaled, y)
        final_scaled = True

    elif best_name == "Random Forest":
        final_model = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
        )
        final_model.fit(X, y)
        final_scaled = False

    else:
        final_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1500,
            early_stopping=False,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(X_full_scaled, y)
        final_scaled = True

    return {
        "raw": raw,
        "processed": processed,
        "decision": decision,
        "weights": weights,
        "critic_norm": critic_norm,
        "corr": corr,
        "critic_details": critic_details,
        "topsis": t_result,
        "labeled": labeled,
        "metrics": metrics_df,
        "best_name": best_name,
        "best_f1": float(metrics_df.loc[0, "F1 macro"]),
        "model": final_model,
        "scaled": final_scaled,
        "scaler": full_scaler if final_scaled else None,
        "feature_columns": list(X.columns),
        "class_counts": (
            labeled["Classe"].value_counts().reindex(["A", "B", "C"])
        ),
    }


def csv_bytes_from_default():
    return (BASE_DIR / "inventory_data.csv").read_bytes()


def pastel_bar(data, x, y, title, color="#9A84D8"):
    chart = (
        alt.Chart(data)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
            color=color,
        )
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            tooltip=list(data.columns),
        )
        .properties(height=300, title=title)
    )
    return chart


# ============================================================
# SIDEBAR : IMPORT CSV
# ============================================================

st.sidebar.markdown("## StockDecision AI")
st.sidebar.caption("MCDM + Machine Learning")

uploaded = st.sidebar.file_uploader(
    "Importer votre fichier CSV",
    type=["csv"],
    help="Le fichier doit contenir les 8 variables du devoir.",
)

if uploaded is not None:
    active_bytes = uploaded.getvalue()
    data_source = uploaded.name
    source_text = "CSV importé"
else:
    active_bytes = csv_bytes_from_default()
    data_source = "inventory_data.csv"
    source_text = "Base par défaut"

try:
    with st.spinner("Préparation de l'étude et du modèle..."):
        study = train_models_from_csv(active_bytes)
except Exception as exc:
    st.error(f"Impossible d'utiliser ce fichier : {exc}")
    st.info(
        "Vérifiez que le CSV contient bien les 8 colonnes attendues "
        "et utilisez la version corrigée de l'application."
    )
    st.stop()

raw_df = study["raw"]
processed_df = study["processed"]
decision_matrix = study["decision"]
abc_df = study["labeled"]
weights = study["weights"]
metrics = study["metrics"]

st.sidebar.success(f"{source_text} · {len(raw_df)} articles")
st.sidebar.caption(data_source)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Données",
        "Prétraitement",
        "Critères agrégés",
        "Pondération CRITIC",
        "TOPSIS",
        "Classification ABC",
        "Machine Learning",
        "Prédire un article",
        "Approche floue",
        "Export",
    ],
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="brand">
      <div class="brand-kicker">Decision Intelligence</div>
      <div class="brand-title">StockDecision AI</div>
      <p class="brand-sub">
        Une application interactive pour transformer une étude CRITIC–TOPSIS–ABC
        en moteur de classification automatique des articles.
      </p>
      <div class="source-pill">● {source_text} · {len(raw_df)} lignes actives</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.subheader("Vue d'ensemble")

    st.markdown(
        """
        <div class="pipeline">
          <div class="pipe-step p1">Données</div>
          <div class="pipe-step p2">Prétraitement</div>
          <div class="pipe-step p3">Agrégation</div>
          <div class="pipe-step p4">CRITIC</div>
          <div class="pipe-step p5">TOPSIS</div>
          <div class="pipe-step p6">ABC</div>
          <div class="pipe-step p7">ML</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Articles</div>
              <div class="metric-value">{len(raw_df)}</div>
              <div class="metric-foot">dans la base active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Méthode de pondération</div>
              <div class="metric-value" style="font-size:1.55rem">CRITIC</div>
              <div class="metric-foot">poids calculés automatiquement</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Modèle retenu</div>
              <div class="metric-value" style="font-size:1.55rem">{study["best_name"]}</div>
              <div class="metric-foot">sélectionné par F1 macro</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">F1 macro</div>
              <div class="metric-value">{study["best_f1"]:.3f}</div>
              <div class="metric-foot">jeu de test</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Poids CRITIC")
        wdf = pd.DataFrame({
            "Critère": [c.replace("_score", "") for c in weights.index],
            "Poids": weights.values,
        })

        chart = (
            alt.Chart(wdf)
            .mark_bar(
                cornerRadiusTopRight=8,
                cornerRadiusBottomRight=8,
                color=PASTELS["purple"],
            )
            .encode(
                x=alt.X("Poids:Q", title="Poids"),
                y=alt.Y("Critère:N", sort="-x", title=None),
                tooltip=[
                    "Critère:N",
                    alt.Tooltip("Poids:Q", format=".4f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown("### Répartition ABC")
        cdf = pd.DataFrame({
            "Classe": ["A", "B", "C"],
            "Nombre": [
                int(study["class_counts"]["A"]),
                int(study["class_counts"]["B"]),
                int(study["class_counts"]["C"]),
            ],
        })

        donut = (
            alt.Chart(cdf)
            .mark_arc(innerRadius=72, outerRadius=120)
            .encode(
                theta=alt.Theta("Nombre:Q"),
                color=alt.Color(
                    "Classe:N",
                    scale=alt.Scale(
                        domain=["A", "B", "C"],
                        range=[
                            PASTELS["mint"],
                            PASTELS["peach"],
                            PASTELS["lavender"],
                        ],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=["Classe:N", "Nombre:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(donut, use_container_width=True)

    st.markdown("### Performance des modèles")
    metric_choice = st.selectbox(
        "Indicateur à visualiser",
        ["F1 macro", "Accuracy", "Balanced Accuracy", "F1 pondéré"],
        index=0,
    )

    perf = metrics[["Modèle", metric_choice]].copy()
    perf_chart = (
        alt.Chart(perf)
        .mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8,
            color=PASTELS["blue"],
        )
        .encode(
            x=alt.X("Modèle:N", sort="-y", title=None),
            y=alt.Y(
                f"{metric_choice}:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=metric_choice,
            ),
            tooltip=[
                "Modèle:N",
                alt.Tooltip(f"{metric_choice}:Q", format=".4f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(perf_chart, use_container_width=True)


# ============================================================
# DONNÉES
# ============================================================

elif page == "Données":
    st.subheader("Explorateur de données")

    c1, c2, c3 = st.columns(3)
    c1.metric("Articles", len(raw_df))
    c2.metric("Variables", len(REQUIRED_COLUMNS))
    c3.metric("Valeurs manquantes", int(raw_df[REQUIRED_COLUMNS].isna().sum().sum()))

    f1, f2 = st.columns(2)

    with f1:
        risk_filter = st.multiselect(
            "Filtrer par Risk",
            sorted(raw_df["Risk"].unique()),
            default=[],
        )

    with f2:
        size_filter = st.multiselect(
            "Filtrer par Unit size",
            sorted(raw_df["Unit size"].unique()),
            default=[],
        )

    view = raw_df.copy()

    if risk_filter:
        view = view[view["Risk"].isin(risk_filter)]
    if size_filter:
        view = view[view["Unit size"].isin(size_filter)]

    st.caption(f"{len(view)} article(s) affiché(s)")
    st.dataframe(
        view,
        use_container_width=True,
        height=470,
    )

    st.markdown("### Nature des variables")
    variable_types = pd.DataFrame({
        "Variable": REQUIRED_COLUMNS,
        "Nature": [
            "Qualitative", "Qualitative", "Quantitative", "Quantitative",
            "Quantitative", "Quantitative", "Qualitative", "Qualitative",
        ],
    })
    st.dataframe(variable_types, hide_index=True, use_container_width=True)


# ============================================================
# PRÉTRAITEMENT
# ============================================================

elif page == "Prétraitement":
    st.subheader("Transformation des variables")

    st.markdown(
        """
        <div class="soft-note">
        Les variables qualitatives sont converties avec les scores du devoir.
        La mise à l'échelle des variables numériques utilisée pour les agrégats
        est distincte de la normalisation vectorielle TOPSIS.
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(
        ["Risk", "Demand fluctuation", "Consignment", "Unit size"]
    )

    with t1:
        st.dataframe(
            pd.DataFrame(RISK_SCORES.items(), columns=["Modalité", "Score"]),
            hide_index=True,
            use_container_width=True,
        )

    with t2:
        st.dataframe(
            pd.DataFrame(DEMAND_SCORES.items(), columns=["Modalité", "Score"]),
            hide_index=True,
            use_container_width=True,
        )

    with t3:
        st.dataframe(
            pd.DataFrame(CONSIGNMENT_SCORES.items(), columns=["Modalité", "Score"]),
            hide_index=True,
            use_container_width=True,
        )

    with t4:
        st.dataframe(
            pd.DataFrame(UNIT_SIZE_SCORES.items(), columns=["Modalité", "Score"]),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Résultat du prétraitement")
    cols = [
        "ID",
        "Risk", "Risk_score",
        "Demand fluctuation", "Demand_fluctuation_score",
        "Average stock", "Average_stock_norm",
        "Daily usage", "Daily_usage_norm",
        "Lead time", "Lead_time_norm",
        "Consignment stock", "Consignment_score",
        "Unit size", "Unit_size_score",
    ]
    st.dataframe(processed_df[cols].head(80), use_container_width=True)


# ============================================================
# AGRÉGATION
# ============================================================

elif page == "Critères agrégés":
    st.subheader("Construction des critères agrégés")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("**Criticality**\n\n0.78 × Risk + 0.22 × Demand fluctuation")
    with c2:
        st.info("**Demand**\n\n0.71 × Daily usage + 0.29 × Average stock")
    with c3:
        st.info("**Supply**\n\n0.75 × Lead time + 0.25 × Consignment")

    st.dataframe(
        processed_df[["ID", "Criticality", "Demand", "Supply"]].head(100),
        use_container_width=True,
    )

    agg_long = (
        processed_df[["Criticality", "Demand", "Supply"]]
        .melt(var_name="Critère", value_name="Valeur")
    )

    hist = (
        alt.Chart(agg_long)
        .mark_bar(opacity=.72)
        .encode(
            x=alt.X("Valeur:Q", bin=alt.Bin(maxbins=25)),
            y=alt.Y("count():Q", title="Nombre d'articles"),
            color=alt.Color(
                "Critère:N",
                scale=alt.Scale(
                    range=[
                        PASTELS["purple"],
                        PASTELS["blue"],
                        PASTELS["mint"],
                    ]
                ),
            ),
            tooltip=["Critère:N", "count():Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(hist, use_container_width=True)


# ============================================================
# CRITIC
# ============================================================

elif page == "Pondération CRITIC":
    st.subheader("Pondération objective par CRITIC")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Poids calculés")
        st.dataframe(
            study["critic_details"].round(6),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        wdf = study["critic_details"][["Critère", "Poids"]].copy()
        chart = (
            alt.Chart(wdf)
            .mark_bar(
                cornerRadiusTopRight=8,
                cornerRadiusBottomRight=8,
                color=PASTELS["purple"],
            )
            .encode(
                x=alt.X("Poids:Q", title="Poids"),
                y=alt.Y("Critère:N", sort="-x", title=None),
                tooltip=["Critère:N", alt.Tooltip("Poids:Q", format=".5f")],
            )
            .properties(height=290)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Carte de corrélation")

    corr = study["corr"].copy()
    corr_long = (
        corr.reset_index(names="Critère 1")
        .melt(id_vars="Critère 1", var_name="Critère 2", value_name="Corrélation")
    )

    heatmap = (
        alt.Chart(corr_long)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Critère 2:N", title=None),
            y=alt.Y("Critère 1:N", title=None),
            color=alt.Color(
                "Corrélation:Q",
                scale=alt.Scale(
                    domain=[-1, 0, 1],
                    range=[PASTELS["blue"], "#FFFDFB", PASTELS["rose"]],
                ),
            ),
            tooltip=[
                "Critère 1:N",
                "Critère 2:N",
                alt.Tooltip("Corrélation:Q", format=".3f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(heatmap, use_container_width=True)


# ============================================================
# TOPSIS
# ============================================================

elif page == "TOPSIS":
    st.subheader("Classement TOPSIS")

    st.latex(
        r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}"
    )

    top_n = st.slider(
        "Nombre d'articles à afficher",
        min_value=10,
        max_value=min(100, len(abc_df)),
        value=min(30, len(abc_df)),
        step=5,
    )

    top = abc_df.head(top_n).copy()

    rank_chart = (
        alt.Chart(top)
        .mark_circle(size=120, color=PASTELS["purple"])
        .encode(
            x=alt.X("Rang:Q", title="Rang TOPSIS"),
            y=alt.Y("TOPSIS_score:Q", title="Coefficient de proximité"),
            tooltip=[
                "ID:Q",
                "Rang:Q",
                alt.Tooltip("TOPSIS_score:Q", format=".5f"),
                "Classe:N",
            ],
        )
        .interactive()
        .properties(height=350)
    )
    st.altair_chart(rank_chart, use_container_width=True)

    st.dataframe(
        top[
            [
                "ID",
                "Rang",
                "TOPSIS_score",
                "Criticality",
                "Demand",
                "Supply",
                "Unit cost",
                "Unit_size_score",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Voir les étapes intermédiaires TOPSIS"):
        tabs = st.tabs(
            ["Matrice normalisée", "Matrice pondérée", "Idéaux", "Distances"]
        )

        with tabs[0]:
            st.dataframe(
                study["topsis"]["normalized"].head(50),
                use_container_width=True,
            )
            verification = (
                study["topsis"]["normalized"] ** 2
            ).sum(axis=0)
            st.caption("Somme des carrés des colonnes normalisées")
            st.dataframe(
                verification.round(6).to_frame("Somme"),
                use_container_width=True,
            )

        with tabs[1]:
            st.dataframe(
                study["topsis"]["weighted"].head(50),
                use_container_width=True,
            )

        with tabs[2]:
            ideals = pd.DataFrame({
                "Critère": CRITERIA,
                "Idéal positif": study["topsis"]["ideal_positive"].values,
                "Idéal négatif": study["topsis"]["ideal_negative"].values,
            })
            st.dataframe(ideals, hide_index=True, use_container_width=True)

        with tabs[3]:
            d = pd.DataFrame({
                "ID": processed_df["ID"],
                "S+": study["topsis"]["s_plus"],
                "S-": study["topsis"]["s_minus"],
                "Score": study["topsis"]["score"],
            })
            st.dataframe(d.head(80), hide_index=True, use_container_width=True)


# ============================================================
# ABC
# ============================================================

elif page == "Classification ABC":
    st.subheader("Classification ABC")

    c1, c2, c3 = st.columns(3)
    c1.metric("Classe A", int(study["class_counts"]["A"]), "20 % premiers")
    c2.metric("Classe B", int(study["class_counts"]["B"]), "30 % suivants")
    c3.metric("Classe C", int(study["class_counts"]["C"]), "50 % restants")

    cdf = pd.DataFrame({
        "Classe": ["A", "B", "C"],
        "Nombre": [
            int(study["class_counts"]["A"]),
            int(study["class_counts"]["B"]),
            int(study["class_counts"]["C"]),
        ],
    })

    donut = (
        alt.Chart(cdf)
        .mark_arc(innerRadius=70)
        .encode(
            theta="Nombre:Q",
            color=alt.Color(
                "Classe:N",
                scale=alt.Scale(
                    domain=["A", "B", "C"],
                    range=[
                        PASTELS["mint"],
                        PASTELS["peach"],
                        PASTELS["lavender"],
                    ],
                ),
            ),
            tooltip=["Classe:N", "Nombre:Q"],
        )
        .properties(height=330)
    )
    st.altair_chart(donut, use_container_width=True)

    class_filter = st.multiselect(
        "Afficher uniquement les classes",
        ["A", "B", "C"],
        default=["A", "B", "C"],
    )

    st.dataframe(
        abc_df[
            abc_df["Classe"].isin(class_filter)
        ][["ID", "Rang", "TOPSIS_score", "Classe"]],
        hide_index=True,
        use_container_width=True,
        height=420,
    )


# ============================================================
# ML
# ============================================================

elif page == "Machine Learning":
    st.subheader("Apprentissage supervisé")

    st.markdown(
        """
        <div class="soft-note">
        Les classes ABC construites par l'étude MCDM deviennent les labels.
        Le modèle apprend ensuite à prédire A, B ou C à partir des 8 variables
        originales de l'article.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(metrics.round(4), hide_index=True, use_container_width=True)

    metric_choice = st.selectbox(
        "Comparer les modèles selon",
        ["F1 macro", "Accuracy", "Balanced Accuracy", "F1 pondéré"],
    )

    p = metrics[["Modèle", metric_choice]]
    chart = (
        alt.Chart(p)
        .mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8,
            color=PASTELS["blue"],
        )
        .encode(
            x=alt.X("Modèle:N", sort="-y", title=None),
            y=alt.Y(
                f"{metric_choice}:Q",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                "Modèle:N",
                alt.Tooltip(f"{metric_choice}:Q", format=".4f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    st.success(
        f"Modèle retenu : {study['best_name']} "
        f"— F1 macro = {study['best_f1']:.4f}"
    )


# ============================================================
# PRÉDICTION
# ============================================================

elif page == "Prédire un article":
    st.subheader("Prédire la classe d'un nouvel article")

    st.markdown(
        """
        <div class="soft-note">
        Le nouvel article n'est pas ajouté au dataset.
        L'étude CRITIC–TOPSIS–ABC n'est pas recalculée.
        Le modèle entraîné prédit directement sa classe à partir des 8 caractéristiques.
        </div>
        """,
        unsafe_allow_html=True,
    )

    med = {
        col: float(raw_df[col].median())
        for col in ["Average stock", "Daily usage", "Unit cost", "Lead time"]
    }

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            risk = st.selectbox("Risk", list(RISK_SCORES.keys()))
            demand_fluctuation = st.selectbox(
                "Demand fluctuation",
                list(DEMAND_SCORES.keys()),
            )
            average_stock = st.number_input(
                "Average stock",
                min_value=0.0,
                value=med["Average stock"],
                step=1.0,
            )
            daily_usage = st.number_input(
                "Daily usage",
                min_value=0.0,
                value=med["Daily usage"],
                step=0.1,
            )

        with right:
            unit_cost = st.number_input(
                "Unit cost",
                min_value=0.0,
                value=med["Unit cost"],
                step=0.1,
            )
            lead_time = st.number_input(
                "Lead time",
                min_value=0.0,
                value=med["Lead time"],
                step=1.0,
            )
            consignment = st.selectbox(
                "Consignment stock",
                list(CONSIGNMENT_SCORES.keys()),
            )
            unit_size = st.selectbox(
                "Unit size",
                list(UNIT_SIZE_SCORES.keys()),
            )

        submitted = st.form_submit_button(
            "Analyser l'article",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        new_article = pd.DataFrame([{
            "Risk": risk,
            "Demand fluctuation": demand_fluctuation,
            "Average stock": average_stock,
            "Daily usage": daily_usage,
            "Unit cost": unit_cost,
            "Lead time": lead_time,
            "Consignment stock": consignment,
            "Unit size": unit_size,
        }])

        X_new = pd.get_dummies(
            new_article,
            columns=CATEGORICAL,
            drop_first=False,
            dtype=int,
        )

        X_new = X_new.reindex(
            columns=study["feature_columns"],
            fill_value=0,
        )

        if study["scaled"]:
            X_input = study["scaler"].transform(X_new)
        else:
            X_input = X_new

        model = study["model"]
        pred = model.predict(X_input)[0]

        descriptions = {
            "A": "Priorité élevée — suivi rapproché recommandé.",
            "B": "Priorité intermédiaire — suivi régulier.",
            "C": "Priorité plus faible — gestion plus légère.",
        }

        st.markdown(
            f"""
            <div class="prediction-result class-{pred}">
              <div class="pred-small">Classe prédite</div>
              <div class="pred-class">{pred}</div>
              <div class="pred-text">{descriptions[pred]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_input)[0]

            p_df = pd.DataFrame({
                "Classe": model.classes_,
                "Probabilité": probabilities,
            })

            prob_chart = (
                alt.Chart(p_df)
                .mark_bar(
                    cornerRadiusTopLeft=8,
                    cornerRadiusTopRight=8,
                )
                .encode(
                    x=alt.X("Classe:N", title=None),
                    y=alt.Y(
                        "Probabilité:Q",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    color=alt.Color(
                        "Classe:N",
                        scale=alt.Scale(
                            domain=["A", "B", "C"],
                            range=[
                                PASTELS["mint"],
                                PASTELS["peach"],
                                PASTELS["lavender"],
                            ],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        "Classe:N",
                        alt.Tooltip("Probabilité:Q", format=".1%"),
                    ],
                )
                .properties(height=280)
            )

            st.markdown("### Niveau de confiance")
            st.altair_chart(prob_chart, use_container_width=True)

        with st.expander("Voir les caractéristiques analysées"):
            st.dataframe(
                new_article,
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# FUZZY
# ============================================================

elif page == "Approche floue":
    st.subheader("Extension floue")

    st.write(
        "L'approche floue permet d'assouplir les frontières entre A, B et C "
        "en utilisant des degrés d'appartenance."
    )

    p = np.linspace(0, 100, 501)

    mu_A = np.where(
        p <= 20, 1.0,
        np.where(p < 30, (30 - p) / 10, 0.0)
    )

    mu_B = np.select(
        [
            p <= 15,
            (p > 15) & (p < 25),
            (p >= 25) & (p <= 45),
            (p > 45) & (p < 55),
        ],
        [
            0.0,
            (p - 15) / 10,
            1.0,
            (55 - p) / 10,
        ],
        default=0.0,
    )

    mu_C = np.where(
        p <= 45, 0.0,
        np.where(p < 55, (p - 45) / 10, 1.0)
    )

    f = pd.DataFrame({
        "Rang cumulé (%)": np.tile(p, 3),
        "Degré": np.concatenate([mu_A, mu_B, mu_C]),
        "Classe": np.repeat(["A", "B", "C"], len(p)),
    })

    fuzzy_chart = (
        alt.Chart(f)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Rang cumulé (%):Q"),
            y=alt.Y("Degré:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Classe:N",
                scale=alt.Scale(
                    domain=["A", "B", "C"],
                    range=[
                        PASTELS["mint"],
                        PASTELS["peach"],
                        PASTELS["purple"],
                    ],
                ),
            ),
            tooltip=[
                "Classe:N",
                alt.Tooltip("Rang cumulé (%):Q", format=".1f"),
                alt.Tooltip("Degré:Q", format=".2f"),
            ],
        )
        .interactive()
        .properties(height=360)
    )
    st.altair_chart(fuzzy_chart, use_container_width=True)


# ============================================================
# EXPORT
# ============================================================

elif page == "Export":
    st.subheader("Exporter les résultats")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "Télécharger la base CRITIC–TOPSIS–ABC",
            abc_df.to_csv(index=False).encode("utf-8"),
            file_name="inventory_CRITIC_TOPSIS_ABC.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Télécharger les poids CRITIC",
            study["critic_details"].to_csv(index=False).encode("utf-8"),
            file_name="poids_CRITIC.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "Télécharger les performances ML",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name="performances_ML.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Télécharger la base prétraitée",
            processed_df.to_csv(index=False).encode("utf-8"),
            file_name="inventory_preprocessed.csv",
            mime="text/csv",
            use_container_width=True,
        )
