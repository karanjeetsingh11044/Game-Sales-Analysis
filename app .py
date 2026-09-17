"""
🎮 Game Sales Analytics & Prediction Dashboard
Streamlit app for the AIML Internship Project
Run with: streamlit run app.py   (or: python -m streamlit run app.py)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Game Sales Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — ATTRACTIVE UI
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 15% 0%, #2a2560 0%, #14112b 45%, #0b0a1a 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(200deg, #1c1740 0%, #0d0b23 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] * {
        color: #e6e4ff;
    }

    /* ---------- HERO BANNER ---------- */
    .hero-banner {
        background: linear-gradient(120deg, rgba(248,87,166,0.18), rgba(255,88,88,0.14) 45%, rgba(255,212,82,0.14));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 26px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        animation: fadeIn 0.6s ease;
    }

    .hero-eyebrow {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #ffd452;
        background: rgba(255, 212, 82, 0.12);
        border: 1px solid rgba(255, 212, 82, 0.35);
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(90deg, #ffffff, #ffd452 60%, #ff5858);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }

    .hero-subtitle {
        color: #c9c9e8;
        font-size: 1.02rem;
        max-width: 720px;
        margin: 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 20px 18px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        animation: fadeIn 0.6s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 212, 82, 0.55);
    }
    .metric-icon { font-size: 1.6rem; margin-bottom: 4px; }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffd452;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #b9b8dd;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 2px;
    }

    /* ---------- SECTION TITLES ---------- */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 5px solid #ff5858;
        padding-left: 14px;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
    }

    .styled-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 10px;
    }

    /* ---------- BADGES (About page) ---------- */
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.16);
        color: #e6e4ff;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 4px 6px 4px 0;
    }

    /* ---------- BUTTONS ---------- */
    .stButton>button {
        background: linear-gradient(90deg, #f857a6, #ff5858);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.6rem;
        font-weight: 600;
        transition: 0.2s ease;
        box-shadow: 0 4px 14px rgba(255, 88, 88, 0.25);
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(255, 88, 88, 0.45);
    }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #c9c9e8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffd452 !important;
    }

    div[data-testid="stMetricValue"] { color: #ffd452; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
DATA_PATH = r"C:\Users\karan\OneDrive\Desktop\game sales programming\game_sales_dataset.csv"


def hero(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="hero-eyebrow">{eyebrow}</span>
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = df.drop_duplicates()
    df["Developer"] = df["Developer"].fillna("Unknown")
    df["Critic_Score"] = df["Critic_Score"].fillna(df["Critic_Score"].median())
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
    df["User_Score"] = df["User_Score"].fillna(df["User_Score"].median())
    return df


@st.cache_resource
def train_models(df):
    model_df = df.copy()
    le_platform = LabelEncoder()
    le_publisher = LabelEncoder()
    model_df["Platform_enc"] = le_platform.fit_transform(model_df["Platform"])
    model_df["Publisher_enc"] = le_publisher.fit_transform(model_df["Publisher"])

    features = ["Platform_enc", "Publisher_enc", "Critic_Score", "User_Score", "Year"]
    target = "Total_Shipped"

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        }

    results = {
        "Linear Regression": metrics(y_test, lr_preds),
        "Random Forest": metrics(y_test, rf_preds),
    }

    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(
        ascending=False
    )

    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "le_platform": le_platform,
        "le_publisher": le_publisher,
        "features": features,
        "results": results,
        "importances": importances,
    }


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Dataset not found at:\n\n`{DATA_PATH}`\n\n"
        "Make sure `game_sales_dataset.csv` is in that exact folder, "
        "or update `DATA_PATH` near the top of app.py."
    )
    st.stop()

bundle = train_models(df)

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 10px 0 4px 0;">
        <div style="font-size: 2.4rem;">🎮</div>
        <div style="font-weight:700; font-size:1.15rem; color:#ffffff;">Game Sales AI</div>
        <div style="font-size:0.78rem; color:#a9a7d1;">Analytics & Prediction Suite</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Explore Data", "🤖 Model & Predict", "ℹ️ About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.markdown("#### 🔍 Filters")
year_range = st.sidebar.slider(
    "Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max())),
)
platforms = st.sidebar.multiselect(
    "Platform (leave empty = all)", sorted(df["Platform"].unique().tolist())
)

filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
if platforms:
    filtered = filtered[filtered["Platform"].isin(platforms)]

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.sidebar.caption("Built with ❤️ using Streamlit · AIML Internship Project")

# ----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ----------------------------------------------------------------------------
if page == "🏠 Overview":
    hero(
        "Live Dashboard",
        "Game Sales Analytics Dashboard",
        "Explore 19,600+ video games — sales trends, platforms, publishers, and an AI model that predicts shipped units.",
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🎮", f"{len(filtered):,}", "Games"),
        ("🏢", f"{filtered['Publisher'].nunique():,}", "Publishers"),
        ("🕹️", f"{filtered['Platform'].nunique():,}", "Platforms"),
        ("📦", f"{filtered['Total_Shipped'].sum():,.0f}M", "Units Shipped"),
    ]
    for col, (icon, value, label) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"""<div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    section("🏆 Top 10 Best-Selling Games")
    top_games = filtered.sort_values("Total_Shipped", ascending=False).head(10)
    fig = px.bar(
        top_games,
        x="Total_Shipped",
        y="Name",
        color="Total_Shipped",
        orientation="h",
        color_continuous_scale="sunset",
        template=PLOTLY_TEMPLATE,
        labels={"Total_Shipped": "Units Shipped (Millions)", "Name": ""},
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Preview raw data"):
        st.dataframe(filtered.head(50), use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 2: EXPLORE DATA (EDA)
# ----------------------------------------------------------------------------
elif page == "📊 Explore Data":
    hero(
        "Interactive EDA",
        "Explore the Data",
        "Hover, zoom, and filter using the sidebar to dig into sales patterns.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦 Publishers", "🕹️ Platforms", "📈 Trend Over Years", "🧮 Correlation"]
    )

    with tab1:
        top_publishers = (
            filtered.groupby("Publisher")["Total_Shipped"].sum().sort_values(ascending=False).head(10)
        )
        fig = px.bar(
            top_publishers,
            x=top_publishers.values,
            y=top_publishers.index,
            orientation="h",
            color=top_publishers.values,
            color_continuous_scale="viridis",
            template=PLOTLY_TEMPLATE,
            labels={"x": "Total Units Shipped (M)", "y": ""},
            title="Top 10 Publishers by Total Games Shipped",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_platforms = filtered["Platform"].value_counts().head(10)
        fig = px.pie(
            names=top_platforms.index,
            values=top_platforms.values,
            hole=0.5,
            template=PLOTLY_TEMPLATE,
            title="Top 10 Platforms by Number of Games",
            color_discrete_sequence=px.colors.sequential.Plasma,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        yearly = filtered.groupby("Year").size().reset_index(name="Games Released")
        fig = px.area(
            yearly,
            x="Year",
            y="Games Released",
            template=PLOTLY_TEMPLATE,
            title="Game Releases Over the Years",
            color_discrete_sequence=["#ff5858"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            filtered,
            x="Critic_Score",
            y="Total_Shipped",
            color="Total_Shipped",
            color_continuous_scale="turbo",
            template=PLOTLY_TEMPLATE,
            opacity=0.55,
            title="Critic Score vs Total Units Shipped",
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        numeric_cols = ["Critic_Score", "User_Score", "Total_Shipped", "Year"]
        corr = filtered[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template=PLOTLY_TEMPLATE,
            title="Correlation Heatmap",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3: MODEL & PREDICT
# ----------------------------------------------------------------------------
elif page == "🤖 Model & Predict":
    hero(
        "Machine Learning",
        "Model Performance & Live Prediction",
        "Compare Linear Regression vs Random Forest, then predict sales for a hypothetical game.",
    )

    section("📐 Model Comparison")
    results_df = pd.DataFrame(bundle["results"]).T.reset_index().rename(columns={"index": "Model"})

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(
            results_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
            use_container_width=True,
        )
    with c2:
        fig = px.bar(
            results_df,
            x="Model",
            y="R2",
            color="Model",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#f857a6", "#ffd452"],
            title="R² Score Comparison",
            text_auto=".3f",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    section("🌟 Feature Importance (Random Forest)")
    imp_df = bundle["importances"].reset_index()
    imp_df.columns = ["Feature", "Importance"]
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="plasma",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    section("🔮 Try a Live Prediction")
    st.markdown('<div class="styled-card">', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)
    with colA:
        platform_choice = st.selectbox("Platform", sorted(df["Platform"].unique().tolist()))
        year_choice = st.slider("Release Year", 1980, 2026, 2020)
    with colB:
        publisher_choice = st.selectbox("Publisher", sorted(df["Publisher"].unique().tolist()))
        critic_score = st.slider("Expected Critic Score", 0.0, 10.0, 7.5, 0.1)
    with colC:
        user_score = st.slider("Expected User Score", 0.0, 10.0, 7.5, 0.1)
        model_choice = st.selectbox("Model to use", ["Random Forest", "Linear Regression"])

    predict_clicked = st.button("🚀 Predict Total Units Shipped")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        le_platform = bundle["le_platform"]
        le_publisher = bundle["le_publisher"]

        platform_enc = le_platform.transform([platform_choice])[0]
        publisher_enc = le_publisher.transform([publisher_choice])[0]

        input_row = pd.DataFrame(
            [[platform_enc, publisher_enc, critic_score, user_score, year_choice]],
            columns=bundle["features"],
        )

        model = bundle["rf_model"] if model_choice == "Random Forest" else bundle["lr_model"]
        prediction = model.predict(input_row)[0]
        prediction = max(prediction, 0)

        st.markdown(
            f"""
            <div style="text-align:center; padding: 26px; margin-top: 14px;
                        background: linear-gradient(90deg, rgba(248,87,166,0.18), rgba(255,212,82,0.18));
                        border: 1px solid rgba(255,255,255,0.18); border-radius: 18px;
                        animation: fadeIn 0.5s ease;">
                <div style="font-size: 0.95rem; color:#c9c9e8; letter-spacing:0.05em; text-transform:uppercase;">Predicted Units Shipped</div>
                <div style="font-size: 2.8rem; font-weight:800; color:#ffd452;">{prediction:.2f} Million</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# PAGE 4: ABOUT
# ----------------------------------------------------------------------------
elif page == "ℹ️ About":
    hero(
        "Project Info",
        "About This Project",
        "An AIML internship project covering the full data science workflow — from raw data to a deployed, interactive ML app.",
    )

    section("🧰 Tech Stack")
    for badge in ["Python", "Pandas", "NumPy", "Scikit-learn", "Streamlit", "Plotly"]:
        st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)

    section("📋 What This App Does")
    st.markdown(
        """
        <div class="styled-card">
        <ul style="color:#e6e4ff; line-height:1.9;">
            <li><b>Overview</b> — key stats and top-selling games at a glance</li>
            <li><b>Explore Data</b> — interactive charts on publishers, platforms, yearly trends, and correlations</li>
            <li><b>Model & Predict</b> — Linear Regression vs Random Forest comparison, feature importance, and a live prediction tool</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("📊 Dataset")
    st.markdown(
        """
        <div class="styled-card">
        19,600+ video games with columns: <code>Rank, Name, Platform, Publisher, Developer,
        Critic_Score, User_Score, Total_Shipped, Year</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
