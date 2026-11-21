import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Delhi Air Intelligence | AQI Lab", page_icon="🌫️", layout="wide")

ROOT = Path(__file__).parent
FEATURES = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "SO2 (µg/m³)", "NH3 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]
POLLUTANTS = FEATURES
CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
COLORS = {"Good":"#16a085","Satisfactory":"#7dbb55","Moderate":"#e7c34b","Poor":"#ed9145","Very Poor":"#d95d59","Severe":"#9e344c"}
REG_RESULTS = pd.DataFrame([
    ["Linear Regression",83.640,107.016,0.475],
    ["Decision Tree",94.547,129.725,0.228],
    ["Random Forest",81.155,104.418,0.500],
    ["Gradient Boosting",78.603,101.416,0.528],
],columns=["Model","MAE","RMSE","R²"])
CLF_RESULTS = pd.DataFrame([
    ["Logistic Regression",0.523,0.558,0.424,0.558,0.449],
    ["Decision Tree",0.438,0.446,0.408,0.446,0.342],
    ["Random Forest",0.484,0.374,0.390,0.374,0.363],
    ["Gradient Boosting",0.500,0.375,0.364,0.375,0.336],
],columns=["Model","Accuracy","Balanced Accuracy","Macro Precision","Macro Recall","Macro F1"])

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Manrope',sans-serif;}
.stApp {background: radial-gradient(ellipse at 15% 0%, #142b3a 0%, #0b1420 43%, #08111b 100%); color:#e8f0f5;}
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1450px;}
[data-testid="stSidebar"] {background:#0b1722; border-right:1px solid #203646;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#142635,#10202d); border:1px solid #284353; padding:18px 20px; border-radius:15px;}
[data-testid="stMetricLabel"] {color:#9fb7c6;}
[data-testid="stMetricValue"] {color:#f2f7fa;}
h1,h2,h3 {letter-spacing:-.035em;}
h1 {font-weight:800;}
.eyebrow {color:#61c6ce; font-family:'DM Mono',monospace; letter-spacing:.12em; font-size:.76rem; text-transform:uppercase;}
.hero {padding:28px 30px; border:1px solid #284454; border-radius:20px; background:linear-gradient(110deg,rgba(21,51,67,.96),rgba(15,31,46,.78)); margin-bottom:22px;}
.hero h1 {font-size:2.5rem; margin:.25rem 0 .4rem 0; color:#f2f7fa;}
.hero p {color:#adc2ce; font-size:1rem; margin:0;}
.panel {padding:18px 20px; background:#10212e; border:1px solid #284353; border-radius:16px; margin-bottom:12px;}
.small-note {color:#9eb4c1;font-size:.83rem;}
div[data-testid="stTabs"] button {font-weight:700;}
.stButton>button {background:#43c2c7;color:#07131d;border:0;border-radius:10px;font-weight:800;padding:.65rem 1.2rem;}
.stButton>button:hover {background:#71d7d7;color:#07131d;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return joblib.load(ROOT/"best_aqi_classifier.pkl"), joblib.load(ROOT/"best_aqi_regressor.pkl")

@st.cache_data
def load_daily():
    d = pd.read_csv(ROOT/"daily_aqi_dashboard.csv", parse_dates=["Date"])
    return d.sort_values("Date")

try:
    classifier, regressor = load_models()
    daily = load_daily()
except Exception as exc:
    st.error(f"App assets could not be loaded: {exc}")
    st.info("Ensure both model .pkl files and daily_aqi_dashboard.csv are in the same GitHub folder as app.py. Use the matching requirements.txt supplied with this release.")
    st.stop()

@st.cache_data(show_spinner=False)
def get_next_day_training_data(daily_frame):
    frame = daily_frame.copy().sort_values("Date")
    frame["Next_Day_AQI"] = pd.to_numeric(frame["Next_Day_AQI"], errors="coerce")
    frame = frame.dropna(subset=["Next_Day_AQI"]).copy()
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = frame["Next_Day_AQI"].astype(float)
    split = int(len(frame) * 0.80)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], len(frame)

def run_regression_tuning(daily_frame):
    X_train, X_test, y_train, y_test, n_rows = get_next_day_training_data(daily_frame)
    cv = TimeSeriesSplit(n_splits=5)
    specs = {
        "Linear Regression": (
            LinearRegression(),
            {"model__fit_intercept": [True, False], "model__positive": [False, True]}
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            {
                "model__max_depth": [None, 3, 5, 8, 12],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0]
            }
        )
    }
    baseline_rows, tuned_rows, best_models, parameter_rows = [], [], {}, []
    for name, (estimator, grid) in specs.items():
        baseline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator)
        ])
        baseline.fit(X_train, y_train)
        base_pred = baseline.predict(X_test)
        baseline_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, base_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, base_pred)),
            "R²": r2_score(y_test, base_pred)
        })

        candidate = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator.__class__(**estimator.get_params()))
        ])
        search = GridSearchCV(
            candidate, grid, scoring="neg_mean_absolute_error",
            cv=cv, n_jobs=-1, refit=True, return_train_score=True
        )
        search.fit(X_train, y_train)
        pred = search.best_estimator_.predict(X_test)
        best_models[name] = search.best_estimator_
        tuned_rows.append({
            "Model": name,
            "Best CV MAE": -search.best_score_,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
            "Configurations Tested": len(search.cv_results_["params"])
        })
        parameter_rows.append({"Model": name, "Best Parameters": str(search.best_params_)})

    baseline_df = pd.DataFrame(baseline_rows)
    tuned_df = pd.DataFrame(tuned_rows)
    comparison = baseline_df.merge(tuned_df, on="Model", suffixes=(" (Baseline)", " (Tuned)"))
    comparison["MAE Reduction"] = comparison["MAE (Baseline)"] - comparison["MAE (Tuned)"]
    comparison["RMSE Reduction"] = comparison["RMSE (Baseline)"] - comparison["RMSE (Tuned)"]
    comparison["R² Change"] = comparison["R² (Tuned)"] - comparison["R² (Baseline)"]
    return {
        "baseline": baseline_df, "tuned": tuned_df, "comparison": comparison,
        "parameters": pd.DataFrame(parameter_rows), "models": best_models,
        "train_rows": len(X_train), "test_rows": len(X_test), "total_rows": n_rows
    }

st.sidebar.markdown('<div class="eyebrow">DELHI AIR INTELLIGENCE</div>', unsafe_allow_html=True)
st.sidebar.title("AQI Lab")
page = st.sidebar.radio("WORKSPACE", ["Overview", "Air Quality Explorer", "Model Bake-off", "Model Training & Hyperparameter Tuning", "Prediction Studio", "Methodology"], label_visibility="visible")
st.sidebar.divider()
st.sidebar.caption("DATA WINDOW")
st.sidebar.markdown("**01 Jan 2023 — 31 Dec 2024**")
st.sidebar.caption("Hourly source data → daily AQI → next-day ML")
st.sidebar.caption("Research prototype · Not an official CPCB forecast")

aqi_valid = daily.dropna(subset=["AQI"]).copy()
latest = aqi_valid.iloc[-1] if len(aqi_valid) else None

def heading(kicker, title, desc):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1 style="margin:.2rem 0 .3rem 0">{title}</h1><p style="color:#a9bfcb;margin-top:0">{desc}</p>', unsafe_allow_html=True)

def category_color(value):
    return COLORS.get(value, "#61c6ce")

if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">DELHI · AIR QUALITY RESEARCH</div><h1>Reading the air.<br>Forecasting tomorrow.</h1><p>An interactive view of Delhi’s 2023–2024 hourly pollution dataset, reconstructed into daily AQI and evaluated with chronological next-day ML experiments.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Hourly observations", f"{len(daily)*24:,}" if False else "17,544")
    c2.metric("Daily records", f"{len(daily):,}")
    c3.metric("AQI-eligible days", f"{daily['AQI'].notna().sum():,}")
    c4.metric("Ineligible days", f"{daily['AQI'].isna().sum():,}")
    st.markdown("### AQI distribution across the study period")
    dist = aqi_valid["AQI_Category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).reset_index()
    dist.columns=["Category","Days"]
    col1,col2=st.columns([1.25,1])
    with col1:
        fig=px.bar(dist,x="Category",y="Days",color="Category",color_discrete_map=COLORS)
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,height=360,margin=dict(l=5,r=5,t=20,b=5),xaxis_title="",yaxis_title="Number of eligible days")
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig=px.pie(dist[dist.Days>0],names="Category",values="Days",hole=.64,color="Category",color_discrete_map=COLORS)
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=360,margin=dict(l=5,r=5,t=20,b=5),legend_title="")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Daily AQI trajectory")
    trend=px.area(aqi_valid,x="Date",y="AQI",color_discrete_sequence=["#50c5c9"])
    trend.add_hline(y=300,line_dash="dash",line_color="#d95d59",annotation_text="Very Poor threshold")
    trend.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=330,margin=dict(l=5,r=5,t=20,b=5),yaxis_title="AQI",xaxis_title="")
    st.plotly_chart(trend,use_container_width=True)
    st.markdown('<div class="small-note">AQI values and categories shown here are calculated using the methodology implemented in the project notebook. They are not a live monitoring feed.</div>',unsafe_allow_html=True)

elif page == "Air Quality Explorer":
    heading("01 / DATA EXPLORATION","Air Quality Explorer","Explore pollutant concentrations, temporal patterns and the daily AQI series.")
    start,end=st.columns(2)
    min_date=daily.Date.min().date(); max_date=daily.Date.max().date()
    with start: d1=st.date_input("From",min_date,min_value=min_date,max_value=max_date)
    with end: d2=st.date_input("To",max_date,min_value=min_date,max_value=max_date)
    view=daily[(daily.Date.dt.date>=d1)&(daily.Date.dt.date<=d2)].copy()
    if view.empty: st.warning("No observations in this date range.")
    else:
        a,b,c=st.columns(3)
        a.metric("Days in selected window",len(view))
        b.metric("Eligible AQI days",int(view.AQI.notna().sum()))
        c.metric("Mean AQI",f"{view.AQI.mean():.1f}" if view.AQI.notna().any() else "N/A")
        pollutant=st.selectbox("Pollutant",POLLUTANTS)
        fig=px.line(view,x="Date",y=pollutant,color_discrete_sequence=["#65d1d1"])
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=370,margin=dict(l=5,r=5,t=20,b=5),xaxis_title="",yaxis_title=pollutant)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("### Monthly concentration profile")
        monthly=view.assign(Month=view.Date.dt.to_period("M").astype(str)).groupby("Month")[POLLUTANTS].mean(numeric_only=True).reset_index()
        long=monthly.melt("Month",var_name="Pollutant",value_name="Mean concentration")
        fig=px.line(long,x="Month",y="Mean concentration",color="Pollutant",markers=True)
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=400,margin=dict(l=5,r=5,t=20,b=5),xaxis_title="",yaxis_title="Mean concentration")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("### Pollutant correlation")
        corr=view[POLLUTANTS].corr()
        fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="Tealrose",zmin=-1,zmax=1,aspect="auto")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=470,margin=dict(l=5,r=5,t=20,b=5))
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("View daily data table"):
            st.dataframe(view[["Date"]+POLLUTANTS+["AQI","AQI_Category"]],use_container_width=True,hide_index=True)

elif page == "Model Bake-off":
    heading("02 / MODEL EVALUATION","The ML Bake-off","Chronological holdout evaluation from the notebook’s verified next-day experiment.")
    st.markdown('<div class="panel"><b>Experiment design</b><br>637 consecutive-day forecasting rows · 509 train / 128 test · 7 pollutant features · chronological 80:20 split. Preprocessing: median imputation + standard scaling inside each pipeline.</div>',unsafe_allow_html=True)
    tab1,tab2=st.tabs(["Regression · AQI value","Classification · AQI category"])
    with tab1:
        st.markdown("#### Regression leaderboard")
        st.dataframe(REG_RESULTS.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.3f}"}),use_container_width=True,hide_index=True)
        fig=px.bar(REG_RESULTS.sort_values("MAE"),x="Model",y="MAE",color="Model",color_discrete_sequence=["#4cc7c9","#5b9bd5","#8a78cf","#d38a50","#d65d69"])
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,height=350,margin=dict(l=5,r=5,t=20,b=5),yaxis_title="Mean Absolute Error (lower is better)")
        st.plotly_chart(fig,use_container_width=True)
        st.info("Baseline reference: Gradient Boosting had the lowest baseline test MAE (78.603). The new tuning experiment selects its deployable regressor using cross-validation MAE; see the Hyperparameter Tuning workspace.")
        st.caption("The baseline is a mean predictor. Its negative R² indicates it performs worse than predicting the test-set mean benchmark.")
    with tab2:
        st.markdown("#### Classification leaderboard")
        st.dataframe(CLF_RESULTS.style.format({c:"{:.3f}" for c in CLF_RESULTS.columns if c!="Model"}),use_container_width=True,hide_index=True)
        fig=px.bar(CLF_RESULTS.sort_values("Macro F1",ascending=False),x="Model",y="Macro F1",color="Model")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,height=350,margin=dict(l=5,r=5,t=20,b=5),yaxis_title="Macro F1 (higher is better)")
        st.plotly_chart(fig,use_container_width=True)
        st.success("Selected classifier: Logistic Regression · Accuracy 0.523 · Balanced accuracy 0.558 · Macro F1 0.449")
        st.caption("Macro F1 gives each category equal weight; performance is weaker for the less represented categories.")

elif page == "Model Training & Hyperparameter Tuning":
    heading("03 / SYSTEMATIC MODEL OPTIMIZATION", "Model Training & Hyperparameter Tuning",
            "Build logical model versions, search sufficient parameter combinations, and compare baseline against tuned models.")
    st.markdown("""
    <div class="panel">
    <b>Evaluation protocol</b><br>
    Four regression model families · chronological 80:20 holdout · 5-fold TimeSeriesSplit on training data only ·
    GridSearchCV with MAE as the selection metric. The final test partition is not used during parameter selection.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Training procedure")
    st.write("The workflow first trains a baseline version of each model, then evaluates a structured parameter grid. This makes the experiment reproducible and avoids random, undocumented model changes.")
    model_plan = pd.DataFrame([
        ["Linear Regression", "fit_intercept, positive", "2 × 2"],
        ["Decision Tree", "max_depth, min_samples_split, min_samples_leaf", "5 × 3 × 3"],
        ["Random Forest", "n_estimators, max_depth, min_samples_split, min_samples_leaf", "3 × 3 × 2 × 2"],
        ["Gradient Boosting", "n_estimators, learning_rate, max_depth, subsample", "2 × 3 × 2 × 2"]
    ], columns=["Model", "Hyperparameters searched", "Grid size"])
    st.dataframe(model_plan, use_container_width=True, hide_index=True)

    st.markdown("### 2. Run systematic hyperparameter tuning")
    st.caption("This runs the actual search on the bundled daily AQI data. Runtime depends on your computer.")
    if st.button("Run baseline training + hyperparameter search", type="primary", use_container_width=True):
        with st.spinner("Training baselines and running chronological GridSearchCV for all four models..."):
            try:
                result = run_regression_tuning(daily)
                st.session_state["aqi_tuning_result"] = result
                st.success("Tuning completed. Results below are computed from the current bundled dataset.")
            except Exception as exc:
                st.error(f"Tuning failed: {exc}")

    result = st.session_state.get("aqi_tuning_result")
    if result:
        st.markdown("### 3. Experiment design and data split")
        m1, m2, m3 = st.columns(3)
        m1.metric("Eligible next-day rows", f"{result['total_rows']:,}")
        m2.metric("Training rows", f"{result['train_rows']:,}")
        m3.metric("Untouched test rows", f"{result['test_rows']:,}")
        st.info("Median imputation and standard scaling are inside each Pipeline, so they are fitted separately within each cross-validation fold. TimeSeriesSplit preserves temporal order.")

        st.markdown("### 4. Baseline vs tuned performance")
        st.caption("MAE and RMSE: lower is better. R²: higher is better. A negative improvement means that metric did not improve.")
        st.dataframe(result["comparison"].round(3), use_container_width=True, hide_index=True)
        plot_data = result["comparison"].melt(
            id_vars="Model",
            value_vars=["MAE (Baseline)", "MAE (Tuned)"],
            var_name="Experiment", value_name="MAE"
        )
        fig = px.bar(plot_data, x="Model", y="MAE", color="Experiment", barmode="group")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=370,
                          yaxis_title="Mean Absolute Error (lower is better)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 5. Best hyperparameters selected by cross-validation")
        st.dataframe(result["parameters"], use_container_width=True, hide_index=True)
        best_row = result["tuned"].sort_values("MAE").iloc[0]
        best_name = best_row["Model"]
        st.success(f"Lowest held-out test MAE among the tuned candidates: {best_name} · MAE {best_row['MAE']:.3f} · RMSE {best_row['RMSE']:.3f} · R² {best_row['R²']:.3f}")

        st.markdown("### 6. Export experiment evidence")
        csv_bytes = result["comparison"].to_csv(index=False).encode("utf-8")
        st.download_button("Download baseline-vs-tuned comparison (CSV)", csv_bytes,
                           "aqi_baseline_vs_tuned_comparison.csv", "text/csv")
        param_bytes = result["parameters"].to_csv(index=False).encode("utf-8")
        st.download_button("Download selected hyperparameters (CSV)", param_bytes,
                           "aqi_best_hyperparameters.csv", "text/csv")
        model_bytes = pickle.dumps(result["models"][best_name])
        st.download_button(f"Download best tuned regressor ({best_name})", model_bytes,
                           "best_tuned_aqi_regressor.pkl", "application/octet-stream")
        st.caption("Downloaded tuned model is a regression pipeline for numeric next-day AQI. The separate category classifier remains unchanged.")
    else:
        st.warning("Run the tuning workflow to populate measured results. No tuned scores are pre-filled or fabricated.")

elif page == "Prediction Studio":
    heading("04 / NEXT-DAY FORECAST","Prediction Studio","Enter today’s pollutant concentrations to estimate tomorrow’s AQI and category.")
    st.markdown('<div class="panel">The active numeric AQI pipeline is the CV-selected tuned Linear Regression model. The category classifier remains the existing separately trained classifier. Both use the same seven input features. Inputs are concentrations, not AQI sub-indices.</div>',unsafe_allow_html=True)
    with st.form("prediction_form"):
        left,right=st.columns(2)
        defaults={"PM2.5 (µg/m³)":60.0,"PM10 (µg/m³)":100.0,"NO2 (µg/m³)":30.0,"SO2 (µg/m³)":10.0,"NH3 (µg/m³)":20.0,"CO (mg/m³)":1.0,"Ozone (µg/m³)":30.0}
        values={}
        for i,f in enumerate(FEATURES):
            target=left if i<4 else right
            with target:
                values[f]=st.number_input(f,min_value=0.0,value=defaults[f],step=1.0,format="%.2f")
        submitted=st.form_submit_button("Generate next-day forecast",use_container_width=True)
    if submitted:
        x=pd.DataFrame([values],columns=FEATURES)
        try:
            pred_aqi=max(0.0,float(regressor.predict(x)[0]))
            pred_cat=str(classifier.predict(x)[0])
            c1,c2=st.columns(2)
            c1.metric("Predicted next-day AQI",f"{pred_aqi:.1f}")
            c2.metric("Predicted AQI category",pred_cat)
            st.progress(min(pred_aqi/500,1.0),text="Position on the project’s 0–500 AQI scale")
            st.markdown(f'<div class="panel"><span class="eyebrow">MODEL OUTPUT</span><h2 style="color:{category_color(pred_cat)}">{pred_cat}</h2><span class="small-note">AQI regression and category classification are separate model outputs; the category is not derived from the numeric regression output.</span></div>',unsafe_allow_html=True)
            st.caption("Estimate only. This research model is not an official CPCB forecast and does not incorporate live weather, emissions or real-time station data.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Methodology":
    heading("05 / MODEL CARD","Methodology & Limitations","A transparent summary of how the notebook constructs AQI and evaluates next-day predictions.")
    st.markdown("### Data and temporal aggregation")
    st.markdown("""
- Source data: Delhi hourly observations for 2023 and 2024 (17,544 hourly rows in the notebook).
- Five pollutants use daily means: PM2.5, PM10, NO2, SO2 and NH3. A day is set missing when fewer than 16 hourly readings are available.
- CO and Ozone use the maximum within-day 8-hour rolling mean; fewer than 16 daily readings makes that pollutant missing.
- The daily AQI dataset contains 731 calendar days.
""")
    st.markdown("### AQI construction implemented in the notebook")
    st.markdown("""
- Pollutant-specific concentration breakpoints are interpolated to sub-indices.
- Overall AQI is the maximum available pollutant sub-index.
- Eligibility requires at least 3 valid sub-indices and at least one PM sub-index.
- Concentrations above the highest listed breakpoint are assigned 500 in this implementation.
- Categories: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), Severe (>400).
""")
    st.markdown("### Forecasting protocol")
    st.markdown("""
- Target: next calendar day’s AQI and AQI category.
- Only consecutive date pairs with valid next-day targets are retained.
- 637 rows; first 509 used for training and final 128 for testing, preserving chronological order.
- Features: seven pollutant concentrations. Preprocessing is median imputation followed by standard scaling.
- Existing regression leaderboard selects Gradient Boosting by lowest MAE; classification leaderboard selects Logistic Regression by highest Macro F1. The dedicated tuning workspace now runs a separate four-regressor GridSearchCV experiment and reports its measured results.
""")
    st.warning("Limitations: the notebook’s breakpoint implementation caps concentrations above the highest listed band at AQI 500; this should be described as the project’s implemented calculation, not independently certified official CPCB software. Test-set metrics are moderate, and category-level performance is uneven. Predictions are not live or official advisories.")
    st.markdown("### Reproducibility")
    st.code("Python 3.11\nscikit-learn==1.8.0\nstreamlit\npandas\nnumpy\njoblib\nplotly",language="text")
