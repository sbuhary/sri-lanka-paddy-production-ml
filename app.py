import plotly.express as px
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data" / "Sri_Lanka_Paddy_Production.csv"

st.set_page_config(page_title="Sri Lanka Paddy Production", page_icon="🌾", layout="wide")
st.title("🌾 Sri Lanka Paddy Production")
st.caption("Applied Machine Learning · Random Forest and boosting model comparison")

if not (MODELS / "metadata.json").exists():
    st.error("Trained model files are missing. Run `python train_models.py` first.")
    st.stop()

@st.cache_resource
def load_models():
    meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    loaded = {name: joblib.load(MODELS / filename)
              for name, filename in meta["models"].items()}
    return meta, loaded

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_data
def load_results():
    return pd.read_csv(MODELS / "metrics.csv"), pd.read_csv(MODELS / "test_predictions.csv")

meta, models = load_models()
metrics, test_predictions = load_results()
data = load_data()

tab1, tab2, tab3 = st.tabs(["🔮 Predict production", "📊 Compare models", "📚 Dataset and method"])

with tab1:
    st.subheader("Predict paddy production")
    st.write("Choose a trained model and enter the district, season, year and sown extent.")
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Model", list(models), index=0)
        district = st.selectbox("District", meta["districts"])
        season = st.selectbox("Season", meta["seasons"])
    with col2:
        year = st.number_input("Year", min_value=1978, max_value=2100, value=2026, step=1)
        extent = st.number_input("Sown extent (hectares)", min_value=0.0,
                                 value=50000.0, step=1000.0, format="%.1f")
    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "District": district, "Season": season, "Year": int(year),
            "Sown_Extent_Hectares": float(extent)
        }])
        value = float(models[selected_model].predict(row)[0])
        st.metric("Predicted production", f"{value:,.2f} thousand metric tons")
        st.caption(f"Approximately {value * 1000:,.0f} metric tons. "
                   "This is a model estimate, not an official production forecast.")
        if year > meta["year_max"]:
            st.warning("This year is beyond the training data. Predictions for future years "
                       "are extrapolations and have not been validated with a chronological test.")

with tab2:
    st.subheader("Model performance on the same test split")
    st.dataframe(metrics.sort_values("RMSE").style.format({
        "R2": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"
    }), hide_index=True, use_container_width=True)
    st.caption("R²: higher is better. MAE and RMSE: lower is better. "
               "Error units are thousand metric tons.")
    st.bar_chart(metrics.set_index("Model")[["RMSE", "MAE"]])
    chart_model = st.selectbox("Actual vs predicted", metrics["Model"].tolist())
    plot_df = pd.DataFrame({
        "Actual": test_predictions["Actual"],
        "Predicted": test_predictions[chart_model]
    })

    plot_df["Absolute Error"] = (
        plot_df["Actual"] - plot_df["Predicted"]
    ).abs()

    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        hover_data=["Absolute Error"],
        labels={
            "Actual": "Actual production (thousand metric tons)",
            "Predicted": "Predicted production (thousand metric tons)"
        },
        height=450,
        opacity=0.6,
        title=f"{chart_model}: Actual vs Predicted"
    )

    axis_max = max(
        plot_df["Actual"].max(),
        plot_df["Predicted"].max()
    ) * 1.05

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=axis_max,
        y1=axis_max,
        line=dict(color="gray", dash="dash")
    )

    fig.update_layout(
        xaxis=dict(range=[0, axis_max]),
        yaxis=dict(range=[0, axis_max]),
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("This comparison uses a random 80/20 split. It does not establish "
            "performance on genuinely future seasons.")

with tab3:
    st.subheader("About the data")
    a, b, c = st.columns(3)
    a.metric("Original records", f"{meta['original_rows']:,}")
    b.metric("Records used", f"{meta['rows']:,}")
    c.metric("Years covered", f"{meta['year_min']}–{meta['year_max']}")
    st.markdown(
        "**Source:** [Department of Census and Statistics, Sri Lanka — "
        "Paddy Statistics](https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics)"
    )
    st.write("The source spreadsheets were consolidated "
             "by district, year and Maha/Yala season into one CSV.")
    st.write("**Inputs:** District, Season, Year, Sown Extent (hectares)")
    st.write("**Target:** Production (thousand metric tons)")
    st.write("Harvested extent and average yield were excluded from the model inputs "
             "because they are closely tied to the outcome and may not be known "
             "at the time a prediction is requested.")
    st.write(f"**Evaluation:** {meta['split']}. All five models use the same split.")
    st.write("**Models:** Random Forest, XGBoost, LightGBM, CatBoost, and an "
             "RF + ANN voting ensemble. The hybrid averages two regression predictions.")
    st.write("**Limitations:** This is an exploratory model comparison. Rainfall, "
             "irrigation, fertilizer use and other influential factors are not included.")
    with st.expander("Preview source data"):
        st.dataframe(data.head(25), use_container_width=True)
