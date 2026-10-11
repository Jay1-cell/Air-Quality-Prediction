import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================
# 1. Page configuration
# ======================================================
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# 2. Styling
# ======================================================
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Header */
    .app-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f3d5e 0%, #1b6b8f 55%, #2a9d8f 100%);
        color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .app-hero h1 { margin: 0; font-size: 2rem; color: #ffffff; }
    .app-hero p  { margin: .4rem 0 0 0; font-size: 1.02rem; opacity: .92; }

    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        color: #ffffff;
        margin-bottom: .8rem;
    }
    .result-card .label { font-size: .95rem; opacity: .9; margin: 0; }
    .result-card .value { font-size: 2.3rem; font-weight: 700; margin: .1rem 0 .3rem 0; line-height: 1.15; }
    .result-card .advice { font-size: 1rem; margin: 0; opacity: .95; }

    /* AQI scale strip */
    .scale { display: flex; gap: 4px; margin: .4rem 0 .2rem 0; }
    .scale .seg {
        flex: 1; text-align: center; padding: .55rem .2rem;
        font-size: .78rem; font-weight: 600; color: #fff;
        border-radius: 8px; opacity: .35;
    }
    .scale .seg.active {
        opacity: 1;
        outline: 3px solid rgba(128,128,128,.55);
        outline-offset: 2px;
    }

    /* Small stat boxes */
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.2);
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# 3. Category styling helpers
# ======================================================
# (keyword, severity rank, colour, emoji, advice) – checked in order,
# so more specific phrases come before shorter ones.
CATEGORY_RULES = [
    ("hazardous", 5, "#7e0023", "☠️", "Emergency conditions. Everyone should avoid outdoor exposure."),
    ("severe", 5, "#7e0023", "☠️", "Serious risk for everyone. Stay indoors and use air purifiers."),
    ("very unhealthy", 4, "#8f3f97", "🟣", "Health alert. Avoid outdoor activity."),
    ("very poor", 4, "#8f3f97", "🟣", "Prolonged exposure can cause respiratory illness. Limit time outdoors."),
    ("unhealthy for sensitive", 2, "#ff7e00", "🟠", "Children, older adults and people with lung or heart disease should limit outdoor effort."),
    ("unhealthy", 3, "#e03131", "🔴", "Everyone may begin to feel effects. Reduce prolonged outdoor activity."),
    ("poor", 3, "#e03131", "🔴", "Breathing discomfort is likely on longer exposure. Limit outdoor activity."),
    ("moderate", 2, "#e0a800", "🟡", "Acceptable for most people. Unusually sensitive people should take it easy outdoors."),
    ("satisfactory", 1, "#7cb518", "🟢", "Minor discomfort possible for people with breathing issues."),
    ("good", 0, "#2f9e44", "😊", "Air quality is healthy. Enjoy outdoor activities."),
]
DEFAULT_STYLE = (2, "#6c757d", "🌫️", "No guidance available for this category.")


def category_style(label: str):
    """Return (rank, colour, emoji, advice) for an AQI category label."""
    text = str(label).lower()
    for keyword, rank, colour, emoji, advice in CATEGORY_RULES:
        if keyword in text:
            return rank, colour, emoji, advice
    return DEFAULT_STYLE


def pretty(name: str) -> str:
    return str(name).replace("_", " ").strip()


# ======================================================
# 4. Load saved files
# ======================================================
@st.cache_resource
def load_files():
    model = joblib.load("best_aqi_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, label_encoder, feature_columns


try:
    model, scaler, label_encoder, feature_columns = load_files()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(
        "Place best_aqi_model.pkl, scaler.pkl, label_encoder.pkl and "
        "feature_columns.pkl in the same folder as app.py, and run the app "
        "from the Python environment that has the required libraries installed."
    )
    st.stop()

all_categories = list(label_encoder.classes_)
ordered_categories = sorted(all_categories, key=lambda c: category_style(c)[0])

# ======================================================
# 5. Session state (inputs + history)
# ======================================================
for i in range(len(feature_columns)):
    st.session_state.setdefault(f"input_{i}", 0.0)
st.session_state.setdefault("history", [])


def fill_all(value: float):
    for i in range(len(feature_columns)):
        st.session_state[f"input_{i}"] = float(value)


# ======================================================
# 6. Sidebar
# ======================================================
with st.sidebar:
    st.header("⚙️ Controls")

    st.markdown("**Quick fill**")
    st.caption("Set every pollutant to a typical AQI level, then adjust.")
    c1, c2 = st.columns(2)
    c1.button("Clean air (25)", on_click=fill_all, args=(25,), use_container_width=True)
    c2.button("Moderate (75)", on_click=fill_all, args=(75,), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.button("Poor (175)", on_click=fill_all, args=(175,), use_container_width=True)
    c4.button("Severe (350)", on_click=fill_all, args=(350,), use_container_width=True)
    st.button("Reset to zero", on_click=fill_all, args=(0,), use_container_width=True)

    st.divider()
    st.markdown("**Model**")
    st.write(type(model).__name__)
    st.caption(f"{len(feature_columns)} input features • {len(all_categories)} categories")

    st.markdown("**AQI categories**")
    for cat in ordered_categories:
        _, colour, emoji, _ = category_style(cat)
        st.markdown(
            f"<span style='color:{colour};font-weight:600'>●</span> {cat}",
            unsafe_allow_html=True,
        )

# ======================================================
# 7. Header
# ======================================================
st.markdown(
    """
    <div class="app-hero">
        <h1>🌍 Air Quality Prediction System</h1>
        <p>Enter pollutant AQI values and get the predicted AQI category,
        with the model's confidence in every category.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_history, tab_about = st.tabs(["🔮 Predict", "🕘 History", "ℹ️ About"])

# ======================================================
# 8. Predict tab
# ======================================================
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    # ---------- Inputs ----------
    with left:
        st.subheader("Pollutant AQI values")
        with st.form("aqi_prediction_form"):
            input_values = {}
            cols = st.columns(2)
            for i, feature in enumerate(feature_columns):
                with cols[i % 2]:
                    input_values[feature] = st.number_input(
                        label=pretty(feature),
                        min_value=0.0,
                        max_value=1000.0,
                        step=1.0,
                        key=f"input_{i}",
                        help=f"AQI sub-index for {pretty(feature)} (0–1000).",
                    )
            submitted = st.form_submit_button(
                "Predict AQI category", type="primary", use_container_width=True
            )

    # ---------- Prediction ----------
    if submitted:
        input_df = pd.DataFrame(
            [[input_values[col] for col in feature_columns]], columns=feature_columns
        )
        try:
            input_scaled = scaler.transform(input_df)
            pred_encoded = model.predict(input_scaled)
            prediction = label_encoder.inverse_transform(
                np.asarray(pred_encoded, dtype=int)
            )[0]

            probs_df = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_scaled)[0]
                probs_df = pd.DataFrame(
                    {
                        "AQI Category": label_encoder.inverse_transform(
                            np.asarray(model.classes_, dtype=int)
                        ),
                        "Probability": probs,
                    }
                )

            st.session_state["last_result"] = {
                "prediction": prediction,
                "inputs": dict(input_values),
                "probs": probs_df,
            }
            st.session_state["history"].append(
                {"Predicted category": prediction, **{pretty(k): v for k, v in input_values.items()}}
            )
        except Exception as e:
            st.session_state.pop("last_result", None)
            st.error(f"Prediction failed: {e}")

    # ---------- Result ----------
    with right:
        st.subheader("Prediction result")
        result = st.session_state.get("last_result")

        if result is None:
            st.info(
                "Fill in the pollutant values and choose **Predict AQI category**. "
                "You can also use *Quick fill* in the sidebar to start from a typical level."
            )
        else:
            prediction = result["prediction"]
            rank, colour, emoji, advice = category_style(prediction)

            st.markdown(
                f"""
                <div class="result-card" style="background:{colour}">
                    <p class="label">Predicted AQI category</p>
                    <p class="value">{emoji} {prediction}</p>
                    <p class="advice">{advice}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Scale strip showing where the result falls
            segs = "".join(
                f'<div class="seg{" active" if c == prediction else ""}" '
                f'style="background:{category_style(c)[1]}">{c}</div>'
                for c in ordered_categories
            )
            st.markdown(f'<div class="scale">{segs}</div>', unsafe_allow_html=True)

            # Summary metrics
            probs_df = result["probs"]
            m1, m2 = st.columns(2)
            if probs_df is not None:
                top_conf = probs_df["Probability"].max()
                m1.metric("Model confidence", f"{top_conf:.1%}")
            else:
                m1.metric("Model confidence", "n/a")

            inputs_series = pd.Series(result["inputs"])
            if inputs_series.max() > 0:
                m2.metric("Highest pollutant", pretty(inputs_series.idxmax()), f"{inputs_series.max():.0f}")
            else:
                m2.metric("Highest pollutant", "—")

    # ---------- Charts ----------
    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        chart_a, chart_b = st.columns(2, gap="large")

        with chart_a:
            st.markdown("**Category probabilities**")
            probs_df = result["probs"]
            if probs_df is not None:
                probs_df = probs_df.sort_values("Probability", ascending=False)
                domain = list(probs_df["AQI Category"])
                colours = [category_style(c)[1] for c in domain]
                prob_chart = (
                    alt.Chart(probs_df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("AQI Category:N", sort=domain, title=None),
                        color=alt.Color(
                            "AQI Category:N",
                            scale=alt.Scale(domain=domain, range=colours),
                            legend=None,
                        ),
                        tooltip=["AQI Category", alt.Tooltip("Probability:Q", format=".2%")],
                    )
                    .properties(height=max(180, 38 * len(domain)))
                )
                st.altair_chart(prob_chart, use_container_width=True)
                st.caption("Probabilities are model estimates, not guaranteed real-world likelihoods.")
            else:
                st.info("This model does not provide class probabilities.")

        with chart_b:
            st.markdown("**Your pollutant inputs**")
            inp_df = pd.DataFrame(
                {"Pollutant": [pretty(k) for k in result["inputs"]], "AQI value": list(result["inputs"].values())}
            )
            inp_chart = (
                alt.Chart(inp_df)
                .mark_bar(cornerRadiusEnd=4, color="#1b6b8f")
                .encode(
                    x=alt.X("AQI value:Q"),
                    y=alt.Y("Pollutant:N", sort="-x", title=None),
                    tooltip=["Pollutant", "AQI value"],
                )
                .properties(height=max(180, 38 * len(inp_df)))
            )
            st.altair_chart(inp_chart, use_container_width=True)

        # Download
        report = pd.DataFrame(
            [{"Predicted category": result["prediction"], **{pretty(k): v for k, v in result["inputs"].items()}}]
        )
        st.download_button(
            "⬇️ Download this result (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="aqi_prediction.csv",
            mime="text/csv",
        )

# ======================================================
# 9. History tab
# ======================================================
with tab_history:
    st.subheader("Prediction history")
    history = st.session_state["history"]
    if not history:
        st.info("No predictions yet. Run one from the Predict tab and it will appear here.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df.index = np.arange(1, len(hist_df) + 1)
        hist_df.index.name = "#"

        counts = hist_df["Predicted category"].value_counts().reset_index()
        counts.columns = ["AQI Category", "Count"]

        h1, h2 = st.columns([1.6, 1], gap="large")
        with h1:
            st.dataframe(hist_df, use_container_width=True)
        with h2:
            domain = list(counts["AQI Category"])
            hist_chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Count:Q", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("AQI Category:N", sort="-x", title=None),
                    color=alt.Color(
                        "AQI Category:N",
                        scale=alt.Scale(domain=domain, range=[category_style(c)[1] for c in domain]),
                        legend=None,
                    ),
                )
                .properties(height=max(150, 38 * len(domain)))
            )
            st.altair_chart(hist_chart, use_container_width=True)

        b1, b2 = st.columns(2)
        b1.download_button(
            "⬇️ Download history (CSV)",
            data=hist_df.to_csv().encode("utf-8"),
            file_name="aqi_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if b2.button("🗑️ Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.session_state.pop("last_result", None)
            st.rerun()

# ======================================================
# 10. About tab
# ======================================================
with tab_about:
    st.subheader("About this project")
    st.write(
        "This app predicts an air quality category from pollutant AQI values using a "
        "machine learning classifier trained for a Probability and Statistics project."
    )
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**How it works**")
        st.markdown(
            "1. Your inputs are arranged in the same column order used in training.\n"
            "2. The saved scaler applies the same scaling as training.\n"
            "3. The model predicts a category and, when supported, class probabilities.\n"
            "4. The label encoder converts the result back to a readable category."
        )
    with a2:
        st.markdown("**Models compared in the project**")
        st.markdown(
            "- Gaussian Naive Bayes\n"
            "- Logistic Regression\n"
            "- Decision Tree\n"
            "- Random Forest"
        )
        st.caption(f"Model currently loaded: {type(model).__name__}")

    st.markdown("**Input features**")
    st.write(", ".join(pretty(f) for f in feature_columns))

# ======================================================
# 11. Footer
# ======================================================
st.divider()
st.caption(
    "Academic project | Probability and Statistics | "
    "GaussianNB, Logistic Regression, Decision Tree and Random Forest"
)