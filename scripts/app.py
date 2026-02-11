# app.py
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import altair as alt

# Meridian
from meridian.model import model as meridian_model
from meridian.analysis import visualizer
from schema.serde import meridian_serde  # recommended serialization (binpb/txtpb)

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Meridian MMM Explorer", layout="wide")

st.title("Meridian MMM Explorer (pre-trained models)")
st.caption("Select granularity + prior quality → load model → explore fit, diagnostics, ROI, contributions, and curves.")

# ----------------------------
# User choices
# ----------------------------
GRANULARITIES = ["df_l4", "df_l5"]
PRIOR_QUALITIES = ["optimistic", "neutral", "pessimistic"]

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    granularity = st.selectbox("Granularity", GRANULARITIES, index=0)
with colB:
    prior_quality = st.selectbox("Prior quality", PRIOR_QUALITIES, index=1)
with colC:
    models_dir = st.text_input("Models folder", value="models")

model_key = f"{granularity}_{prior_quality}"
default_model_path = Path(models_dir) / f"{model_key}.binpb"

st.write(f"**Selected model:** `{model_key}`")
st.write(f"**Expected file:** `{default_model_path}`")

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_meridian_model(path_str: str):
    """
    Loads a Meridian model object from a protobuf file (recommended approach).
    Supported extensions: .binpb, .txtpb, .textproto
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")

    # Preferred: Meridian serde (protobuf)
    if path.suffix.lower() in [".binpb", ".txtpb", ".textproto"]:
        return meridian_serde.load_meridian(str(path))

    # Fallback: legacy pickle (deprecated in docs)
    if path.suffix.lower() == ".pkl":
        return meridian_model.load_mmm(str(path))

    raise ValueError(f"Unsupported model format: {path.suffix}. Use .binpb/.txtpb or .pkl")


# Let user override model path if needed
model_path = st.text_input("Model file path", value=str(default_model_path))

try:
    mmm = load_meridian_model(model_path)
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ----------------------------
# Visualizers (these are Meridian's "plot methods")
# ----------------------------
@st.cache_resource(show_spinner=False)
def make_visualizers(_mmm):
    return {
        "media_summary": visualizer.MediaSummary(_mmm),
        "media_effects": visualizer.MediaEffects(_mmm),
        "model_fit": visualizer.ModelFit(_mmm),
        "diagnostics": visualizer.ModelDiagnostics(_mmm),
    }

viz = make_visualizers(mmm)
media_summary = viz["media_summary"]
media_effects = viz["media_effects"]
model_fit = viz["model_fit"]
diagnostics = viz["diagnostics"]

# ----------------------------
# Sidebar controls for plots
# ----------------------------
st.sidebar.header("Plot options")

time_granularity = st.sidebar.selectbox("Time granularity (MediaSummary)", ["quarterly", "weekly"], index=0)
include_ci = st.sidebar.checkbox("Include credible intervals (where applicable)", value=True)
confidence_level = st.sidebar.slider("Confidence level", min_value=0.50, max_value=0.99, value=0.90, step=0.01)

disable_bubble_size = st.sidebar.checkbox("Disable bubble sizing (ROI bubbles)", value=False)
equal_axes = st.sidebar.checkbox("Equal axes (ROI vs mROI)", value=False)

plot_separately = st.sidebar.checkbox("Response curves: facet separately", value=True)
num_channels_displayed = st.sidebar.number_input(
    "Response curves: num channels on layered plot (ignored if faceted)",
    min_value=1,
    value=7,
    step=1,
)

# ----------------------------
# Helper: render Altair charts safely
# ----------------------------
def show_chart(chart, *, use_container_width=True):
    if chart is None:
        st.info("No chart returned.")
        return
    st.altair_chart(chart, use_container_width=use_container_width)

# ----------------------------
# Tabs
# ----------------------------
tab_fit, tab_diag, tab_summary, tab_effects, tab_tables = st.tabs(
    ["Model fit", "Diagnostics", "Media summary", "Media effects", "Tables"]
)

# ----------------------------
# MODEL FIT
# ----------------------------
with tab_fit:
    st.subheader("Expected vs actual outcome (ModelFit.plot_model_fit)")
    # Note: ModelFit uses its own confidence_level internally; you can update it
    model_fit.update_confidence_level(confidence_level)
    fit_chart = model_fit.plot_model_fit(
        include_baseline=True,
        include_ci=include_ci,
    )
    show_chart(fit_chart)

# ----------------------------
# DIAGNOSTICS
# ----------------------------
with tab_diag:
    st.subheader("Convergence (R-hat) — ModelDiagnostics.plot_rhat_boxplot")
    show_chart(diagnostics.plot_rhat_boxplot())

    st.subheader("Prior vs posterior for a parameter — ModelDiagnostics.plot_prior_and_posterior_distribution")
    param = st.selectbox("Parameter", ["roi_m", "alpha_m", "ec_m", "beta_gm"], index=0)
    try:
        dist_chart = diagnostics.plot_prior_and_posterior_distribution(parameter=param)
        show_chart(dist_chart)
    except Exception as e:
        st.warning(f"Could not plot prior/posterior for `{param}`: {e}")

# ----------------------------
# MEDIA SUMMARY
# ----------------------------
with tab_summary:
    st.subheader("Contribution over time — MediaSummary.plot_channel_contribution_area_chart")
    show_chart(media_summary.plot_channel_contribution_area_chart(time_granularity=time_granularity))

    st.subheader("Contribution rank over time — MediaSummary.plot_channel_contribution_bump_chart")
    show_chart(media_summary.plot_channel_contribution_bump_chart(time_granularity=time_granularity))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Contribution waterfall — MediaSummary.plot_contribution_waterfall_chart")
        show_chart(media_summary.plot_contribution_waterfall_chart())
    with c2:
        st.subheader("Contribution pie — MediaSummary.plot_contribution_pie_chart")
        show_chart(media_summary.plot_contribution_pie_chart())

    st.subheader("Spend vs contribution — MediaSummary.plot_spend_vs_contribution")
    show_chart(media_summary.plot_spend_vs_contribution())

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("ROI by channel — MediaSummary.plot_roi_bar_chart")
        show_chart(media_summary.plot_roi_bar_chart(include_ci=include_ci))
    with c4:
        st.subheader("CPIK by channel — MediaSummary.plot_cpik")
        show_chart(media_summary.plot_cpik(include_ci=include_ci))

    st.subheader("ROI bubbles — MediaSummary.plot_roi_vs_effectiveness")
    show_chart(
        media_summary.plot_roi_vs_effectiveness(
            selected_channels=None,
            disable_size=disable_bubble_size,
        )
    )

    st.subheader("ROI vs mROI bubbles — MediaSummary.plot_roi_vs_mroi")
    show_chart(
        media_summary.plot_roi_vs_mroi(
            selected_channels=None,
            disable_size=disable_bubble_size,
            equal_axes=equal_axes,
        )
    )

# ----------------------------
# MEDIA EFFECTS
# ----------------------------
with tab_effects:
    st.subheader("Response curves — MediaEffects.plot_response_curves")
    show_chart(
        media_effects.plot_response_curves(
            confidence_level=confidence_level,
            plot_separately=plot_separately,
            include_ci=include_ci,
            num_channels_displayed=None if plot_separately else int(num_channels_displayed),
        )
    )

    st.subheader("Adstock decay — MediaEffects.plot_adstock_decay")
    try:
        adstock_chart = media_effects.plot_adstock_decay(
            confidence_level=confidence_level,
            include_ci=include_ci,
        )
        show_chart(adstock_chart)
    except Exception as e:
        st.warning(f"Could not plot adstock decay: {e}")

    st.subheader("Hill saturation curves — MediaEffects.plot_hill_curves")
    try:
        hill_charts = media_effects.plot_hill_curves(
            confidence_level=confidence_level,
            include_prior=True,
            include_ci=include_ci,
        )
        if not hill_charts:
            st.info("No Hill curves available for this model/data.")
        else:
            for k, ch in hill_charts.items():
                st.markdown(f"**{k}**")
                show_chart(ch)
    except Exception as e:
        st.warning(f"Could not plot Hill curves: {e}")

# ----------------------------
# TABLES
# ----------------------------
with tab_tables:
    st.subheader("Media summary table — MediaSummary.summary_table")
    currency = st.text_input("Currency symbol", value="$")
    try:
        summary_df = media_summary.summary_table(
            include_prior=True,
            include_posterior=True,
            include_non_paid_channels=False,
            currency=currency,
        )
        st.dataframe(summary_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate summary table: {e}")

    st.subheader("Predictive accuracy — ModelDiagnostics.predictive_accuracy_table")
    try:
        acc_df = diagnostics.predictive_accuracy_table()
        st.dataframe(acc_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute predictive accuracy table: {e}")

# ----------------------------
# Optional: download model file path (just for convenience)
# ----------------------------
st.divider()
st.markdown("### Notes")
st.write(
    "- Meridian’s charting is mainly through `meridian.analysis.visualizer.*` (MediaSummary, MediaEffects, ModelFit, ModelDiagnostics)."
)
st.write(
    "- If you want to **save** a model after fitting: prefer protobuf via `schema.serde.meridian_serde.save_meridian(mmm, 'model.binpb')`."
)
