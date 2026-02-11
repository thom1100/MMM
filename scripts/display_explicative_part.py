from pathlib import Path
import pandas as pd
import arviz as az
import tensorflow_probability as tfp

from meridian.data import load
from meridian import model, spec
from meridian.model import prior_distribution, constants

MODELS_DIR = Path("models")

# --------------- 1) Granularity routing (CSV + input_data) -----------------

GRANULARITY_TO_CSV = {
    "df_l4": "path/to/df_l4.csv",
    "df_l5": "path/to/df_l5.csv",
}

def build_input_data_from_granularity(granularity: str):
    csv_path = GRANULARITY_TO_CSV[granularity]
    df = pd.read_csv(csv_path)

    df["total_units"] = (
        df["UK L'Oreal Paris Haircare Total Online Sellout Units"]
        + df["UK L'Oreal Paris Haircare Total Offline Sellout Units"]
    )

    coord_to_columns = load.CoordToColumns(
        time="Starting week",
        kpi="total_units",
        controls=[
            "UK L'Oreal Paris Haircare Offline Average Price (in pound)",
            "UK L'Oreal Paris Haircare Online Average Price (in pound)",
            "UK L'Oreal Paris Haircare Total Weigheted Promotion Distribution (%)",
        ],
        media=[
            "impressions_digital_tv",
            "impressions_online_multiformat_ads_transaction",
            "impressions_online_video_content_platforms",
            "impressions_paid_search_awarenessconsideration",
            "impressions_paid_search_transaction",
            "impressions_social_media_awarenessconsideration",
            "impressions_social_media_transaction",
        ],
        media_spend=[
            "spend_digital_tv",
            "spend_online_multiformat_ads_transaction",
            "spend_online_video_content_platforms",
            "spend_paid_search_awarenessconsideration",
            "spend_paid_search_transaction",
            "spend_social_media_awarenessconsideration",
            "spend_social_media_transaction",
        ],
    )

    media_to_channel = {
        "impressions_digital_tv": "digital_tv",
        "impressions_online_multiformat_ads_transaction": "multiformat_ads",
        "impressions_online_video_content_platforms": "video_content_platforms",
        "impressions_paid_search_awarenessconsideration": "paid_search_awareness_consideration",
        "impressions_paid_search_transaction": "search_transaction",
        "impressions_social_media_awarenessconsideration": "social_media_awareness_consideration",
        "impressions_social_media_transaction": "social_media_transaction",
    }

    media_spend_to_channel = {
        "spend_digital_tv": "digital_tv",
        "spend_online_multiformat_ads_transaction": "multiformat_ads",
        "spend_online_video_content_platforms": "video_content_platforms",
        "spend_paid_search_awarenessconsideration": "paid_search_awareness_consideration",
        "spend_paid_search_transaction": "search_transaction",
        "spend_social_media_awarenessconsideration": "social_media_awareness_consideration",
        "spend_social_media_transaction": "social_media_transaction",
    }

    loader = load.DataFrameDataLoader(
        df=df,
        kpi_type="non_revenue",
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )

    return loader.load()


# --------------- 2) Prior-quality routing (prior + model_spec) --------------

PRIOR_QUALITY_TO_ROI = {
    # your current one could be "baseline" or "wide"
    "baseline": dict(mu=0.2, sigma=0.9),

    # the 3 classic ones
    "optimistic": dict(mu=0.5, sigma=0.6),
    "neutral": dict(mu=0.0, sigma=0.5),
    "pessimistic": dict(mu=-0.7, sigma=0.6),
}

def build_model_spec_from_prior_quality(prior_quality: str):
    cfg = PRIOR_QUALITY_TO_ROI[prior_quality]

    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(cfg["mu"], cfg["sigma"], name=constants.ROI_M)
    )

    return spec.ModelSpec(
        knots=20,
        prior=prior,
        media_prior_type="coefficient",
        max_lag=18,
    )


# --------------- 3) Load pretrained posterior and return ready model --------

def load_pretrained_meridian(granularity: str, prior_quality: str):
    # Build the matching input_data + model_spec
    input_data = build_input_data_from_granularity(granularity)
    model_spec = build_model_spec_from_prior_quality(prior_quality)

    # Load posterior (InferenceData) saved earlier
    key = f"{granularity}_{prior_quality}"
    path = MODELS_DIR / f"{key}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved model posterior: {path}")

    idata = az.from_netcdf(path)

    # Recreate Meridian object and attach posterior
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    mmm.inference_data = idata

    return mmm, key