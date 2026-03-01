import datetime
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

## PAGE SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="California Wildfire Prediction", layout="wide")


# Asset Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    path = "model_assets"
    model = joblib.load(os.path.join(path, "wildfire_model.pkl"))

    # fuel_encoder.pkl stores the list of one-hot encoded column names that were
    # created from the EVT_FUEL_N categorical feature during training
    # (we need the exact same structure for predictions to work correctly)
    fuel_columns = joblib.load(os.path.join(path, "fuel_encoder.pkl"))

    # feature_names.pkl stores the list of all feature column names used during training
    # (we need the exact same structure for predictions to work correctly)
    all_features = joblib.load(os.path.join(path, "feature_names.pkl"))

    # Strip the "EVT_FUEL_N_" prefix from each encoded column name to get clean labels
    fuel_options = [c.replace("EVT_FUEL_N_", "") for c in fuel_columns]
    return model, all_features, fuel_options


model, feature_names, fuel_options = load_assets()


def make_prediction(input_data):
    input_df = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

    for key, value in input_data.items():
        if key in input_df.columns:
            input_df[key] = value

    selected_fuel_col = f"EVT_FUEL_N_{input_data['selected_fuel']}"
    if selected_fuel_col in input_df.columns:
        input_df[selected_fuel_col] = 1

    try:
        prob = model.predict_proba(input_df)[0][1]
    except ValueError as e:
        st.error(f"Full error: {e}")
        st.stop()

    return prob


# ── Region Coordinates ────────────────────────────────────────────────────
region_coords = {
    "Northern CA (Redding / Shasta)": (40.6, -122.4),
    "Sierra Nevada Foothills": (38.9, -120.7),
    "Sacramento Valley": (38.5, -121.5),
    "San Francisco Bay Area": (37.7, -122.2),
    "Central Valley": (36.7, -119.8),
    "Central Coast": (35.3, -120.7),
    "Los Angeles / SoCal Coast": (34.1, -118.4),
    "Inland Empire / San Bernardino": (34.1, -117.3),
    "San Diego": (32.8, -117.1),
    "Mojave Desert": (34.9, -116.9),
}


# ── Session Reset ─────────────────────────────────────────────────────────
if "version" not in st.session_state:
    st.session_state.version = 0


def reset_all():
    st.session_state.version += 1
    st.rerun()


v = st.session_state.version

# ── Header ────────────────────────────────────────────────────────────────
st.title("California Wildfire Prediction", anchor="main title")

url1 = "https://data.ca.gov/dataset/climate-land-cover-landfire-derived"
url2 = "https://meteostat.net/en/"
url3 = "https://firms.modaps.eosdis.nasa.gov/map/#d:24hrs;@0.0,0.0,3.0z"
url4 = "https://github.com/hsamala688/CaliforniaWildfirePrediction"
emiliano = "https://github.com/emilianotorneltaki"
aliya = "https://github.com/aliyatang"
joseph = "https://github.com/Potato12fff"
will = "https://github.com/wllamjp"
arjun = "https://github.com/ArjunBrahmandam"
lipika = "https://github.com/lipikagoel"
hayden = "https://github.com/hsamala688"

st.write("National Student Data Corp @ UCLA, Winter 2026 Project")
st.write(
    f"Data from: [California Landfire]({url1}), [Meteostat]({url2}), [NASA FIRMS]({url3})"
)
st.write(f"Predictions Made Through a [Random Forest Classifier Model (RCF)]({url4})")
st.write(
    f"Data Engineering Team: [Emiliano]({emiliano}), [Arjun]({arjun}), [Will]({will}) | "
    f"RCF Team: [Aliya]({aliya}), [Joseph]({joseph}) | "
    f"Streamlit Team: [Lipika]({lipika}), [Hayden]({hayden})"
)

st.markdown("---")

## SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Adjust Risk Factors:")

    # ── Location ──────────────────────────────────────────────────────────
    selected_region = st.selectbox(
        "Region",
        options=list(region_coords.keys()),
        key=f"region_{v}",
    )
    latitude, longitude = region_coords[selected_region]
    st.caption(f"📍 Coordinates: ({latitude}, {longitude})")

    st.markdown("---")

    # ── Weather ───────────────────────────────────────────────────────────
    wx_tavg_c = st.number_input(
        "Average Daily Temperature (°C)",
        min_value=0,
        key=f"temp_{v}",
        step=1,
        format="%d",
    )

    wx_prcp_mm = st.number_input(
        "Total Daily Precipitation (mm)",
        min_value=0.0,
        key=f"prec_{v}",
        step=0.01,
        format="%.2f",
    )

    wx_wspd_ms = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        key=f"wind_{v}",
        step=0.01,
        format="%.2f",
    )

    snow = st.selectbox(
        "Snow Present?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x else "No",
        key=f"snow_{v}",
    )

    st.markdown("---")

    # ── Vegetation ────────────────────────────────────────────────────────
    lf_evc = st.slider("Vegetation Cover (%)", 0, 100, 50, key=f"cov_{v}")
    lf_evh = st.slider("Vegetation Height (cm)", 0, 1000, 100, key=f"hei_{v}")

    st.markdown("---")

    # ── Fuel Type ─────────────────────────────────────────────────────────────
    st.subheader("Fuel Type")

    friendly_fuel_map = {
        "Chaparral (Dense Shrubs)": "Sh Northern and Central California Dry-Mesic Chaparral",
        "Coastal Scrub": "Sh Southern California Coastal Scrub",
        "Grassland": "He Great Basin & Intermountain Introduced Annual Grassland",
        "Oak Woodland": "Tr California Lower Montane Blue Oak Forest and Woodland",
        "Mixed Conifer Forest": "Tr Mediterranean California Mesic Mixed Conifer Forest and Woodland",
        "Desert Scrub": "Sh Sonora-Mojave Creosotebush-White Bursage Desert Scrub",
        "Alpine / Subalpine": "Sh Sierra Nevada Alpine Dwarf-Shrubland",
        "Riparian / Wetland": "Tr California Central Valley Riparian Woodland and Shrubland",
        "Agricultural / Cropland": "Da Row Crop",
        "Developed / Urban": "Bau Developed-Low Intensity",
    }

    selected_friendly = st.selectbox(
        "Vegetation / Land Type",
        options=list(friendly_fuel_map.keys()),
        key=f"fuel_{v}",
    )
    evt_fuel_n = friendly_fuel_map[selected_friendly]

    # ── Date (display only) ───────────────────────────────────────────────
    st.subheader("Date")
    selected_date = st.date_input(
        "Prediction Date", value=datetime.date.today(), key=f"date_{v}"
    )
    if isinstance(selected_date, tuple):
        selected_date = selected_date[0]

    st.markdown("---")

# ── Buttons ───────────────────────────────────────────────────────────────
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predict_clicked = st.button("Predict Wildfire Risk", type="primary")

with col_btn2:
    if st.button("Clear Values"):
        reset_all()

# ── Input Dictionary ──────────────────────────────────────────────────────
input_dict = {
    "latitude": latitude,
    "longitude": longitude,
    "wx_tavg_c": wx_tavg_c,
    "wx_prcp_mm": wx_prcp_mm,
    "wx_wspd_ms": wx_wspd_ms,
    "snow": snow,
    "lf_evc": lf_evc,
    "lf_evh": lf_evh,
    "selected_fuel": evt_fuel_n,
}

if predict_clicked:
    with st.spinner("Running prediction..."):
        st.session_state.risk = make_prediction(input_dict)

# ── Display Table ─────────────────────────────────────────────────────────
data = {
    "Pred Date": [selected_date.strftime("%B %d, %Y")],
    "Region": [selected_region],
    "Lat": [latitude],
    "Lon": [longitude],
    "Avg Daily Temp (°C)": [wx_tavg_c],
    "Daily Precip (mm)": [wx_prcp_mm],
    "Wind Speed (m/s)": [wx_wspd_ms],
    "Snow Present": ["Yes" if snow else "No"],
    "Veg Cover (%)": [lf_evc],
    "Veg Height (cm)": [lf_evh],
    "Fuel Type": [evt_fuel_n],
}

df = pd.DataFrame(data)
df = df.rename(index={0: "Values:"})
numeric_cols = df.select_dtypes(include="number").columns
st.dataframe(df.style.format({c: "{:.2f}" for c in numeric_cols}))

st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

# ── Risk Result ───────────────────────────────────────────────────────────
risk = st.session_state.get("risk", None)

if risk is not None:
    # Thresholds recalibrated from retrained model percentiles:
    # [0.391, 0.597, 0.704, 0.754, 0.804, 0.830]
    if risk < 0.391:
        risk_label = "Very Low"
    elif risk < 0.597:
        risk_label = "Low"
    elif risk < 0.704:
        risk_label = "Moderate"
    elif risk < 0.754:
        risk_label = "High"
    else:
        risk_label = "Extreme"

    st.metric("Wildfire Risk", f"{risk:.1%}", delta=risk_label, delta_color="off")
