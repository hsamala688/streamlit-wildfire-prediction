import datetime
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
from PIL import Image

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
        "Desert Scrub (Mojave / Sonora)": "Sh Sonora-Mojave Creosotebush-White Bursage Desert Scrub",
        "Sagebrush / Great Basin Shrub": "Sh Inter-Mountain Basins Big Sagebrush Shrubland",
        "Grassland": "He California Central Valley and Southern Coastal Grassland",
        "Marsh / Wetland": "He Temperate Pacific Freshwater Emergent Marsh",
        "Oak Woodland / Savanna": "Tr California Lower Montane Blue Oak Forest and Woodland",
        "Mixed Conifer Forest": "Tr Mediterranean California Mesic Mixed Conifer Forest and Woodland",
        "Subalpine / Alpine Forest": "Tr Mediterranean California Subalpine Woodland",
        "Riparian / Streamside Forest": "Tr California Central Valley Riparian Woodland and Shrubland",
        "Coastal Redwood / Closed-Cone": "Tr California Coastal Redwood Forest",
        "Agricultural / Cropland": "Da Row Crop",
        "Developed / Urban": "Bau Developed-Low Intensity",
        "Sparse / Bare Vegetation": "Sps Mediterranean California Sparsely Vegetated Systems",
        "Snow / Ice": "Ba Snow-Ice",
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

# ── Map Setup ───────────────────────────────────────────────────────────────
# st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=4)
ca_lat = [32.5, 42.0]
ca_long = [-124.4, -114.1]

map_col, risk_col = st.columns([0.7, 0.3])

with map_col:
    m = folium.Map(location=[np.average(ca_lat), np.average(ca_long)], 
                    zoom_start=4.5, control_scale=True)      # use the mean latitude and longitude to center the map

    # loop through each row in the dataframe & add to map using latitude & longitude
    for i,row in df.iterrows():
        # set up the content of the popup
        iframe = folium.IFrame("Risk at (" + str(row["Lat"]) + ", " + str(row["Lon"]) + ") is " + str(int(make_prediction(input_dict)*100)) + "%.", width=300, height=25)

        # initialize the popup using the iframe
        popup = folium.Popup(iframe, max_width=500)

        # add each row to the map with latitude & longitude
        folium.Marker(location=[row['Lat'],row['Lon']],
                    popup = popup).add_to(m)

    st_data = folium_static(m)
    fig = px.scatter_map(df, lat="Lat", lon="Lon", zoom=4.5)


# ── Risk result ───────────────────────────────────────────────────────────────
st.session_state.risk = make_prediction(input_dict)
risk = st.session_state.get("risk", 0.0)

arrow_angle = 0
text_color = "green"

if risk is not None:
    if risk < 0.291:
        risk_label = "Very Low"
        arrow_angle = -18*1
        text_color = "green"
    elif risk < 0.629:
        risk_label = "Low"
        arrow_angle = -18*3
        text_color = "blue"
    elif risk < 0.833:
        risk_label = "Moderate"
        arrow_angle = -18*5
        text_color = "yellow"
    elif risk < 0.876:
        risk_label = "High"
        arrow_angle = -18*7
        text_color = "orange"
    else:
        risk_label = "Extreme"
        arrow_angle = -18*9
        text_color = "red"

    risk_col.header(f"Wildfire Risk:", text_alignment="center")
    risk_col.subheader(f":{text_color}[{risk_label}]", text_alignment="center")

bg = Image.open("wildfire_meter.png").convert("RGBA")
arrow = Image.open("arrow.png").convert("RGBA")

scale_factor = 0.20
new_width = int(bg.width * scale_factor)
aspect_ratio = arrow.height / arrow.width
new_height = int(new_width * aspect_ratio)
arrow_resized = arrow.resize((new_width, new_height), Image.LANCZOS)

rotated_arrow = arrow_resized.rotate(
    90+arrow_angle,
    resample=Image.BICUBIC,
    expand=True)

combined = bg.copy()

bg_w, bg_h = combined.size
position = (bg_w // 4, bg_h // 2)
combined.paste(rotated_arrow, position, rotated_arrow)
risk_col.image(combined.convert("RGB"), width="stretch")

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
