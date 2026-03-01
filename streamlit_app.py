import datetime
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

## PAGE SETUP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="TEST California Cities Map", layout="wide")


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

    # Strip the "EVT_FUEL_N_" prefix from each encoded column name to get clean labels: Grass or Shrub for the sidebar selectbox
    fuel_options = [c.replace("EVT_FUEL_N_", "") for c in fuel_columns]
    return model, all_features, fuel_options


model, feature_names, fuel_options = load_assets()


# Prediction Function
def make_prediction(input_data):

    # start the dataframe filled in with 0s because model was trained on set number of columns
    # any column that is abset we have should be 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # filing the keys from the inputs we place, any values not found in the dataframe are ignored
    for key, value in input_data.items():
        if key in input_df.columns:
            input_df[key] = value

    # Model wasn't trained on grass string or smthn so it essentially the specific fuel type was switched into
    # a binary value and so when that value is selected it is set to 1 in the dataframe and everything else is 0
    selected_fuel_col = f"EVT_FUEL_N_{input_data['selected_fuel']}"
    if selected_fuel_col in input_df.columns:
        input_df[selected_fuel_col] = 1

    # predict_proba returns a 2D array of shape of [0] gets the first (and only) row,
    # [1] gets the probability of class 1

    prob = model.predict_proba(input_df)[0][1]
    st.write("Non-zero features:", input_df.loc[:, (input_df != 0).any()].to_dict())
    st.write("Raw probability:", prob)
    st.stop()
    return prob

st.write(fuel_options)

# ── Session Reset Feature ─────────────────────────────────────────────────────────────
if "version" not in st.session_state:
    st.session_state.version = 0


def reset_all():
    st.session_state.version += 1
    st.rerun()


v = st.session_state.version

# ── Header ─────────────────────────────────────────────────────────────

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
    f"Data from: Acquired from [California Landfire]({url1}), [Meteostat]({url2}), [NASA FIRMS]({url3})"
)
st.write(f"Predictions Made Through a [Random Forest Classifier Model (RCF)]({url4})")
st.write(
    f"Data Engineering Team: [Emiliano]({emiliano}), [Arjun]({arjun}), [Will]({will}) | RCF Team: [Aliya]({aliya}), [Joseph]({joseph}) | Streamlit Team: [Lipika]({lipika}), [Hayden]({hayden})"
)

st.markdown("---")

## SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Adjust Risk Factors:")

    # ── Location ──────────────────────────────────────────────────────────────
    latitude = st.slider(
        "Latitude",
        32.5,
        42.0,
        34.0689,
        step=0.01,
        key=f"lat_{v}",
        format="%.2f",
        # value=st.session_state.get("lat_input",34.0689),
    )  # changed these so that they only have 2 decimal points

    longitude = st.slider(
        "Longitutde",
        -124.4,
        -114.1,
        -118.4452,
        key=f"lon_{v}",
        step=0.01,
        format="%.2f",
    )
    # value=st.session_state.get("lon_input",-118.4452),
    # key = "lon_input") # changed these so that they only have 2 decimal points

    # acq_hour = st.slider("Acquired Hour:",
    # 0, 23, 12,
    # key=f"hour_{v}"
    # )
    # value = st.session_state.get("hour_input", 12),
    # key = "hour_input")

    st.markdown("---")

    # ── Weather ───────────────────────────────────────────────────────────────
    wx_tavg_c = st.number_input(
        "Average Daily Temperature (C)",
        0,
        key=f"temp_{v}",
        step=1,
        format="%d",
    )
    # value = st.session_state.get("temp_input", 12),
    # key = "temp_input")

    wx_prcp_mm = st.number_input(
        "Total Daily Precipitation (mm)", 0.0, key=f"prec_{v}", step=0.01, format="%.2f"
    )
    # value = st.session_state.get("prec_input", 0.0),
    # key = "prec_input") # changed these so that they only have 2 decimal points

    wx_wspd_ms = st.number_input(
        "Wind Speed (m/s)",
        0.0,
        key=f"wind_{v}",
        step=0.01,
        format="%.2f",
    )
    # value = st.session_state.get("wind_input", 0.0),
    # key = "wind_input") # changed these so that they only have 2 decimal points

    snow = st.selectbox(
        "Snow Present?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x else "No",
        key=f"snow_{v}",
    )

    st.markdown("---")

    # ── Vegetation ───────────────────────────────────────────────────────────────
    lf_evc = st.slider("Vegetation Cover (%)", 0, 100, 50, key=f"cov_{v}")
    # value = st.session_state.get("veg_cov_input", 50),
    # key = "veg_cov_input")

    lf_evh = st.slider("Vegetation Height (cm)", 0, 1000, 100, key=f"hei_{v}")
    # value = st.session_state.get("veg_hei_input", 100),
    # key = "veg_hei_input")

    st.markdown("---")

    # ── Fuel Type ─────────────────────────────────────────────────────────────
    st.subheader("Fuel Type")
    evt_fuel_n = st.selectbox(
        "Existing Vegetation Fuel Type", options=fuel_options, key=f"fuel_{v}"
    )

    # ── Date ──────────────────────────────────────────────────────────────────
    st.subheader("Date")
    selected_date = st.date_input(
        "Prediction Date", value=datetime.date.today(), key=f"date_{v}"
    )
    # month, day_of_year, year all derived from the single date picker
    pred_month = selected_date.month
    pred_day_of_year = selected_date.timetuple().tm_yday
    pred_year = selected_date.year

    st.markdown("---")

col_btn1, col_btn2 = st.columns(2)

# ── Input Directory ────────────────────────────────────────────────────────────
input_dict = {
    "latitude": latitude,
    "longitude": longitude,
    "month": pred_month,
    "day_of_year": pred_day_of_year,
    "year": pred_year,
    "wx_tavg_c": wx_tavg_c,
    "wx_prcp_mm": wx_prcp_mm,
    "wx_wspd_ms": wx_wspd_ms,
    "snow": snow,
    "lf_evc": lf_evc,
    "lf_evh": lf_evh,
    "selected_fuel": evt_fuel_n,
}

# ── buttons ────────────────────────────────────────────────────────────
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predict_clicked = st.button("Predict Wildfire Risk", type="primary")

with col_btn2:
    if st.button("Clear Values"):
        reset_all()

if predict_clicked:
    with st.spinner("Running prediction..."):
        st.session_state.risk = make_prediction(input_dict)

# ── Dataframe Setup ────────────────────────────────────────────────────
data = {
    "Pred Date": [selected_date.strftime("%B %d, %Y")],
    "Month": [pred_month],
    "Day of Year": [pred_day_of_year],
    "Year": [pred_year],
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

st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=4)


# ── Risk result ───────────────────────────────────────────────────────────────
risk = st.session_state.get("risk", 0.0)
# print(np.percentile(y_prob, [25, 50, 70, 85, 95, 99]))-->
# [0.29101556 0.62917867 0.83271943 0.87625205 0.901101   0.91034244]
risk = st.session_state.get("risk", None)

if risk is not None:
    if risk < 0.291:
        risk_label = "Very Low"
    elif risk < 0.629:
        risk_label = "Low"
    elif risk < 0.833:
        risk_label = "Moderate"
    elif risk < 0.876:
        risk_label = "High"
    else:
        risk_label = "Extreme"

    st.metric("Wildfire Risk", f"{risk:.1%}", delta=risk_label, delta_color="off")

# ight everything should be good for you to finish up the map and adding in the national park risk thing
# ran a quick print in the mlmodel to recalibrate our risk labeling to better fit with our models biases
