%%writefile /kaggle/working/app.py

import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Weather Station Anomaly Detection",
    page_icon="🌦️",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "/kaggle/input/datasets/chandananedium/weather-anomaly-model-files/weather_anomaly_model.pkl"
SCALER_PATH = "/kaggle/input/datasets/chandananedium/weather-anomaly-model-files/weather_scaler.pkl"
FEATURES_PATH = "/kaggle/input/datasets/chandananedium/weather-anomaly-model-files/model_features.pkl"
HISTORY_PATH = "/kaggle/working/historical_weather_anomalies.csv"

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    return model, scaler, model_features

model, scaler, model_features = load_model()

# -----------------------------
# Gemini
# -----------------------------
@st.cache_resource
def load_gemini():
    try:
        from kaggle_secrets import UserSecretsClient
        from google import genai

        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(api_key=api_key)

    except Exception:
        return None


gemini_client = load_gemini()

# -----------------------------
# Anomaly Type
# -----------------------------
def identify_anomaly_type(
    temperature,
    humidity,
    pressure,
    wind_speed,
    wind_direction
):
    anomalies = []

    if temperature >= 45 or temperature <= -20:
        anomalies.append("Temperature Anomaly")

    if humidity <= 10 or humidity >= 100:
        anomalies.append("Humidity Anomaly")

    if pressure >= 1080 or pressure <= 950:
        anomalies.append("Pressure Anomaly")

    if wind_speed >= 20:
        anomalies.append("Wind Speed Anomaly")

    if wind_direction < 0 or wind_direction > 360:
        anomalies.append("Wind Direction Anomaly")

    if len(anomalies) >= 2:
        return "Multiple Sensor Anomaly"

    if len(anomalies) == 1:
        return anomalies[0]

    return "Pattern-Based Anomaly"


# -----------------------------
# Severity
# -----------------------------
def get_severity(
    score,
    temperature,
    humidity,
    pressure,
    wind_speed
):
    # Extreme physical conditions
    if (
        temperature >= 50
        or temperature <= -20
        or humidity <= 10
        or pressure >= 1080
        or pressure <= 950
        or wind_speed >= 30
    ):
        return "High"

    if (
        temperature >= 45
        or temperature <= -10
        or humidity <= 20
        or pressure >= 1060
        or pressure <= 970
        or wind_speed >= 20
    ):
        return "Medium"

    # ML score
    if score < -0.2:
        return "High"

    if score < -0.1:
        return "Medium"

    return "Low"


# -----------------------------
# Gemini Explanation
# -----------------------------
def get_gemini_analysis(
    temperature,
    humidity,
    pressure,
    wind_speed,
    wind_direction,
    anomaly_type,
    severity,
    anomaly_score
):
    prompt = f"""
You are assisting an automatic weather station anomaly detection system.

The ML model has detected a weather station reading that requires attention.

Temperature: {temperature} °C
Humidity: {humidity} %
Pressure: {pressure} mbar
Wind Speed: {wind_speed} m/s
Wind Direction: {wind_direction}°
Anomaly Type: {anomaly_type}
Severity: {severity}
Isolation Forest Score: {anomaly_score:.4f}

Give a concise response with exactly these three sections:

Explanation:
Explain why this reading is unusual.

Possible Cause:
Give possible environmental or sensor-related causes. Do not claim certainty.

Recommended Action:
Give practical actions for a weather station operator.

Keep the answer simple and suitable for a dashboard.
"""

    if gemini_client is None:
        return (
            "Explanation: The weather station reading contains unusual "
            "sensor values that require attention.\n\n"
            "Possible Cause: The anomaly may be caused by an unusual "
            "weather event or a sensor/data-quality issue.\n\n"
            "Recommended Action: Compare the reading with nearby stations "
            "and inspect the sensors if the abnormal values continue."
        )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        if response and response.text:
            return response.text

    except Exception:
        pass

    return (
        "Explanation: The weather station reading contains unusual "
        "sensor values that require attention.\n\n"
        "Possible Cause: The anomaly may be caused by an unusual "
        "weather event or a sensor/data-quality issue.\n\n"
        "Recommended Action: Compare the reading with nearby stations "
        "and inspect the sensors if the abnormal values continue."
    )


# -----------------------------
# Current Reading Detection
# -----------------------------
def detect_anomaly(
    temperature,
    humidity,
    pressure,
    wind_speed,
    wind_direction,
    hour,
    month
):
    data = pd.DataFrame([{
        "T (degC)": temperature,
        "rh (%)": humidity,
        "p (mbar)": pressure,
        "wv (m/s)": wind_speed,
        "wd (deg)": wind_direction,

        "hour": hour,
        "month": month,

        # Simplified values for single-reading prototype
        "temp_rolling_mean": temperature,
        "temp_rolling_std": 0,
        "humidity_rolling_mean": humidity,
        "humidity_rolling_std": 0,
        "pressure_rolling_mean": pressure,
        "pressure_rolling_std": 0,

        "temp_change": 0,
        "humidity_change": 0,
        "pressure_change": 0
    }])

    data = data[model_features]

    X_scaled = scaler.transform(data)

    prediction = model.predict(X_scaled)[0]
    anomaly_score = model.decision_function(X_scaled)[0]

    # Physical sensor safety check
    extreme_sensor_condition = (
        temperature >= 50
        or temperature <= -20
        or humidity <= 10
        or pressure >= 1080
        or pressure <= 950
        or wind_speed >= 30
        or wind_direction < 0
        or wind_direction > 360
    )

    # Final anomaly decision
    # Isolation Forest remains the ML detector.
    # Physical check acts as an additional safety layer.
    is_anomaly = prediction == -1 or extreme_sensor_condition

    anomaly_type = identify_anomaly_type(
        temperature,
        humidity,
        pressure,
        wind_speed,
        wind_direction
    )

    if not is_anomaly:
        anomaly_type = "Normal Reading"

    severity = get_severity(
        anomaly_score,
        temperature,
        humidity,
        pressure,
        wind_speed
    )

    if not is_anomaly:
        severity = "Normal"

    return (
        is_anomaly,
        anomaly_score,
        anomaly_type,
        severity
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌦️ Weather Station Input")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-50.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.1
)

pressure = st.sidebar.number_input(
    "Pressure (mbar)",
    min_value=900.0,
    max_value=1150.0,
    value=1015.0,
    step=0.1
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=60.0,
    value=2.5,
    step=0.1
)

wind_direction = st.sidebar.number_input(
    "Wind Direction (°)",
    min_value=0.0,
    max_value=360.0,
    value=180.0,
    step=1.0
)

hour = st.sidebar.number_input(
    "Hour",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)

month = st.sidebar.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=9,
    step=1
)

detect_button = st.sidebar.button(
    "🔍 Detect Anomaly",
    use_container_width=True
)

# -----------------------------
# Title
# -----------------------------
st.title("🌦️ Intelligent Weather Station Anomaly Detection")

st.write(
    "AI/ML-based monitoring system for detecting unusual "
    "weather station observations and generating actionable explanations."
)

# -----------------------------
# Detection
# -----------------------------
if detect_button:

    (
        is_anomaly,
        anomaly_score,
        anomaly_type,
        severity
    ) = detect_anomaly(
        temperature,
        humidity,
        pressure,
        wind_speed,
        wind_direction,
        hour,
        month
    )

    st.subheader("Current Weather Reading")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Temperature", f"{temperature:.1f} °C")
    c2.metric("Humidity", f"{humidity:.1f} %")
    c3.metric("Pressure", f"{pressure:.1f} mbar")
    c4.metric("Wind Speed", f"{wind_speed:.1f} m/s")
    c5.metric("Wind Direction", f"{wind_direction:.0f}°")

    st.divider()

    if is_anomaly:
        st.error("⚠️ ANOMALY DETECTED")
    else:
        st.success("✅ NORMAL READING")

    a1, a2, a3 = st.columns(3)

    a1.metric("Anomaly Type", anomaly_type)
    a2.metric("Severity", severity)
    a3.metric("Anomaly Score", f"{anomaly_score:.4f}")

    st.subheader("🤖 Gemini AI Analysis")

    with st.spinner("Generating AI analysis..."):

        analysis = get_gemini_analysis(
            temperature,
            humidity,
            pressure,
            wind_speed,
            wind_direction,
            anomaly_type,
            severity,
            anomaly_score
        )

    st.info(analysis)

else:

    st.info(
        "Enter weather station values in the sidebar and click "
        "**Detect Anomaly**."
    )

# -----------------------------
# Historical Analysis
# -----------------------------
st.divider()

st.header("📊 Historical Weather Analysis")

if os.path.exists(HISTORY_PATH):

    try:
        historical_df = pd.read_csv(HISTORY_PATH)

        if "Date Time" in historical_df.columns:
            historical_df["Date Time"] = pd.to_datetime(
                historical_df["Date Time"],
                errors="coerce"
            )

        total_readings = len(historical_df)

        if "prediction" in historical_df.columns:
            anomaly_count = (
                historical_df["prediction"] == -1
            ).sum()
        else:
            anomaly_count = 0

        anomaly_percentage = (
            anomaly_count / total_readings * 100
            if total_readings > 0
            else 0
        )

        h1, h2, h3 = st.columns(3)

        h1.metric(
            "Historical Readings",
            f"{total_readings:,}"
        )

        h2.metric(
            "Detected Anomalies",
            f"{anomaly_count:,}"
        )

        h3.metric(
            "Anomaly Percentage",
            f"{anomaly_percentage:.2f}%"
        )

        # -----------------------------
        # Temperature Trend
        # -----------------------------
        st.subheader("🌡️ Temperature Trend")

        if "T (degC)" in historical_df.columns:

            temp_chart = historical_df[
                ["Date Time", "T (degC)"]
            ].dropna()

            if not temp_chart.empty:

                temp_chart = temp_chart.set_index("Date Time")

                st.line_chart(
                    temp_chart["T (degC)"]
                )

        # -----------------------------
        # Historical Anomalies
        # -----------------------------
        st.subheader("🔴 Historical Anomaly Points")

        if (
            "T (degC)" in historical_df.columns
            and "prediction" in historical_df.columns
        ):

            anomaly_points = historical_df[
                historical_df["prediction"] == -1
            ][
                ["Date Time", "T (degC)"]
            ].dropna()

            if not anomaly_points.empty:

                anomaly_points = anomaly_points.set_index(
                    "Date Time"
                )

                st.scatter_chart(
                    anomaly_points["T (degC)"]
                )

            else:
                st.info("No historical anomalies found.")

        # -----------------------------
        # Humidity Trend
        # -----------------------------
        st.subheader("💧 Humidity Trend")

        if "rh (%)" in historical_df.columns:

            humidity_chart = historical_df[
                ["Date Time", "rh (%)"]
            ].dropna()

            if not humidity_chart.empty:

                humidity_chart = humidity_chart.set_index(
                    "Date Time"
                )

                st.line_chart(
                    humidity_chart["rh (%)"]
                )

        # -----------------------------
        # Anomaly Table
        # -----------------------------
        st.subheader("⚠️ Detected Anomalies")

        if "prediction" in historical_df.columns:

            anomaly_table = historical_df[
                historical_df["prediction"] == -1
            ].copy()

            if not anomaly_table.empty:

                display_columns = [
                    "Date Time",
                    "T (degC)",
                    "rh (%)",
                    "p (mbar)",
                    "wv (m/s)",
                    "wd (deg)",
                    "anomaly_score",
                    "anomaly_type",
                    "severity"
                ]

                available_columns = [
                    col
                    for col in display_columns
                    if col in anomaly_table.columns
                ]

                st.dataframe(
                    anomaly_table[available_columns],
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = anomaly_table.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download Anomaly Report",
                    data=csv_data,
                    file_name="historical_weather_anomalies.csv",
                    mime="text/csv"
                )

            else:
                st.info("No anomalies available.")

    except Exception as e:

        st.error(
            f"Unable to load historical analysis: {e}"
        )

else:

    st.warning(
        "Historical anomaly file not found at "
        f"{HISTORY_PATH}"
    )

# -----------------------------
# System Architecture
# -----------------------------
st.divider()

st.header("⚙️ System Architecture")

st.markdown(
    """
**Weather Station Data**  
↓  
**Feature Processing**  
↓  
**Isolation Forest ML Model**  
↓  
**Normal / Anomaly Detection**  
↓  
**Sensor Safety Check**  
↓  
**Anomaly Type + Severity**  
↓  
**Gemini AI Explanation**  
↓  
**Recommended Action**
"""
)

# -----------------------------
# System Status
# -----------------------------
st.header("🟢 System Status")

s1, s2, s3 = st.columns(3)

s1.success("ML Detector\n\nIsolation Forest")
s2.success("Historical Analysis\n\nActive")
s3.success("AI Assistant\n\nGemini")
