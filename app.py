import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌦️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_DIR, "weather_anomaly_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "weather_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
HISTORICAL_PATH = os.path.join(DATA_DIR, "historical_weather_anomalies.csv")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(FEATURES_PATH)

def identify_anomaly_type(temperature, humidity, pressure, wind_speed, wind_direction):
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

def get_severity(score, temperature, humidity, pressure, wind_speed):
    if (
        temperature >= 50 or temperature <= -20 or
        humidity <= 10 or
        pressure >= 1080 or pressure <= 950 or
        wind_speed >= 30
    ):
        return "High"

    if (
        temperature >= 45 or temperature <= -10 or
        humidity <= 20 or
        pressure >= 1060 or pressure <= 970 or
        wind_speed >= 20
    ):
        return "Medium"

    if score < -0.2:
        return "High"
    if score < -0.1:
        return "Medium"
    return "Low"

def local_explanation(anomaly_type, severity):
    causes = {
        "Temperature Anomaly": "The temperature is outside the configured sensor range.",
        "Humidity Anomaly": "The relative humidity is outside the configured sensor range.",
        "Pressure Anomaly": "The atmospheric pressure is outside the configured sensor range.",
        "Wind Speed Anomaly": "The wind speed is unusually high for the configured threshold.",
        "Wind Direction Anomaly": "The wind direction is outside the valid 0–360 degree range.",
        "Multiple Sensor Anomaly": "Multiple weather variables simultaneously crossed configured anomaly thresholds.",
        "Pattern-Based Anomaly": "The ML model identified an unusual multivariate pattern in the sensor readings."
    }

    actions = {
        "High": "Verify the affected sensor readings, inspect the station, and check for sensor faults or environmental extremes.",
        "Medium": "Recheck the sensor readings and monitor the station for repeated abnormal observations.",
        "Low": "Continue monitoring the station and compare with nearby or recent observations."
    }

    return (
        causes.get(anomaly_type, "The readings show an unusual condition."),
        actions.get(severity, "Continue monitoring the station.")
    )

def gemini_analysis(temperature, humidity, pressure, wind_speed, wind_direction,
                    anomaly_type, severity, score):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or genai is None:
        return local_explanation(anomaly_type, severity)

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are assisting a weather-station anomaly monitoring system.

The ML detector has already detected an anomaly. Do not decide whether it is an anomaly yourself.
Explain the detected result in simple language.

Temperature: {temperature} °C
Relative Humidity: {humidity} %
Pressure: {pressure} mbar
Wind Speed: {wind_speed} m/s
Wind Direction: {wind_direction}°
Anomaly Type: {anomaly_type}
Severity: {severity}
Isolation Forest score: {score:.4f}

Return exactly two short sections:
Possible Cause:
Recommended Action:

Mention that the cause is a possible explanation, not a confirmed diagnosis.
"""

        for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                text = response.text.strip()
                if text:
                    cause = text
                    action = "Follow the recommended action generated above and verify the sensor physically."
                    return cause, action
            except Exception:
                time.sleep(1)

    except Exception:
        pass

    return local_explanation(anomaly_type, severity)

def detect_anomaly(temperature, humidity, pressure, wind_speed, wind_direction, hour, month):
    data = pd.DataFrame([{
        "T (degC)": temperature,
        "rh (%)": humidity,
        "p (mbar)": pressure,
        "wv (m/s)": wind_speed,
        "wd (deg)": wind_direction,
        "hour": hour,
        "month": month,
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

    X_scaled = scaler.transform(data[model_features])

    prediction = model.predict(X_scaled)[0]
    anomaly_score = model.decision_function(X_scaled)[0]

    extreme_sensor_condition = (
        temperature >= 50 or temperature <= -20
        or humidity <= 10
        or pressure >= 1080 or pressure <= 950
        or wind_speed >= 30
        or wind_direction < 0 or wind_direction > 360
    )

    is_anomaly = prediction == -1 or extreme_sensor_condition

    if is_anomaly:
        anomaly_type = identify_anomaly_type(
            temperature, humidity, pressure, wind_speed, wind_direction
        )
        severity = get_severity(
            anomaly_score, temperature, humidity, pressure, wind_speed
        )
    else:
        anomaly_type = "Normal Reading"
        severity = "Normal"

    return is_anomaly, anomaly_score, anomaly_type, severity

st.title("🌦️ SkyGuard AI")
st.subheader("AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations")

st.write(
    "Isolation Forest detects unusual weather-station readings, while Gemini "
    "provides explainable insights and recommended actions."
)

with st.sidebar:
    st.header("Weather Station Input")

    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    pressure = st.number_input("Atmospheric Pressure (mbar)", value=1015.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=2.5)
    wind_direction = st.number_input("Wind Direction (°)", value=180.0)
    hour = st.slider("Hour", 0, 23, 12)
    month = st.slider("Month", 1, 12, 9)

    analyze = st.button("Analyze Reading", use_container_width=True)

if analyze:
    is_anomaly, score, anomaly_type, severity = detect_anomaly(
        temperature, humidity, pressure, wind_speed, wind_direction, hour, month
    )

    if is_anomaly:
        st.error("🚨 ANOMALY DETECTED")
        st.metric("Anomaly Type", anomaly_type)
        st.metric("Severity", severity)

        st.write(f"**Anomaly Score:** `{score:.4f}`")
        st.caption("The Isolation Forest score is a relative anomaly score; lower values indicate more unusual observations.")

        with st.spinner("Generating AI explanation..."):
            possible_cause, recommended_action = gemini_analysis(
                temperature, humidity, pressure, wind_speed,
                wind_direction, anomaly_type, severity, score
            )

        st.subheader("Possible Cause")
        st.write(possible_cause)

        st.subheader("Recommended Action")
        st.write(recommended_action)

    else:
        st.success("✅ NORMAL READING")
        st.metric("Status", "Normal")
        st.write(f"**Anomaly Score:** `{score:.4f}`")
        st.write("The current reading was not flagged as anomalous by the ML detector.")

st.divider()

st.header("Historical Weather Analysis")

if os.path.exists(HISTORICAL_PATH):
    historical_df = pd.read_csv(HISTORICAL_PATH)

    col1, col2, col3 = st.columns(3)

    total_readings = len(historical_df)
    anomaly_count = len(historical_df[historical_df["prediction"] == -1]) if "prediction" in historical_df.columns else len(historical_df)
    anomaly_percentage = (anomaly_count / total_readings * 100) if total_readings else 0

    col1.metric("Historical Readings", f"{total_readings:,}")
    col2.metric("Detected Anomalies", f"{anomaly_count:,}")
    col3.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

    if "Date Time" in historical_df.columns:
        historical_df["Date Time"] = pd.to_datetime(historical_df["Date Time"], errors="coerce")

    if "T (degC)" in historical_df.columns:
        st.subheader("Temperature Trend")
        st.line_chart(historical_df.set_index("Date Time")["T (degC)"])

    if "T (degC)" in historical_df.columns and "prediction" in historical_df.columns:
        st.subheader("Historical Anomaly Points")
        chart_df = historical_df.set_index("Date Time")[["T (degC)"]].copy()
        chart_df["Anomaly"] = np.where(
            historical_df.set_index("Date Time")["prediction"] == -1,
            chart_df["T (degC)"],
            np.nan
        )
        st.line_chart(chart_df)

    if "rh (%)" in historical_df.columns:
        st.subheader("Humidity Trend")
        st.line_chart(historical_df.set_index("Date Time")["rh (%)"])

    st.subheader("Detected Anomalies")
    display_df = historical_df.copy()

    if "prediction" in display_df.columns:
        display_df = display_df[display_df["prediction"] == -1]

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "Download Historical Anomaly Report",
        data=historical_df.to_csv(index=False).encode("utf-8"),
        file_name="historical_weather_anomalies.csv",
        mime="text/csv"
    )
else:
    st.info("Historical anomaly CSV was not found. Add it to the data/ folder.")

st.divider()

st.header("System Architecture")
st.code(
    "Weather Data → Feature Processing → Isolation Forest ML Detector → "
    "Normal/Anomaly → Sensor Safety Check → Anomaly Type & Severity → "
    "Gemini Explanation/Recommendation → Streamlit Dashboard"
)

st.header("System Status")
c1, c2, c3 = st.columns(3)
c1.success("ML Detector: Isolation Forest")
c2.success("Historical Analysis: Active")
c3.success("AI Assistant: Gemini")
