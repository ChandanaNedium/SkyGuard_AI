# SkyGuard AI

## AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations

SkyGuard AI is a prototype weather-station monitoring system that uses **Isolation Forest** to detect unusual multivariate sensor readings and **Gemini** to provide explainable insights and recommended actions.

### Key Features

- ML-based anomaly detection using Isolation Forest
- Temperature, humidity, pressure, wind speed and wind direction monitoring
- Feature engineering using time features, rolling statistics and sensor changes
- Anomaly type identification
- Heuristic severity classification
- Gemini-powered explanation and recommendations
- Historical anomaly analysis
- Temperature and humidity trend visualization
- Downloadable anomaly report
- Streamlit-based interactive dashboard

## System Architecture

```text
Weather Data
      ↓
Feature Processing
      ↓
Isolation Forest ML Detector
      ↓
Normal / Anomaly
      ↓
Sensor Safety Check
      ↓
Anomaly Type + Severity
      ↓
Gemini Explanation / Recommendation
      ↓
Streamlit Dashboard
```

## Important Design Principle

The **ML model detects anomalies**.

Gemini is used only for **explanation and recommendations**. It is not the anomaly detector.

## Project Structure

```text
SkyGuard_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── weather_anomaly_model.pkl
│   ├── weather_scaler.pkl
│   └── model_features.pkl
├── data/
│   └── historical_weather_anomalies.csv
└── screenshots/
```

## Dataset

The prototype was developed using the **BGC Jena Weather Station Dataset (2017–2024)** available through Kaggle.

The dataset is a German weather-station dataset and is used as a prototype/historical data source. It should not be described as Indian IMD data.

## ML Model

The anomaly detector is an **Isolation Forest** model.

The saved model uses weather and temporal features including:

- Temperature
- Relative humidity
- Atmospheric pressure
- Wind speed
- Wind direction
- Hour
- Month
- Rolling temperature statistics
- Rolling humidity statistics
- Rolling pressure statistics
- Temperature change
- Humidity change
- Pressure change

The saved model artifacts are stored in the `model/` directory.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/ChandanaNedium/SkyGuard_AI.git
cd SkyGuard_AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini

Set the `GEMINI_API_KEY` environment variable.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Linux/macOS:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Do not hardcode the API key in `app.py` or commit it to GitHub.

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in the browser.

## Prototype Limitations

- The live single-reading prototype uses simplified rolling/change features.
- Historical analysis uses engineered historical features.
- Severity thresholds are heuristic and are not calibrated confidence probabilities.
- The Isolation Forest decision score is a relative anomaly score, not an anomaly probability.
- Gemini explanations are advisory and should not be treated as confirmed sensor diagnoses.
- The prototype dataset is from Germany rather than an Indian IMD AWS deployment.

## Future Improvements

- Connect the system directly to real-time AWS/IoT sensor streams.
- Use a rolling window of recent readings for live feature generation.
- Add sensor degradation and maintenance prediction.
- Add spatial/multivariate consistency checks across nearby stations.
- Calibrate anomaly confidence and severity using labelled operational data.
- Deploy the application on a persistent cloud platform.

## Project Purpose

This project was developed as a practical ML + GenAI prototype for intelligent weather-station monitoring, anomaly detection, explainability and operational support.
