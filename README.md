# 🌦️ Weather Prediction System Using Machine Learning

An end-to-end Machine Learning and Weather Forecasting web application built with **Python**, **Scikit-Learn**, **Streamlit**, **Plotly**, and the **Open-Meteo Global Weather API**.

This project provides two distinct forecasting approaches:
1. **Machine Learning Engine:** Uses Random Forest models trained on 25 years (2000–2024) of historical meteorological data across 10 major Indian cities to predict next-day rainfall occurrence, rain probability, and maximum daily temperature.
2. **Live Numerical Weather Prediction (NWP) Engine:** Connects to the real-time Open-Meteo API to retrieve live atmospheric conditions and 7-day operational forecasts.
3. **Side-by-Side Comparison Mode:** Compares ML model predictions against live physics-based NWP forecasts using real current-day observations, with detailed explanations of statistical learning vs atmospheric simulation.

---

## 📋 Table of Contents
1. [Project Objectives & Architecture](#-project-objectives--architecture)
2. [Dataset & Cities](#-dataset--cities)
3. [Machine Learning Models & Features](#-machine-learning-models--features)
4. [Project Structure](#-project-structure)
5. [Step-by-Step Setup & Installation](#-step-by-step-setup--installation)
6. [How to Run the Application](#-how-to-run-the-application)
7. [Testing Each Feature & Mode](#-testing-each-feature--mode)
8. [Model Evaluation & Baseline Benchmarks](#-model-evaluation--baseline-benchmarks)
9. [Key Limitations & Meteorological Context](#-key-limitations--meteorological-context)
10. [How to Retrain the Models](#-how-to-retrain-the-models)
11. [Free Cloud Deployment (Streamlit Community Cloud)](#-free-cloud-deployment-streamlit-community-cloud)
12. [Code Walkthrough & Function Explanations](#-code-walkthrough--function-explanations)

---

## 🎯 Project Objectives & Architecture

The system achieves three primary goals:
- **Rainfall Prediction:** Classifies whether it will rain tomorrow (`Rain` / `No Rain`) and estimates calibrated probability (0% to 100%).
- **Temperature Forecasting:** Predicts tomorrow's maximum temperature in degrees Celsius (°C).
- **Comparative Analysis:** Bridges historical data-driven ML with real-time operational meteorology while preventing data leakage and clarifying model boundaries.

```
┌─────────────────────────────────┐           ┌────────────────────────────────┐
│   Kaggle Historical Dataset     │           │      Open-Meteo API (Live)     │
│   (india_2000_2024_daily)       │           │   (Global ECMWF / GFS NWP)     │
└────────────────┬────────────────┘           └───────────────┬────────────────┘
                 │                                            │
                 ▼                                            ▼
       [data_loader.py]                                [api_client.py]
     • Chronological sorting                         • Real-time measurements
     • Lag features (t-1)                            • Today & Yesterday weather
     • Zero future leakage                           • Multi-day NWP forecast
                 │                                            │
                 ▼                                            ▼
       [predict.py] ◄─────────────────────────────────────────┘
     • Validates 8 exact features
     • Rainfall Classifier (Random Forest)
     • Temperature Regressor (Random Forest)
                 │
                 ▼
       [Streamlit Web App (app.py)]
     ├── Mode 1: Machine Learning Historical Mode
     ├── Mode 2: Live Weather Forecast Mode
     ├── Mode 3: Side-by-Side Comparison Mode
     ├── Mode 4: Model Evaluation & Benchmarks
     └── Mode 5: Architecture & Beginner Guide
```

---

## 📍 Dataset & Cities

The dataset (`data/india_2000_2024_daily_weather.csv`) contains **91,320 daily records** spanning **January 1, 2000 to December 31, 2024** for 10 major Indian metropolitan areas:

| City | State / Region | Climate Profile | Latitude | Longitude |
| :--- | :--- | :--- | :--- | :--- |
| **Delhi** | National Capital Region | Subtropical steppe / extreme seasons | 28.6139°N | 77.2090°E |
| **Mumbai** | Maharashtra (Konkan Coast) | Tropical wet & dry / heavy monsoon | 19.0760°N | 72.8777°E |
| **Kolkata** | West Bengal (East India) | Tropical wet & dry / Bay of Bengal | 22.5726°N | 88.3639°E |
| **Chennai** | Tamil Nadu (Coromandel) | Tropical wet & dry / Northeast monsoon | 13.0827°N | 80.2707°E |
| **Bangalore**| Karnataka (Deccan Plateau) | Moderate tropical savanna / high elevation | 12.9716°N | 77.5946°E |
| **Hyderabad**| Telangana (Central Deccan) | Tropical wet & dry / semi-arid | 17.3850°N | 78.4867°E |
| **Ahmedabad**| Gujarat (Western India) | Hot semi-arid / high summer heat | 23.0225°N | 72.5714°E |
| **Pune** | Maharashtra (Western Ghats) | Semi-arid rain shadow / mild winters | 18.5204°N | 73.8567°E |
| **Jaipur** | Rajasthan (North-West) | Hot semi-arid near Thar Desert | 26.9124°N | 75.7873°E |
| **Lucknow** | Uttar Pradesh (Gangetic Plain)| Humid subtropical / cold winters, hot monsoons| 26.8467°N | 80.9462°E |

---

## 🧠 Machine Learning Models & Features

Both models were originally trained using **Random Forest** algorithms and stored as dictionaries with `model` and `features` keys.

### Trained Model Files
- `weather_models/rainfall_model.pkl`: `RandomForestClassifier` (predicts Rain = 1 or No Rain = 0, plus probability).
- `weather_models/temperature_model.pkl`: `RandomForestRegressor` (predicts next-day maximum temperature in °C).

### The 8 Required Inference Features
To run inference, the models require exactly these 8 features in this exact order:

1. `temperature_2m_max`: Maximum temperature recorded today (°C).
2. `temperature_2m_min`: Minimum temperature recorded today (°C).
3. `rain_sum`: Total rainfall recorded today (mm).
4. `wind_speed_10m_max`: Maximum wind speed today (km/h).
5. `previous_rain`: Total rainfall recorded yesterday (mm).
6. `previous_max_temp`: Maximum temperature recorded yesterday (°C).
7. `month`: Current calendar month (1 to 12).
8. `day_of_year`: Current day of the year (1 to 366).

### Leakage-Free Design
- **No Future Information:** Models strictly use information available at observation time $t$ (including lag $t-1$) to predict targets at $t+1$.
- **City Group Isolation:** Lag variables are grouped by city so that data from one city never spills over into another.
- **Chronological Split:** Evaluated on future holdout periods (2023–2024) rather than random shuffle cross-validation.

---

## 📁 Project Structure

```
weather_prediction/
├── app.py                  # Main Streamlit web application & modern weather UI
├── predict.py              # Model inference wrapper, validation, and probability calculation
├── data_loader.py          # Data ingestion, lag feature calculation, and city coordinates
├── api_client.py           # Open-Meteo API client for real-time weather & NWP forecasts
├── evaluation.py           # Chronological holdout evaluation & baseline benchmarking
├── train_models.py         # Reproducible script to retrain Random Forest models
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Git ignore rules for virtual environments and large binaries
├── README.md               # Complete project documentation & guide
├── data/
│   └── india_2000_2024_daily_weather.csv  # 25-year daily weather records
└── weather_models/
    ├── rainfall_model.pkl                 # Trained Random Forest Classifier
    └── temperature_model.pkl              # Trained Random Forest Regressor
```

---

## 💻 Step-by-Step Setup & Installation

### Step 1: Open Terminal / PowerShell
Navigate to the project root directory:
```bash
cd "C:\Users\ashum\OneDrive\Desktop\weather prediction"
```

### Step 2: Create a Python Virtual Environment
A virtual environment isolates your project packages so they do not conflict with system Python:
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

- **On Windows (Command Prompt `cmd`):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **On macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Required Packages
Install all dependencies using pip:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

With your virtual environment activated, run:

```bash
streamlit run app.py
```

Streamlit will launch the web server and automatically open the application in your default web browser at:
```
http://localhost:8501
```

---

## 🧪 Testing Each Feature & Mode

The web application is organized into 5 dedicated modes via the sidebar:

### 1. Machine Learning Mode (Historical Dataset)
- **Latest Dataset Record Tab:** Select any of the 10 cities. The app immediately loads the latest record in the dataset (**December 31, 2024**), constructs the 8-feature vector, and executes inference:
  - Displays **Rain Tomorrow: Rain / No Rain** with model probability (e.g. 68.4%).
  - Displays **Predicted Maximum Temperature: 22.8 °C**.
- **Historical Date Explorer Tab:** Choose any historical date (e.g., July 15, 2024 during monsoon season). The app displays the model's prediction alongside the **actual ground truth** of what happened the next day, showing the exact prediction error!
- **What-If Simulator Tab:** Adjust temperature, rainfall, wind speed, and month sliders to test how the Random Forest responds to extreme or hypothetical conditions.
- **Interactive Climate Charts:** Inspect 30-day, 90-day, or 1-year dual-axis Plotly charts of daily temperatures and precipitation.

### 2. Live Weather Forecast Mode (Open-Meteo)
- Select any Indian city.
- The app sends a secure, real-time request to the Open-Meteo API.
- Displays live temperature, apparent temperature ("feels like"), humidity, wind speed, and current weather icon (WMO weather code).
- Displays tomorrow's operational NWP forecast (max/min temperature, precipitation sum in mm, and precipitation probability).
- Renders an interactive **7-day outlook chart**.
- **Error Handling Test:** Disconnect your internet connection or simulate a network timeout; the app displays an informative error card without crashing.

### 3. Side-by-Side Comparison Mode (ML vs NWP)
- Automatically pulls **today's and yesterday's live observations** from Open-Meteo.
- Feeds those real-world measurements into the **ML Random Forest models**.
- Places the ML prediction and Open-Meteo's physics-based NWP forecast side-by-side in a comparative delta table.
- Analyzes where the models agree or disagree, with an educational breakdown of statistical data learning vs atmospheric physics simulations.

### 4. Model Performance & Evaluation
- View chronological holdout metrics computed on **7,300 unseen test records (2023–2024)**.
- Compares the ML models against simple **Persistence** and **Climatology** baselines.
- Interactive **Confusion Matrix** and **Feature Importance** charts.

### 5. Architecture & Guide
- Explains the complete pipeline, feature engineering, and answers common beginner questions.

---

## 📊 Model Evaluation & Baseline Benchmarks

To ensure realistic performance estimates, models were evaluated on a **chronological holdout period (January 1, 2023 to December 30, 2024)** consisting of **7,300 test days across all 10 cities**.

### Rainfall Classification Results (Holdout Set)
- **Accuracy:** **80.8%**
- **Precision:** **0.891** (89.1% of rain alarms were correct)
- **Recall:** **0.655**
- **F1-Score:** **0.755**
- **ROC-AUC:** **0.890**

#### Baseline Comparison:
| Model / Baseline | Accuracy | F1-Score | Strategy |
| :--- | :---: | :---: | :--- |
| **Random Forest Classifier** | **80.8%** | **0.755** | Trained on 8 meteorological features |
| **Persistence Baseline** | 85.6% | 0.841 | Predicts tomorrow's rain = today's rain |
| **Majority Class Baseline** | 54.9% | 0.000 | Always predicts "No Rain" (dominant class) |

### Maximum Temperature Regression Results (Holdout Set)
- **Mean Absolute Error (MAE):** **1.09 °C** (predictions are within ~1.1 °C on average)
- **Root Mean Squared Error (RMSE):** **1.46 °C**
- **$R^2$ Score:** **0.907** (explains 90.7% of temperature variance)

#### Baseline Comparison:
| Model / Baseline | MAE (°C) | RMSE (°C) | $R^2$ Score | Strategy |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **1.09** | **1.46** | **0.907** | Trained on 8 meteorological features |
| **Persistence Baseline** | 0.94 | 1.31 | 0.925 | Tomorrow's max temp = Today's max temp |
| **Monthly Climatology Baseline** | 1.85 | 2.43 | 0.743 | City's historical mean temp for current month |

> **Key Machine Learning Insight:** In daily weather, short-term persistence (tomorrow being similar to today) is naturally strong. The Random Forest captures both the short-term inertia and non-linear transitions (such as sudden monsoon onset or cold waves), significantly outperforming seasonal climatology ($R^2$ 0.907 vs 0.743).

You can also run the evaluation from the command line at any time:
```bash
python evaluation.py
```

---

## ⚠️ Key Limitations & Meteorological Context

1. **Dataset Cutoff (December 31, 2024):**
   The historical Kaggle dataset concludes on December 31, 2024. Therefore, historical data alone cannot provide live observations for current dates in 2026. The application clearly indicates this cutoff and uses the Open-Meteo API for live conditions.
2. **Statistical ML vs Atmospheric Physics (NWP):**
   - The Random Forest is a purely **data-driven statistical estimator** trained on local historical patterns.
   - External weather APIs (such as Open-Meteo) run global **Numerical Weather Prediction (NWP)** physics simulations (ECMWF, GFS) that model atmospheric fluid dynamics, cloud microphysics, and satellite data.
   - An NWP forecast is an independent numerical estimate, **not ground truth**. True verification only occurs when actual physical observations are recorded on the following day.
3. **Calibrated Probabilities:**
   Random Forest probability estimates represent tree voting fractions. While highly correlated with rainfall likelihood, they should be interpreted as model confidence estimates rather than operational ensemble percentages.

---

## 🔄 How to Retrain the Models

If you ever wish to retrain the models locally (for example, after updating the CSV dataset or testing different hyperparameters):

```bash
python train_models.py
```

This script:
1. Loads `data/india_2000_2024_daily_weather.csv`.
2. Computes all lag features strictly within city boundaries.
3. Splits data chronologically (2000–2022 Train, 2023–2024 Test).
4. Fits new `RandomForestClassifier` and `RandomForestRegressor` models.
5. Evaluates metrics against holdout data.
6. Overwrites `weather_models/rainfall_model.pkl` and `weather_models/temperature_model.pkl`.

---

## ☁️ Free Cloud Deployment (Streamlit Community Cloud)

You can deploy this application for free on **Streamlit Community Cloud** so anyone can access it online via a public URL:

### Step 1: Push Code to GitHub
1. Create a new repository on [GitHub](https://github.com/) named `weather-prediction-system`.
2. In your terminal, initialize Git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Weather Prediction System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/weather-prediction-system.git
   git push -u origin main
   ```
   *(The included `.gitignore` ensures virtual environment folders are excluded automatically).*

### Step 2: Deploy on Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
2. Click **New app**.
3. Select your repository: `YOUR_USERNAME/weather-prediction-system`.
4. Set the Main file path to: `app.py`.
5. Click **Deploy!**

In a few minutes, your live weather application will be online and accessible from any smartphone, tablet, or PC worldwide.

---

## 🔍 Code Walkthrough & Function Explanations

Here is a simple explanation of each core file and function in the codebase:

### `app.py`
- **`st.set_page_config()`**: Sets the browser tab title, favicon, and responsive layout.
- **`CUSTOM_CSS`**: Custom styling for clean glassmorphism cards, badges, and fonts.
- **`get_cached_dataset()`**: Loads the historical CSV once and caches it in memory using `@st.cache_data` for speed.
- **`get_cached_predictor()`**: Loads the ML models into memory once using `@st.cache_resource`.
- **Mode 1–5 Renderers**: Render the UI tabs, metric cards, Plotly charts, and comparison tables.

### `data_loader.py`
- **`CITIES_METADATA`**: Dictionary of city names, coordinates (latitude, longitude), regions, and climate descriptions.
- **`load_raw_dataset(file_path)`**: Reads the CSV, validates column schema, parses dates, and sorts chronologically.
- **`prepare_engineered_dataset(df)`**: Creates lag features (`previous_rain`, `previous_max_temp`) by shifting within city groups, and generates calendar features (`month`, `day_of_year`).
- **`get_latest_inference_features(df, city)`**: Extracts the most recent valid record for a city and formats the 8-feature DataFrame.
- **`get_historical_record_for_date(df, city, date)`**: Retrieves features for a historical date and pairs it with the actual next day's ground truth for accuracy testing.
- **`get_wmo_info(code)`**: Translates numerical WMO weather codes into human-readable descriptions and emojis (e.g., Code 61 → "Slight rain 🌦️").

### `predict.py`
- **`check_models_exist()`**: Verifies that both pickle files exist on disk before running inference.
- **`WeatherPredictor`**: Encapsulates model unpickling, schema checking, and prediction.
- **`WeatherPredictor.validate_features(df)`**: Confirms that the incoming DataFrame contains the exact 8 features in the correct order, without missing or NaN values.
- **`WeatherPredictor.predict(df)`**: Runs `predict()` and `predict_proba()` to output rainfall label, percentage probability, and forecasted max temperature.

### `api_client.py`
- **`fetch_live_city_weather(city_name)`**: Queries the Open-Meteo REST endpoint for real-time conditions, yesterday's observations, and tomorrow's NWP forecast.
- **`convert_live_weather_to_ml_features(live_data)`**: Adapts live measurements into the 8 input features required by the Random Forest model for Mode 3 comparison.

### `evaluation.py`
- **`create_chronological_split(df, split_date)`**: Splits data chronologically at 2023-01-01 into training and holdout test sets without temporal leakage.
- **`evaluate_models_on_holdout(test_df, train_df, city)`**: Computes Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE, $R^2$, and baseline comparisons.
- **`print_evaluation_report()`**: Formats an evaluation summary in the terminal.

### `train_models.py`
- **`train_and_save_models()`**: Full training pipeline that fits `RandomForestClassifier` and `RandomForestRegressor`, evaluates on holdout data, and saves compressed pickle files.

---

## 📜 License & Acknowledgments
- Historical Weather Dataset: Kaggle Indian Daily Weather (2000–2024).
- Numerical Weather Prediction: Open-Meteo API (Open Database License).
- Machine Learning: Scikit-Learn Random Forest algorithms.
