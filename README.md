# FinIntel: UPI Macro Intelligence Platform 📊🤖

FinIntel is an enterprise-grade data intelligence, time-series forecasting, and anomaly detection platform designed to analyze and predict Indian Unified Payments Interface (UPI) transaction flows. By scraping macro-level transaction statistics from the official National Payments Corporation of India (NPCI) portal and ingesting localized merchant/peer transaction structures from the PhonePe Pulse dataset, FinIntel implements a multi-stage **Medallion Lakehouse Architecture** to feed advanced deep learning forecasting models ([N-BEATSx](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/train_baseline.py) and [Temporal Fusion Transformer](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/train_tft.py)). It automatically flags and alerts volume anomalies using residuals-based analysis, presenting all insights in a secure, interactive web dashboard.

---

## 🌟 Key Features

* **🧠 Advanced Time-Series Forecasting**: Out-of-the-box forecasting models using state-of-the-art neural architectures ([N-BEATSx](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/train_baseline.py) and [Temporal Fusion Transformer](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/train_tft.py)) via the `neuralforecast` library, as well as a simulated long-term Prophet model.
* **📐 Medallion Data Pipeline**: Structured data lake progression from Raw CSV/JSON files ➔ Bronze schema preservation ➔ Silver cleaned & feature-engineered parquet tables ➔ Gold unified single-source-of-truth.
* **🛠️ Contextual Feature Engineering**: Incorporates monthly lag values, rolling statistics, and calendar attributes including day of week, day of month, month, salary windows (1st–7th), and custom Indian holidays (Republic Day, Diwali, Holi, Independence Day).
* **⚠️ Residual-Based Anomaly Detection**: Monitors prediction deviations ($y - NBEATSx$) and automatically isolates anomalies (Spikes, Drops, Seasonal Outliers) exceeding a standard-deviation threshold.
* **🔒 Secure FastAPI Gateway**: Protects data assets (actuals, forecasts, anomalies) behind basic HTTP Authentication and provides high-performance JSON endpoints.
* **📈 Interactive Streamlit Dashboard**: Offers user access control, interactive Plotly visualizations with confidence bands (95%-105%), STL Decomposition components (Trend, Seasonal, Residual), and automated natural language insights.
* **⏰ Automated Retraining & Alerting**: Scheduler system retrains models periodically and pushes real-time email alerts (via SMTP) if a transaction anomaly is detected.

---

## 🏗️ System Architecture

FinIntel is designed around decoupled data processing, modeling, serving, and visualization layers.

```mermaid
graph TD
    %% Styling Configuration
    classDef raw fill:#3B82F6,stroke:#1D4ED8,stroke-width:2px,color:#fff;
    classDef bronze fill:#9A3412,stroke:#7C2D12,stroke-width:2px,color:#fff;
    classDef silver fill:#64748B,stroke:#475569,stroke-width:2px,color:#fff;
    classDef gold fill:#D97706,stroke:#B45309,stroke-width:2px,color:#fff;
    classDef service fill:#0F766E,stroke:#09504a,stroke-width:2px,color:#fff;
    classDef ml fill:#7C3AED,stroke:#6D28D9,stroke-width:2px,color:#fff;
    classDef client fill:#0369A1,stroke:#0284c7,stroke-width:2px,color:#fff;

    %% Data Sources
    subgraph Data Sources (Raw)
        NPCIWeb["🌐 NPCI Product Stats Website"]:::raw
        PhonePeRepo["📂 PhonePe Pulse State JSONs"]:::raw
    end

    %% Ingestion Layer
    subgraph Ingestion & Transformation (Medallion)
        Ingest["📥 Data Ingestors<br/>(ingest_npci.py / ingest_phonepe.py)"]:::service
        Bronze["🟫 Bronze Parquet Tables<br/>(npci_stats.parquet / phonepe_txn.parquet)"]:::bronze
        Transform["⚙️ Transformers<br/>(transform.py / transform_npci.py)"]:::service
        Silver["🥈 Silver Parquet Tables<br/>(npci_stats_silver.parquet / phonepe_txn_silver.parquet)"]:::silver
        Features["🛠️ Feature Builder<br/>(feature_builder.py / holiday_calendar.py)"]:::service
        SilverFeat["🥈 Silver Features Table<br/>(npci_features.parquet)"]:::silver
        Merge["🔗 Merger<br/>(merge_silver.py)"]:::service
        Gold["🥇 Gold Parquet Table<br/>(upi_macro_gold.parquet)"]:::gold
    end

    NPCIWeb -->|Scraped by src/data/ingest_npci.py| Ingest
    PhonePeRepo -->|Loaded by src/data/ingest_phonepe.py| Ingest
    Ingest --> Bronze
    Bronze --> Transform
    Transform --> Silver
    Silver --> Features
    Features --> SilverFeat
    Silver --> Merge
    Merge --> Gold

    %% Machine Learning Layer
    subgraph Machine Learning & Modeling
        TrainBaseline["💜 Train N-BEATSx<br/>(train_baseline.py)"]:::ml
        TrainTFT["💜 Train TFT<br/>(train_tft.py)"]:::ml
        ProphetSim["🧡 Future Simulation<br/>(generate_future_forecast.py)"]:::ml
        AnomalyDetect["⚠️ Anomaly Detector<br/>(run_anomaly_detection.py)"]:::ml
        
        BacktestNBEATS["📄 NBEATSx Backtest CSV<br/>(upi_forecast_backtest.csv)"]:::gold
        BacktestTFT["📄 TFT Backtest CSV<br/>(upi_forecast_backtest_tft.csv)"]:::gold
        FutureForecast["📄 Simulated Future CSV<br/>(future_forecast.csv)"]:::gold
        AnomaliesCSV["⚠️ Flagged Anomalies CSV<br/>(upi_anomalies.csv)"]:::gold
    end

    Gold --> TrainBaseline
    Gold --> TrainTFT
    Gold --> AnomalyDetect
    Gold --> ProphetSim

    TrainBaseline --> BacktestNBEATS
    TrainTFT --> BacktestTFT
    ProphetSim --> FutureForecast
    AnomalyDetect --> AnomaliesCSV
    BacktestNBEATS --> AnomalyDetect

    %% Services Layer
    subgraph Services & Presentation
        API["🚪 FastAPI Service<br/>(src/api/main.py:8000)"]:::service
        Dashboard["📊 Streamlit Application<br/>(src/dashboard/app.py:8501)"]:::client
        Scheduler["⏰ Scheduler Daemon<br/>(schedule_retrain.py / retrain_models.py)"]:::service
        EmailAlert["✉️ SMTP Email Server"]:::service
    end

    Gold -.-> API
    BacktestNBEATS -.-> API
    BacktestTFT -.-> API
    AnomaliesCSV -.-> API
    
    API ==>|Secure HTTP REST| Dashboard
    FutureForecast -.->|Local CSV Load| Dashboard
    
    Scheduler -->|Runs 24h Pipeline| TrainBaseline
    Scheduler -->|Runs 24h Pipeline| TrainTFT
    Scheduler -->|Runs 24h Pipeline| AnomalyDetect
    
    AnomalyDetect -.->|Triggers Alert| EmailAlert
```

---

## 🛠️ Technology Stack

### Data Processing & Engineering
* **Languages & Runtimes**: Python 3.10
* **Storage Engines**: Apache Parquet format (using `pyarrow` engine)
* **Libraries**: `pandas`, `numpy`, `requests`, `lxml` (for HTML tables parsing)

### Machine Learning & Modeling
* **Time-Series Framework**: `neuralforecast` (utilizing deep learning estimators)
* **Underlying Engines**: PyTorch & PyTorch Lightning
* **Traditional Regressors**: `prophet` (meta/facebook forecasting package), `scikit-learn` (Linear Regression baseline)
* **Serialization**: `joblib` (for scikit-learn pipeline saving)

### Serving (API Gateway)
* **Framework**: `fastapi` (high performance HTTP framework)
* **Server Runtime**: `uvicorn` (ASGI server implementation)
* **Security**: Basic HTTP Authentication

### Visualization & Dashboard
* **Framework**: `streamlit` (web dashboard compiler)
* **Charting Engine**: `plotly.graph_objects` (for dynamic overlays and confidence bands)
* **Analysis Toolkit**: `statsmodels.tsa.seasonal.STL` (for Loess seasonal-trend decomposition)
* **Report Compiler**: `fpdf` (generates natural-language PDF export bundles)

---

## 📂 Project Directory Structure

```text
upi-macro-intel/
│
├── configs/
│   └── data_sources.yaml            # Data ingestion source definitions (placeholder)
│
├── data/                            # Medallion Architecture Data Lake
│   ├── raw/
│   │   ├── npci/                    # Raw scraped NPCI website statistics
│   │   └── phonepe_pulse/           # Raw state-level JSON structures from PhonePe Pulse
│   ├── bronze/                      # Schema-preserved parquet formats
│   ├── silver/                      # Cleaned formats and engineered feature tables
│   └── gold/                        # Unified parquet files and backtesting prediction tables
│
├── models/                          # Trained traditional model checkpoints (.pkl)
├── notebooks/                       # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── output/                          # Execution reports and export dumps
├── lightning_logs/                  # PyTorch Lightning execution and tensorboard logs
│
├── src/                             # Source Code Base
│   ├── api/
│   │   ├── Dockerfile               # API Service Container configuration
│   │   └── main.py                  # FastAPI Router and database interface
│   │
│   ├── dashboard/
│   │   ├── Dockerfile               # Streamlit Dashboard Container configuration
│   │   ├── app.py                   # Streamlit Visualization and UI controller
│   │   └── hashed_password.py       # Helper utility to generate bcrypt password hashes
│   │
│   ├── data/                        # Active raw web scraping ingestors
│   │   ├── ingest_npci.py           # Directly downloads & parses fresh statistics from NPCI
│   │   └── ingest_phonepe.py        # Walks and flattens regional PhonePe directory trees
│   │
│   ├── etl/                         # Medallion transformation scripts
│   │   ├── ingest_npci.py           # Ingests raw NPCI CSV into Bronze parquet
│   │   ├── ingest_phonepe.py        # Flattens country state transaction directories to Bronze
│   │   ├── merge_silver.py          # Aggregates PhonePe, joins with NPCI to produce Gold Parquet
│   │   ├── transform.py             # Transforms Bronze PhonePe data & parses NPCI CSV into Parquet
│   │   ├── transform_npci.py        # Standalone parser for raw NPCI dates
│   │   └── verify_phonepe.py        # Verifies integrity of PhonePe columns
│   │
│   ├── features/                    # Feature Engineering
│   │   ├── feature_builder.py       # Computes monthly lags and rolling stats
│   │   └── holiday_calendar.py      # Holiday flags and salary window feature builder
│   │
│   ├── ml/                          # Training automation scripts
│   │   └── retrain_models.py        # Automatic LR retrainer, anomaly flagger, and alerter
│   │
│   ├── models/                      # Forecast Modeling scripts
│   │   ├── anomaly_detection.py     # Standalone residual anomaly identifier (Actual vs Forecast)
│   │   ├── backtest.py              # Rolling-origin backtest validation using lag baselines
│   │   ├── compare_forecasts.py     # Matplotlib forecast evaluation plotter
│   │   ├── evaluate_forecast.py     # Visualizer for prediction residuals
│   │   ├── forecast_baseline.py     # Fits and outputs 6m forecasts to upi_forecast.csv
│   │   ├── forecast_baseline_backtest.py # N-BEATSx evaluation simulator
│   │   ├── forecast_full.py         # Out-of-sample forecast writer (6m forecast beyond actuals)
│   │   ├── run_anomaly_detection.py # Aggregated pipeline executor for anomaly mapping
│   │   ├── save_anomalies.py        # Exports flagged anomalies to gold directory
│   │   ├── train_baseline.py        # Core backtest runner for N-BEATSx models
│   │   └── train_tft.py             # Core backtest runner for TFT models
│   │
│   ├── utils/                       # Common utilities
│   │   ├── alerting.py              # SMTP email alert configuration
│   │   └── schedule_retrain.py      # Background loop to trigger model retraining every 24h
│   │
│   └── train_pipeline.py            # Orchestrator to execute training & anomaly pipelines in series
│
├── export_actuals.py                # Exports historical gold data from API to local CSV
├── generate_future_forecast.py      # Fits Prophet model and simulates 5-year future predictions
├── train_nbeats.py                  # PyTorch Lightning NBeats training script
├── run_training_pipeline.py         # Subprocess runner log wrap for train_pipeline.py
├── run_all.py                       # Unified command-line runner (starts API, ETL, & Streamlit)
├── run_both.py                      # Starts FastAPI, Streamlit Dashboard, and Scheduler
├── start.bat                        # Windows bootstrapper (activates venv and runs run_all.py)
├── docker-compose.yml               # Multi-container service orchestrator
└── .gitignore
```

---

## 📐 Medallion Data Pipeline Deep Dive

FinIntel implements an automated Medallion data engineering flow:

1. **Ingestion Layer (Scrape to Raw)**:
   * [src/data/ingest_npci.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/data/ingest_npci.py) uses `requests` and `pandas.read_html` to fetch the live transaction tables from the NPCI Product Statistics page. It saves them as `data/raw/npci/upi_product_stats.csv`.
   * [src/data/ingest_phonepe.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/data/ingest_phonepe.py) traverses state-level directories nested under `data/raw/phonepe_pulse/` to parse annual and quarterly JSONs, aggregating state-wise volumes and value metrics.
2. **Bronze Layer (Raw to Bronze)**:
   * [src/etl/ingest_npci.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/etl/ingest_npci.py) reads the raw NPCI CSV, cleans up whitespaces, removes numeric formatting commas, and writes it to `data/bronze/npci_stats.parquet`.
   * [src/etl/ingest_phonepe.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/etl/ingest_phonepe.py) flattens nested state transaction metrics into a structural parquet table at `data/bronze/phonepe_transactions.parquet`.
3. **Silver Layer (Bronze to Silver & Features)**:
   * [src/etl/transform.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/etl/transform.py) explodes PhonePe payment instruments, standardizes column headers, maps integer quarters to dates (e.g. Q1 ➔ January, Q2 ➔ April), and saves clean tables to `data/silver/phonepe_txn_silver.parquet`.
   * [src/features/feature_builder.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/features/feature_builder.py) acts on NPCI stats to engineer statistical features:
     * **Lags**: 1-month, 3-month, 6-month, and 12-month transaction volume shifts.
     * **Rolling Statistics**: Mean and Standard Deviation trends computed across 3, 6, and 12-month windows.
     * **Calendar Features**: Handled by [src/features/holiday_calendar.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/features/holiday_calendar.py) to flag days of the week, days of the month, month integers, end-of-month dates, salary dispatch windows (days 1 to 7), and key Indian holidays.
4. **Gold Layer (Silver to Gold)**:
   * [src/etl/merge_silver.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/etl/merge_silver.py) groups silver PhonePe transaction volumes by month (aggregating all regional data to a national monthly metric) and performs an inner join with NPCI silver statistics by month. This blends overall UPI macro-activity with PhonePe's transaction distribution, saving the final training table to `data/gold/upi_macro_gold.parquet`.

---

## 🔮 Time-Series Modeling & Anomaly Detection

### Deep Learning Models
* **N-BEATSx**: Implemented in [src/models/train_baseline.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/train_baseline.py). Uses an input window of 12 months to predict a horizon of 6 months. It isolates data before `2024-07-01` for training, and uses the remaining data for validation/backtesting, exporting results to `data/gold/upi_forecast_backtest.csv`.
* **Temporal Fusion Transformer (TFT)**: Implemented in [src/models/train_tft.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/train_tft.py). Follows the same configuration as N-BEATSx but applies self-attention mechanisms to isolate seasonal patterns, exporting results to `data/gold/upi_forecast_backtest_tft.csv`.

### Simulation & Baselines
* **Prophet Model**: Contained in [generate_future_forecast.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/generate_future_forecast.py). Fits a trend model with yearly seasonal Fourier terms on exported actuals to project transaction volume 5 years (60 months) into the future, writing predictions to `future_forecast.csv` for Streamlit visualization.
* **Retraining Pipeline**: [src/ml/retrain_models.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/ml/retrain_models.py) implements a lightweight Linear Regression training cycle as an automated fallback, outputting forecasts and recalculating residual bounds.

### Anomaly Flagging
* **Residual Analysis**: Evaluates forecasting residuals ($e_t = y_t - \hat{y}_t$). Points where the residual absolute value exceeds a standard deviation multiplier (e.g. $1.5\sigma$ in [src/models/run_anomaly_detection.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/models/run_anomaly_detection.py) or $3.0\sigma$ in [src/ml/retrain_models.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/ml/retrain_models.py)) are categorized as anomalies.
* **SMTP Alerts**: If anomalies are flagged, the scheduler triggers [src/utils/alerting.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/utils/alerting.py) to email a tabular summary to the administrator.

---

## 🚪 API Gateway Reference

The API is served using FastAPI, protected by HTTP Basic Authentication. Authenticated requests require basic credentials mapping to `VALID_USERS` (e.g. `sushrut` / `sushrutpass` or `admin` / `adminpass`).

### Endpoints

* **`GET /`**: Welcome message and status check.
* **`GET /health`**: Health status check returning `{"status": "UP"}`.
* **`GET /actuals`**: Retrieves historical monthly actual data from the Gold Parquet layer.
  * *Query Parameters*: `start_date` (optional), `end_date` (optional).
* **`GET /forecast`**: Retrieves backtest predictions for a selected model.
  * *Query Parameters*: `model` (required: `"NBEATSx"` or `"TFT"`), `start_date` (optional), `end_date` (optional).
* **`GET /anomalies`**: Retrieves flagged transaction anomalies.
  * *Query Parameters*: `start_date` (optional), `end_date` (optional).

---

## 📊 Streamlit Dashboard Tour

The dashboard ([src/dashboard/app.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/src/dashboard/app.py)) provides a visual interface for data analysis:

1. **User Authentication**: A login form verifying credentials against `USER_CREDENTIALS` (supports roles: `admin`, `analyst`, `guest`).
2. **Status Panel**: Displays live API Health and the status of the background scheduler.
3. **Control Filters**: Interactive selectors to alter Date Ranges, anomaly types (Spike, Drop, Seasonal Outlier), and models (NBEATSx, TFT, LinearRegression).
4. **Interactive Plots**: Overlay charts presenting historical actuals vs forecasts, decorated with 95%–105% confidence shading and anomaly markers.
5. **Dynamic Metrics**:
   * Out-of-sample performance matrices (MAE and MAPE) evaluated on the fly.
   * Macro metrics highlighting total data points, forecast length, and average errors.
6. **Time Series STL Decomposition**: Decomposes the actual transaction trend, seasonality, and residual noise components using Loess regression.
7. **Automated Summary Insights**: Auto-generates written reports explaining month-over-month growth, volume ranges, model errors, and recent deviations.
8. **Export Center**: Download buttons to fetch underlying actuals/forecast CSVs, Plotly charts as PNG, or a compiled PDF Summary Report.
9. **Future Projections**: Displays the 5-year Prophet simulation (`future_forecast.csv`) with a vertical divider marking the transition into predictions.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10
* Virtualenv utility (`python -m venv`)
* SMTP credentials (for email notifications)

### Installation

1. **Clone & Initialize Project**:
   ```powershell
   cd upi-macro-intel
   ```

2. **Create and Activate Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Note: Ensure PyTorch, PyTorch Lightning, NeuralForecast, Prophet, Streamlit, and FastAPI dependencies are configured in your env.)*

### Environment Configuration

Configure the following environment variables if enabling SMTP anomaly alerts:
```env
ALERT_EMAIL=bankalgisushrut@gmail.com
ALERT_EMAIL_PASSWORD=your_gmail_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### Running the Platform

There are three ways to launch the services:

#### 1. Unified Shell Script (Windows Bootstrapper)
Double-click [start.bat](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/start.bat) or run it from the command line:
```powershell
.\start.bat
```
This activates the local virtual environment and executes [run_all.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/run_all.py).

> [!IMPORTANT]
> **Orchestrator Path Notice**:
> The orchestrator [run_all.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/run_all.py) is configured to start the scheduler process using:
> `python src/utils/schedule.py`
> If this script fails to find the scheduler daemon, update line 31 of `run_all.py` to point to the correct scheduler script:
> `python src/utils/schedule_retrain.py`
> Alternatively, you can use the secondary launcher script [run_both.py](file:///c:/Sushrut/CODING/Python/ML/upi-macro-intel/run_both.py) which has the correct path already wired in.

#### 2. Secondary Platform Launcher
Run the secondary launcher to boot the API, the Streamlit app, and the retraining scheduler daemon simultaneously:
```powershell
python run_both.py
```

#### 3. Containerized Orchestration (Docker)
Ensure Docker Desktop is running, then boot the containerized microservices:
```powershell
docker-compose up --build
```
This builds and spawns:
* **API Gateway**: Exposed at `http://localhost:8000`
* **Streamlit Dashboard**: Exposed at `http://localhost:8501`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to file issues or submit pull requests.

## 📄 License
This project is proprietary and confidential. Unauthorized distribution or copying is strictly prohibited.