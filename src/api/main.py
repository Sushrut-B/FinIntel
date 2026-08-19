from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Optional
from datetime import datetime
import logging

app = FastAPI(title="UPI Macro Intelligence API")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for local dashboard testing (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Basic HTTP Auth setup
security = HTTPBasic()

VALID_USERS = {
    "sushrut": "sushrutpass",
    "admin": "adminpass"
}

def auth_required(credentials: HTTPBasicCredentials = Depends(security)):
    correct_pw = VALID_USERS.get(credentials.username)
    if not correct_pw or credentials.password != correct_pw:
        logger.warning(f"Unauthorized access attempt with username: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    logger.info(f"Authorized access by user: {credentials.username}")
    return credentials.username

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Helper functions to load data dynamically (prevents stale data serving)
def load_gold_df():
    try:
        df = pd.read_parquet("data/gold/upi_macro_gold.parquet").rename(columns={"month": "ds", "amount": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    except Exception as e:
        logger.error(f"Error loading gold dataset: {e}")
        return pd.DataFrame()

def load_forecast_df(model_name: str):
    try:
        if model_name.lower() == "nbeatsx":
            file_path = "data/gold/upi_forecast_backtest_nbeats.csv"
        elif model_name.lower() == "tft":
            file_path = "data/gold/upi_forecast_backtest_tft.csv"
        elif model_name.lower() in ("linearregression", "linreg", "gradientboosting", "gb"):
            file_path = "data/gold/upi_forecast_backtest_linreg.csv"
        else:
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    except Exception as e:
        logger.error(f"Error loading forecast for model {model_name}: {e}")
        return pd.DataFrame()

def load_anomalies_df():
    try:
        file_path = "data/gold/upi_anomalies.csv"
        if pd.io.common.file_exists(file_path):
            df = pd.read_csv(file_path)
            df["ds"] = pd.to_datetime(df["ds"])
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading anomalies dataset: {e}")
        return pd.DataFrame()

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the UPI Macro Intelligence API"}

@app.get("/actuals")
def get_actuals(start_date: Optional[str] = None, end_date: Optional[str] = None, user: str = Depends(auth_required)):
    try:
        df = load_gold_df()
        if df.empty:
            return []
        if start_date:
            df = df[df["ds"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["ds"] <= pd.to_datetime(end_date)]
        logger.info(f"Actuals requested by user {user}: start={start_date}, end={end_date}, records={len(df)}")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in /actuals endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch actuals")

@app.get("/forecast")
def get_forecast(model: str, start_date: Optional[str] = None, end_date: Optional[str] = None, user: str = Depends(auth_required)):
    try:
        if model.lower() not in ("nbeatsx", "tft", "linearregression", "linreg", "gradientboosting", "gb"):
            logger.warning(f"User {user} passed invalid model: {model}")
            raise HTTPException(status_code=400, detail="Model must be 'NBEATSx', 'TFT', 'LinearRegression', or 'GradientBoosting'")

        df = load_forecast_df(model)
        if df.empty:
            return []

        # Rename column for uniform contract if it is NBEATSx, TFT, or forecast
        if model.lower() == "nbeatsx" and "NBEATSx" in df.columns:
            df = df.rename(columns={"NBEATSx": "forecast"})
        elif model.lower() == "tft" and "TFT" in df.columns:
            df = df.rename(columns={"TFT": "forecast"})
        elif "forecast" not in df.columns:
            non_ds_cols = [c for c in df.columns if c != "ds"]
            if non_ds_cols:
                df = df.rename(columns={non_ds_cols[0]: "forecast"})

        if start_date:
            df = df[df["ds"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["ds"] <= pd.to_datetime(end_date)]
        logger.info(f"Forecast requested by user {user}: model={model}, start={start_date}, end={end_date}, records={len(df)}")
        return df[["ds", "forecast"]].to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch forecast")

@app.get("/anomalies")
def get_anomalies(start_date: Optional[str] = None, end_date: Optional[str] = None, user: str = Depends(auth_required)):
    try:
        df = load_anomalies_df()
        if df.empty:
            return []
        if start_date:
            df = df[df["ds"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["ds"] <= pd.to_datetime(end_date)]
        logger.info(f"Anomalies requested by user {user}: start={start_date}, end={end_date}, records={len(df)}")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in /anomalies endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomalies")
@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify API is up.
    Returns HTTP 200 OK with status message.
    """
    return {"status": "UP", "message": "API is running smoothly"}
