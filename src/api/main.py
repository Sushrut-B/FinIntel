from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import logging

app = FastAPI(title="UPI Macro Intelligence API")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for local dashboard testing allowing POST for login
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# JWT Authentication setup
import jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.database.db_manager import get_db_session, User, check_password

SECRET_KEY = "finintel_super_secret_cryptographic_key_google_standard"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def auth_required(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(login_data: LoginRequest):
    username = login_data.username
    password = login_data.password
    
    with get_db_session() as session:
        user = session.query(User).filter(User.username == username).first()
        if not user or not check_password(password, user.password_hash):
            logger.warning(f"Failed login attempt for user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username, "role": user.role},
        expires_delta=access_token_expires
    )
    logger.info(f"User {username} logged in successfully via JWT.")
    return {"access_token": access_token, "token_type": "bearer"}

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

import os
import threading

class CachedDataset:
    def __init__(self, file_path: str, load_func):
        self.file_path = file_path
        self.load_func = load_func
        self.cached_records = []
        self.last_mtime = 0.0
        self.lock = threading.Lock()

    def get_records(self):
        if not os.path.exists(self.file_path):
            return []
        
        try:
            current_mtime = os.path.getmtime(self.file_path)
        except Exception:
            return self.cached_records

        # Rapid read lockless check
        if self.last_mtime == current_mtime:
            return self.cached_records

        # Acquire lock to update safely
        with self.lock:
            if self.last_mtime != current_mtime:
                try:
                    logger.info(f"Reloading cache for file: {self.file_path}")
                    df = self.load_func(self.file_path)
                    if not df.empty:
                        # Clean columns and coerce timestamps to string format for fast JSON serialization
                        df_copy = df.copy()
                        if "ds" in df_copy.columns:
                            df_copy["ds"] = df_copy["ds"].astype(str)
                        if "month" in df_copy.columns:
                            df_copy["month"] = df_copy["month"].astype(str)
                        self.cached_records = df_copy.to_dict(orient="records")
                    else:
                        self.cached_records = []
                    self.last_mtime = current_mtime
                except Exception as e:
                    logger.error(f"Error reloading cache for {self.file_path}: {e}")
        return self.cached_records

def load_gold_raw(path):
    df = pd.read_parquet(path).rename(columns={"month": "ds", "amount": "y"})
    return df

def load_csv_raw(path):
    return pd.read_csv(path)

# Initialize caches globally
gold_cache = CachedDataset("data/gold/upi_macro_gold.parquet", load_gold_raw)
nbeats_cache = CachedDataset("data/gold/upi_forecast_backtest_nbeats.csv", load_csv_raw)
tft_cache = CachedDataset("data/gold/upi_forecast_backtest_tft.csv", load_csv_raw)
linreg_cache = CachedDataset("data/gold/upi_forecast_backtest_linreg.csv", load_csv_raw)
anomalies_cache = CachedDataset("data/gold/upi_anomalies.csv", load_csv_raw)

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the UPI Macro Intelligence API"}

@app.get("/actuals")
def get_actuals(start_date: Optional[str] = None, end_date: Optional[str] = None, user: str = Depends(auth_required)):
    try:
        records = gold_cache.get_records()
        if not records:
            return []
        
        filtered = records
        if start_date:
            try:
                start_str = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            filtered = [r for r in filtered if r["ds"][:10] >= start_str]
        
        if end_date:
            try:
                end_str = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            filtered = [r for r in filtered if r["ds"][:10] <= end_str]

        logger.info(f"Actuals requested: start={start_date}, end={end_date}, records={len(filtered)}")
        return filtered
    except Exception as e:
        logger.error(f"Error in /actuals endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch actuals")

@app.get("/forecast")
def get_forecast(model: str, start_date: Optional[str] = None, end_date: Optional[str] = None, user: str = Depends(auth_required)):
    try:
        model_name = model.lower()
        if model_name not in ("nbeatsx", "tft", "linearregression", "linreg", "gradientboosting", "gb"):
            logger.warning(f"Invalid model: {model}")
            raise HTTPException(status_code=400, detail="Model must be 'NBEATSx', 'TFT', 'LinearRegression', or 'GradientBoosting'")

        if model_name == "nbeatsx":
            records = nbeats_cache.get_records()
            forecast_col = "NBEATSx"
        elif model_name == "tft":
            records = tft_cache.get_records()
            forecast_col = "TFT"
        else:
            records = linreg_cache.get_records()
            forecast_col = "forecast"

        if not records:
            return []

        # Identify forecast column dynamically
        keys = list(records[0].keys())
        actual_col = forecast_col
        if forecast_col not in keys:
            non_ds_cols = [k for k in keys if k != "ds"]
            if non_ds_cols:
                actual_col = non_ds_cols[0]

        # Pre-normalize filters
        start_str = None
        if start_date:
            try:
                start_str = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")

        end_str = None
        if end_date:
            try:
                end_str = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        filtered = []
        for r in records:
            ds_str = r["ds"][:10]
            if start_str and ds_str < start_str:
                continue
            if end_str and ds_str > end_str:
                continue
            
            val = r.get(actual_col, 0.0)
            if pd.isna(val):
                val = 0.0
            
            filtered.append({
                "ds": r["ds"],
                "forecast": val
            })

        logger.info(f"Forecast requested: model={model}, start={start_date}, end={end_date}, records={len(filtered)}")
        return filtered
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch forecast")

@app.get("/anomalies")
def get_anomalies(start_date: Optional[str] = None, end_date: Optional[str] = None, user: str = Depends(auth_required)):
    try:
        from src.database.db_manager import get_db_session, AnomalyRecord
        
        with get_db_session() as session:
            query = session.query(AnomalyRecord)
            if start_date:
                try:
                    start_str = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
                except ValueError:
                    start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
                    
                # Format to date-only string for comparison
                start_str = start_str[:10]
                query = query.filter(AnomalyRecord.ds >= start_str)
            if end_date:
                try:
                    end_str = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
                except ValueError:
                    end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
                    
                # Format to date-only string for comparison
                end_str = end_str[:10]
                query = query.filter(AnomalyRecord.ds <= end_str)
                
            records = query.order_by(AnomalyRecord.ds.asc()).all()
            
            filtered = []
            for r in records:
                filtered.append({
                    "ds": r.ds,
                    "y": r.y,
                    "forecast": r.forecast,
                    "residual": r.residual,
                    "type": r.type
                })
                
        logger.info(f"Anomalies requested: start={start_date}, end={end_date}, records={len(filtered)}")
        return filtered
    except Exception as e:
        logger.error(f"Error in /anomalies endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomalies")

@app.get("/health")
def health_check():
    return {"status": "UP", "message": "API is running smoothly"}
