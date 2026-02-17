import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, NBeats, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

# Load your historical data CSV (or modify to load from your DB/API)
data = pd.read_csv("upi_historical_data.csv", parse_dates=["ds"])
data["time_idx"] = (data["ds"] - data["ds"].min()).dt.days // 30  # monthly index
data["group"] = 0  # single time series group since you have single series

# Define dataset for training
max_encoder_length = 36  # months used to encode history
max_prediction_length = 60  # predict next 60 months

training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="y",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["y"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Validation set
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# Create dataloaders for PyTorch
train_dataloader = training.to_dataloader(train=True, batch_size=64)
val_dataloader = validation.to_dataloader(train=False, batch_size=64)

# Initialize NBeats model
pl.seed_everything(42)
nbeats = NBeats.from_dataset(training, learning_rate=1e-3, log_interval=10, log_val_interval=1)

# Trainer
trainer = pl.Trainer(max_epochs=30, gpus=1 if torch.cuda.is_available() else 0)

# Train model
trainer.fit(nbeats, train_dataloader, val_dataloader)

# Save model
trainer.save_checkpoint("nbeats_model.ckpt")

# Load trained model (for inference)
best_model = NBeats.load_from_checkpoint("nbeats_model.ckpt")

# Predict future next 60 months
raw_predictions, x = best_model.predict(validation, mode="raw", return_x=True)
forecast = raw_predictions["prediction"].detach().cpu().numpy()

# Map predictions to dates
future_dates = pd.date_range(start=data["ds"].max() + pd.DateOffset(months=1), periods=max_prediction_length, freq='MS')
forecast_df = pd.DataFrame({"ds": future_dates, "forecast": forecast.ravel()})

# Save forecast for dashboard use
forecast_df.to_csv("future_forecast.csv", index=False)
