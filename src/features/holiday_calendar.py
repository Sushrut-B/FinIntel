import pandas as pd

def get_holidays():
    # Minimal festival list for MVP
    holidays = [
        "2025-01-26", # Republic Day
        "2025-03-14", # Holi (example date)
        "2025-08-15", # Independence Day
        "2025-11-12", # Diwali
    ]
    return pd.to_datetime(holidays)

def add_calendar_features(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["dow"] = df[date_col].dt.dayofweek   # 0=Mon
    df["dom"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["eom"] = df[date_col].dt.is_month_end.astype(int)

    # Salary window: 1st–7th
    df["salary_window"] = df["dom"].between(1,7).astype(int)

    # Holidays
    holidays = get_holidays()
    df["is_holiday"] = df[date_col].isin(holidays).astype(int)

    return df
