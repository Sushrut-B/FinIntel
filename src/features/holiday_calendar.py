import pandas as pd
import holidays as pyholidays

def add_calendar_features(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract baseline date indices
    df["month_idx"] = df[date_col].dt.month
    df["quarter_idx"] = df[date_col].dt.quarter
    df["year_idx"] = df[date_col].dt.year
    
    # Query India holidays dynamically for all years in the dataset
    years = list(df[date_col].dt.year.unique())
    if not years:
        years = [2024]
        
    in_holidays = pyholidays.India(years=years)
    
    # Construct holidays dataframe to group by month
    holiday_dates = pd.to_datetime(list(in_holidays.keys()))
    holiday_df = pd.DataFrame({"holiday_date": holiday_dates})
    holiday_df["year_idx"] = holiday_df["holiday_date"].dt.year
    holiday_df["month_idx"] = holiday_df["holiday_date"].dt.month
    
    # Count holidays per month
    holiday_counts = holiday_df.groupby(["year_idx", "month_idx"]).size().rename("holiday_count").reset_index()
    
    # Merge with original dataframe
    df = df.merge(holiday_counts, on=["year_idx", "month_idx"], how="left")
    df["holiday_count"] = df["holiday_count"].fillna(0).astype(int)
    
    # Flag months with substantial holiday concentrations
    df["is_holiday_month"] = (df["holiday_count"] >= 2).astype(int)
    
    return df
