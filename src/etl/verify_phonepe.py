import pandas as pd

if __name__ == "__main__":
    phonepe = pd.read_parquet("data/silver/phonepe_txn_silver.parquet")
    print("PhonePe Silver data sample:")
    print(phonepe.head())
    print(phonepe.dtypes)
