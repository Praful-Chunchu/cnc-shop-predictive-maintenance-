import pandas as pd
import yaml
from google.cloud import storage

from src.features import drop_unused_columns

RAW_PATH = "data/raw/ai4i2020.csv"
CLEANED_PATH = "data/cleaned/ai4i2020_cleaned.csv"

# no spaces or brackets so the API can use these as JSON keys
RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def clean(df):
    df = drop_unused_columns(df)
    df = df.rename(columns=RENAME_MAP)
    return df

def download_from_gcs(bucket_name, blob_name, local_path):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")





if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    raw_df = load_raw(config["data"]["raw_path"])
    cleaned_df = clean(raw_df)
    cleaned_df.to_csv(config["data"]["cleaned_path"], index=False)
    print(f"Saved {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns to {config['data']['cleaned_path']}")
