import numpy as np
import pandas as pd

TARGET = "machine_failure"

# UDI and Product ID are just identifiers
ID_COLS = ["UDI", "Product ID"]

# failure modes are only known after a failure happens (leakage)
FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

INPUT_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

ENGINEERED_COLS = ["temp_diff", "power", "strain"]


def drop_unused_columns(df):
    return df.drop(columns=ID_COLS + FAILURE_COLS)


def add_engineered_features(df):
    df = df.copy()

    # heat dissipation failure (HDF)
    df["temp_diff"] = df["process_temp_k"] - df["air_temp_k"]

    # power failure (PWF), rpm converted to rad/s so power is in watts
    df["power"] = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * np.pi / 60)

    # overstrain failure (OSF)
    df["strain"] = df["tool_wear_min"] * df["torque_nm"]

    return df
