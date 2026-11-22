import numpy as np
import pandas as pd

TARGET = "Machine failure"

# UDI and Product ID are just identifiers
ID_COLS = ["UDI", "Product ID"]

# failure modes are only known after a failure happens (leakage)
FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

INPUT_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_COLS = ["temp_diff", "power", "strain"]


def drop_unused_columns(df):
    return df.drop(columns=ID_COLS + FAILURE_COLS)


def add_engineered_features(df):
    df = df.copy()

    # heat dissipation failure (HDF)
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # power failure (PWF), rpm converted to rad/s so power is in watts
    df["power"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    # overstrain failure (OSF)
    df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return df
