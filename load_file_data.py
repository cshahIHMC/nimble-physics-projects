"""
===============================================================================
Script Name: load_recorded_trial.py
Author: ChatGPT (for Chinmay Shah)
Date: Nov 2025
===============================================================================

Description:
    Provides helper functions to read CSV data logged by your IMU + XSENSOR
    logging script. The CSV is dynamically parsed — any number of IMUs, any
    column ordering, and all insole fields are supported.

Usage:
    from load_recorded_trial import load_trial, get_imu_data, get_insole_data

    df = load_trial("logs/2025-11-17/trial2.csv")

    # Example: extract pelvis IMU quaternion
    quat = get_imu_data(df, imu_index=1, field="quat")

===============================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# LOAD FULL CSV FILE
# ---------------------------------------------------------------------------
def load_trial(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file and return a full pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV saved by your logging script.

    Returns
    -------
    df : pandas.DataFrame
    """
    df = pd.read_csv(csv_path)
    print(f"[load_trial] Loaded {csv_path} with shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    logs_root = os.path.join(os.getcwd(), "logs")
    if not os.path.isdir(logs_root):
        raise FileNotFoundError("logs/ folder not found in current directory")

    # --- Ask for trial name ---
    trial = input("Enter trial file name (e.g., trial2.csv): ").strip()
    if not trial.endswith(".csv"):
        trial += ".csv"

    # --- Search for this file inside logs/ recursively ---
    matches = []
    for root, _, files in os.walk(logs_root):
        if trial in files:
            matches.append(os.path.join(root, trial))

    if not matches:
        raise FileNotFoundError(f"'{trial}' not found in any folder inside logs/")

    # If multiple matches → pick first (shortest code)
    csv_path = matches[0]
    print(f"Using file: {csv_path}")

    # --- Load & print basic data ---
    df = load_trial(csv_path)
    
