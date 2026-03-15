# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 20:39:56 2025

@author: bikra
"""

# =========================
# Evaluation metrics and plots 
# =========================

ELEMENT = "qr"  # <-- choose "wl" (waterlevel) or "q" (discharge)

# Base folder containing forecast results (OrgP/BestP CSVs; nested subfolders OK)
BASE_PATH_WL = r"C:/Users/bikra/Desktop/DA_PF_RRI/Daily/Simulated/nan_wl_2024_likelihood/Results"
BASE_PATH_Q  = r"C:/Users/bikra/Desktop/DA_PF_RRI/10min/Simulated/2024/nan_wl/Results"  # example; adjust if different

# Observed CSV path (must have columns: Datetime, Obs) for WL or Q
OBS_FILE_WL = r"C:/Users/bikra/Desktop/DA_PF_RRI/Daily/Simulated/nan_wl_2024_likelihood/ObsData/WaterLevel/ObsWL.csv"
OBS_FILE_Q  = r"C:/Users/bikra/Desktop/DA_PF_RRI/10min/Simulated/2024/nan_wl/ObsData/Discharge/ObsQ.csv"  # example; adjust

# Output folder for metrics & plots (will be created if missing)
OUT_DIR_WL = r"C:/Users/bikra/Desktop/DA_PF_RRI/Daily/Simulated/nan_wl_2024_likelihood/Results/metrics"
OUT_DIR_Q  = r"C:/Users/bikra/Desktop/DA_PF_RRI/10min/Simulated/2024/nan_wl/Results/metrics"

MAX_LEAD_DAYS = 10  # evaluate lead days 1..10

# =========================
# SCRIPT (no edits needed)
# =========================

import os  # filesystem walking and path ops
import pandas as pd  # data handling
import numpy as np  # metrics math
import matplotlib.pyplot as plt  # plotting
from datetime import datetime  # parse forecast cycle timestamps

# ---- pick paths by ELEMENT ----
if ELEMENT.lower() == "wl":  # waterlevel mode
    base_path = BASE_PATH_WL  # folder to scan for forecast CSVs
    obs_file  = OBS_FILE_WL   # observed waterlevel file (Datetime, Obs)
    out_dir   = OUT_DIR_WL    # output dir for Excel/plots
else:  # discharge mode
    base_path = BASE_PATH_Q   # folder to scan for forecast CSVs
    obs_file  = OBS_FILE_Q    # observed discharge file (Datetime, Obs)
    out_dir   = OUT_DIR_Q     # output dir for Excel/plots

os.makedirs(out_dir, exist_ok=True)  # create output folder if it doesn't exist

# ---- helper: load a CSV and return a value Series indexed by Datetime ----
def load_value_series(csv_path):
    """
    Reads a CSV with a 'Datetime' column plus one or more numeric columns.
    Returns a numeric pd.Series indexed by Datetime (string), using the first numeric column.
    If duplicates exist, they are averaged so .loc[...] yields a scalar.
    """
    df = pd.read_csv(csv_path, dtype={"Datetime": str}).dropna()  # read + drop empty rows
    df = df.set_index("Datetime")  # index by timestamp text like 'YYYYMMDDHHMM'
    # choose the first numeric column robustly (works for WL/Q with different names)
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError(f"No numeric columns found in {csv_path}")
    ser = df[num_cols[0]].astype(float)  # use first numeric column as our value series
    if not ser.index.is_unique:
        ser = ser.groupby(level=0).mean()  # collapse duplicate timestamps to a single value
    return ser  # Series: index=Datetime (str), values=float

# ---- helper: compute lead day (1..MAX_LEAD_DAYS) from init timestamp and target timestamp ----

def compute_lead_day(init_str, ts_str):
    """
    Lead starts from the *next* timestep after init:
      - t == init        -> excluded
      - 0 < Δt <= 1 day  -> Lead 1
      - 1 < Δt <= 2 days -> Lead 2
      - ...
    """
    t0 = datetime.strptime(init_str, "%Y%m%d%H%M")  # init time
    t1 = datetime.strptime(ts_str,  "%Y%m%d%H%M")  # target time
    delta_days = (t1 - t0).total_seconds() / 86400.0  # days difference

    if delta_days <= 0:            # same time or earlier -> NOT a valid lead
        return None

    lead = int(np.ceil(delta_days))  # (0,1] -> 1, (1,2] -> 2, ...
    return lead if 1 <= lead <= MAX_LEAD_DAYS else None


# ---- scan for forecast CSVs and pair by cycle init (basename prefix before first underscore) ----
org_files, best_files = [], []  # buckets for OrgP and BestP
for root, _, files in os.walk(base_path):  # walk through base_path recursively
    for fname in files:  # check each file
        if not fname.endswith(".csv"):
            continue  # skip non-CSV files
        fpath = os.path.join(root, fname)  # full path
        if "OrgP" in fname:
            org_files.append(fpath)  # Org particle CSV
        elif "BestP" in fname:
            best_files.append(fpath)  # Best particle CSV

# map init -> path for each class, using prefix before first underscore as the init key
org_dict  = {os.path.basename(f).split("_")[0]: f for f in org_files}   # e.g., '202408110800': '...OrgP...csv'
best_dict = {os.path.basename(f).split("_")[0]: f for f in best_files}  # e.g., '202408110800': '...BestP...csv'

common_inits = sorted(set(org_dict) & set(best_dict))  # only cycles that have both OrgP and BestP
print(f"Found {len(common_inits)} forecast cycles with both OrgP and BestP files.")

# ---- load observations as a Series (Datetime index -> float value) ----
obs_df = pd.read_csv(obs_file, dtype={"Datetime": str}).dropna()  # read observed
if "Obs" in obs_df.columns:
    obs_series = obs_df.set_index("Datetime")["Obs"].astype(float)  # use 'Obs' col if present
else:
    # fallback to first numeric column if 'Obs' not present
    col = obs_df.select_dtypes(include="number").columns[0]
    obs_series = obs_df.set_index("Datetime")[col].astype(float)

if not obs_series.index.is_unique:
    obs_series = obs_series.groupby(level=0).mean()  # dedup timestamps

# ---- metrics functions (NSE, KGE, RMSE, Bias) ----
def compute_nse(obs, sim):
    """Nash-Sutcliffe Efficiency (requires at least 1 point; variance of obs != 0)."""
    obs = np.array(obs); sim = np.array(sim)
    if obs.size == 0:
        return None
    if np.all(obs == obs[0]):  # constant observations case
        return 1.0 if np.allclose(sim, obs) else float("-inf")
    ss_res = np.sum((sim - obs)**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    return 1 - ss_res/ss_tot

def compute_kge(obs, sim):
    """Kling-Gupta Efficiency (requires at least 2 points for correlation)."""
    obs = np.array(obs); sim = np.array(sim)
    if obs.size < 2:
        return None
    mu_o, mu_s = np.mean(obs), np.mean(sim)
    std_o, std_s = np.std(obs, ddof=0), np.std(sim, ddof=0)
    if std_o == 0 or std_s == 0:
        r = 1.0 if np.allclose(sim, obs) else 0.0
    else:
        r = np.corrcoef(obs, sim)[0, 1]
    alpha = std_s / std_o if std_o != 0 else np.nan
    beta  = mu_s / mu_o if mu_o != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return None
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def compute_rmse(obs, sim):
    obs = np.array(obs); sim = np.array(sim)
    if obs.size == 0:
        return None
    return float(np.sqrt(np.mean((sim - obs)**2)))

def compute_bias(obs, sim):
    obs = np.array(obs); sim = np.array(sim)
    if obs.size == 0:
        return None
    return float(np.mean(sim - obs))

# ---- collect matched triples for ALL timestamps per cycle, assign LeadDay ----
raw_rows = []  # will populate with dicts: cycle, timestamp, lead, obs, org, best

for init in common_inits:  # loop all cycles with both OrgP and BestP
    org_ser  = load_value_series(org_dict[init])   # OrgP Series, index=Datetime, float values
    best_ser = load_value_series(best_dict[init])  # BestP Series, index=Datetime, float values

    # intersection of timestamps across all three series (Obs ∩ OrgP ∩ BestP)
    common_ts = sorted(set(obs_series.index) & set(org_ser.index) & set(best_ser.index))

    for ts in common_ts:  # for each matched timestamp
        lead = compute_lead_day(init, ts)  # bin into 1..MAX_LEAD_DAYS (or None if out of range)
        if lead is None:
            continue  # skip timestamps outside requested lead window
        # Series.loc[...] returns scalar because we deduped indices -> no FutureWarning
        raw_rows.append({
            "ForecastCycle": init,          # e.g., '202408110800'
            "Datetime": ts,                 # e.g., '202408121230'
            "LeadDay": lead,                # 1..MAX_LEAD_DAYS
            "Obs":  float(obs_series.loc[ts]),   # observed value at ts
            "OrgP": float(org_ser.loc[ts]),      # org particle simulated value at ts
            "BestP": float(best_ser.loc[ts]),    # best particle simulated value at ts
        })

# ---- build a tidy DataFrame of raw matched values ----
raw_df = pd.DataFrame(raw_rows)  # columns: ForecastCycle, Datetime, LeadDay, Obs, OrgP, BestP
raw_df = raw_df.sort_values(["ForecastCycle", "LeadDay", "Datetime"]).reset_index(drop=True)  # clean ordering

print(f"Total matched points kept (all cycles, all 10-min stamps within 1..{MAX_LEAD_DAYS} days): {len(raw_df)}")

# ---- compute metrics per lead day using ALL matched 10-min stamps for that lead ----
metrics_records = []  # one row per lead day

for lead in range(1, MAX_LEAD_DAYS + 1):  # lead 1..10
    df_lead = raw_df[raw_df["LeadDay"] == lead]  # slice all rows at this lead day
    obs_vals  = df_lead["Obs"].to_numpy()        # vector of obs at this lead
    org_vals  = df_lead["OrgP"].to_numpy()       # vector of OrgP at this lead
    best_vals = df_lead["BestP"].to_numpy()      # vector of BestP at this lead

    metrics_records.append({
        "Lead": lead,                                  # lead day number
        "NSE_Org":  compute_nse(obs_vals, org_vals),   # NSE for OrgP
        "NSE_Best": compute_nse(obs_vals, best_vals),  # NSE for BestP
        "KGE_Org":  compute_kge(obs_vals, org_vals),   # KGE for OrgP
        "KGE_Best": compute_kge(obs_vals, best_vals),  # KGE for BestP
        "RMSE_Org": compute_rmse(obs_vals, org_vals),  # RMSE for OrgP
        "RMSE_Best":compute_rmse(obs_vals, best_vals), # RMSE for BestP
        "Bias_Org": compute_bias(obs_vals, org_vals),  # Bias for OrgP
        "Bias_Best":compute_bias(obs_vals, best_vals)  # Bias for BestP
    })

metrics_df = pd.DataFrame(metrics_records)  # assemble metrics table
print("Metrics (rounded preview):")
print(metrics_df.round(3))

# ---- write to Excel: Sheet1=Metrics, Sheet2=Raw values ----
excel_path = os.path.join(out_dir, "Forecast_Evaluation.xlsx")  # output excel file path
with pd.ExcelWriter(excel_path) as writer:  # create the workbook
    metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)  # first sheet
    raw_df.to_excel(writer,     sheet_name="Forecast_vs_Obs", index=False)  # second sheet
print(f"Excel written: {excel_path}")

# ---- plot metrics vs lead time (NSE, KGE, RMSE, Bias) ----
# Replace None / -inf with NaN for clean plotting
mp = metrics_df.replace({None: np.nan, float("-inf"): np.nan})  # safe plot values
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots

# NSE
axs[0,0].plot(mp["Lead"], mp["NSE_Org"], marker="o", label="OrgP")
axs[0,0].plot(mp["Lead"], mp["NSE_Best"], marker="s", label="BestP")
axs[0,0].set_title("NSE vs Lead Day"); axs[0,0].set_xlabel("Lead Day"); axs[0,0].set_ylabel("NSE")
axs[0,0].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[0,0].grid(True); axs[0,0].legend()

# KGE
axs[0,1].plot(mp["Lead"], mp["KGE_Org"], marker="o", label="OrgP")
axs[0,1].plot(mp["Lead"], mp["KGE_Best"], marker="s", label="BestP")
axs[0,1].set_title("KGE vs Lead Day"); axs[0,1].set_xlabel("Lead Day"); axs[0,1].set_ylabel("KGE")
axs[0,1].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[0,1].grid(True); axs[0,1].legend()

# RMSE
axs[1,0].plot(mp["Lead"], mp["RMSE_Org"], marker="o", label="OrgP")
axs[1,0].plot(mp["Lead"], mp["RMSE_Best"], marker="s", label="BestP")
axs[1,0].set_title("RMSE vs Lead Day"); axs[1,0].set_xlabel("Lead Day"); axs[1,0].set_ylabel("RMSE")
axs[1,0].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[1,0].grid(True); axs[1,0].legend()

# Bias
axs[1,1].plot(mp["Lead"], mp["Bias_Org"], marker="o", label="OrgP")
axs[1,1].plot(mp["Lead"], mp["Bias_Best"], marker="s", label="BestP")
axs[1,1].set_title("Bias vs Lead Day"); axs[1,1].set_xlabel("Lead Day"); axs[1,1].set_ylabel("Bias")
axs[1,1].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[1,1].grid(True); axs[1,1].legend()

plt.tight_layout()  # neat layout
plot_path = os.path.join(out_dir, "Metrics_vs_LeadDay.png")  # output figure path
plt.savefig(plot_path, dpi=300)  # save image
plt.show()
plt.close()  # close figure to free memory
print(f"Plot saved: {plot_path}")

#%%

#plot from the excel raw values 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load raw data (Forecast vs Obs) ===
excel_file = "C:/Users/bikra/Desktop/DA_PF_RRI/Daily/Simulated/nan_wl_2024/Results/metrics/Forecast_Evaluation.xlsx"
raw_df = pd.read_excel(excel_file, sheet_name="Forecast_vs_Obs")

# === 2. Define metric functions ===
def nse(obs, sim):
    obs, sim = np.array(obs), np.array(sim)
    if len(obs) == 0: return np.nan
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def kge(obs, sim):
    obs, sim = np.array(obs), np.array(sim)
    if len(obs) < 2: return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def rmse(obs, sim):
    obs, sim = np.array(obs), np.array(sim)
    if len(obs) == 0: return np.nan
    return np.sqrt(np.mean((sim - obs) ** 2))

def bias(obs, sim):
    obs, sim = np.array(obs), np.array(sim)
    if len(obs) == 0: return np.nan
    return np.mean(sim - obs)

# === 3. Compute metrics per lead day ===
metrics_summary = []
for lead, group in raw_df.groupby("LeadDay"):
    obs = group["Obs"].values
    org = group["OrgP"].values
    best = group["BestP"].values

    metrics_summary.append({
        "Lead": lead,
        "NSE_Org": nse(obs, org),
        "NSE_Best": nse(obs, best),
        "KGE_Org": kge(obs, org),
        "KGE_Best": kge(obs, best),
        "RMSE_Org": rmse(obs, org),
        "RMSE_Best": rmse(obs, best),
        "Bias_Org": bias(obs, org),
        "Bias_Best": bias(obs, best),
    })

metrics_df = pd.DataFrame(metrics_summary).sort_values("Lead")
print(metrics_df.round(3))

# === 4. Plot metrics ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# NSE
axs[0, 0].plot(metrics_df['Lead'], metrics_df['NSE_Org'], marker='o', label='OrgP')
axs[0, 0].plot(metrics_df['Lead'], metrics_df['NSE_Best'], marker='s', label='BestP')
axs[0, 0].set_title("Nash–Sutcliffe Efficiency (NSE)")
axs[0, 0].set_xlabel("Lead Time (Days)"); axs[0, 0].set_ylabel("NSE")
axs[0, 0].set_xticks(range(1, 11))
axs[0, 0].legend(); axs[0, 0].grid(True)

# KGE
axs[0, 1].plot(metrics_df['Lead'], metrics_df['KGE_Org'], marker='o', label='OrgP')
axs[0, 1].plot(metrics_df['Lead'], metrics_df['KGE_Best'], marker='s', label='BestP')
axs[0, 1].set_title("Kling–Gupta Efficiency (KGE)")
axs[0, 1].set_xlabel("Lead Time (Days)"); axs[0, 1].set_ylabel("KGE")
axs[0, 1].set_xticks(range(1, 11))
axs[0, 1].legend(); axs[0, 1].grid(True)

# RMSE
axs[1, 0].plot(metrics_df['Lead'], metrics_df['RMSE_Org'], marker='o', label='OrgP')
axs[1, 0].plot(metrics_df['Lead'], metrics_df['RMSE_Best'], marker='s', label='BestP')
axs[1, 0].set_title("Root Mean Square Error (RMSE)")
axs[1, 0].set_xlabel("Lead Time (Days)"); axs[1, 0].set_ylabel("RMSE")
axs[1, 0].set_xticks(range(1, 11))
axs[1, 0].legend(); axs[1, 0].grid(True)

# Bias
axs[1, 1].plot(metrics_df['Lead'], metrics_df['Bias_Org'], marker='o', label='OrgP')
axs[1, 1].plot(metrics_df['Lead'], metrics_df['Bias_Best'], marker='s', label='BestP')
axs[1, 1].set_title("Bias")
axs[1, 1].set_xlabel("Lead Time (Days)"); axs[1, 1].set_ylabel("Bias")
axs[1, 1].set_xticks(range(1, 11))
axs[1, 1].legend(); axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("Metrics_vs_LeadTime_fromRaw.png", dpi=300)
plt.show()