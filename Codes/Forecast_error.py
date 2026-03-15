

"""
Forecasting Error plot before and after Data Assimilation
metrics plot
excel files
specific start and end time
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

# Choose element type
ELEMENT = "wl" # "wl" (waterlevel) or "qr" (discharge)

# Define ONE base path
BASE_DIR = r"C:\Users\bikra\Desktop\Paper\PF-DA\ecmwf_2025"
rain = "ECMWF"
date = "2025"

# Water Level paths
BASE_PATH_WL = os.path.join(BASE_DIR, "Results")
OBS_FILE_WL  = os.path.join(BASE_DIR, "ObsData/WaterLevel/ObsWL.csv")
OUT_DIR_WL   = os.path.join(BASE_DIR, "Results/error_plots_wl")

# Discharge paths
BASE_PATH_Q = os.path.join(BASE_DIR, "Results")
OBS_FILE_Q  = os.path.join(BASE_DIR, "ObsData/Discharge/ObsQ.csv")
OUT_DIR_Q   = os.path.join(BASE_DIR, "Results/error_plots_spc")

MAX_LEAD_DAYS = 10

# --- NEW: Optional date window (inclusive). Use 'YYYYMMDDHHMM' or set to None to disable.
DATE_START = None  # e.g., "202405010000" or None
DATE_END   = None  # e.g., "202406300000" or None

#DATE_START = "2025071000000"  # e.g., "202405010000" or None
#DATE_END   = "2025081000000"  # e.g., "202406300000" or None
# -------------------------------------------------------

# =========================
# HELPER FUNCTIONS
# =========================

def _in_window(ts_str):
    """Return True if ts_str is within [DATE_START, DATE_END] (inclusive). If window is None, always True."""
    if DATE_START and ts_str < DATE_START:
        return False
    if DATE_END and ts_str > DATE_END:
        return False
    return True

def load_value_series(csv_path):
    """Load CSV and return a Series indexed by Datetime."""
    try:
        df = pd.read_csv(csv_path, dtype={"Datetime": str}).dropna()
        df = df.set_index("Datetime")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError(f"No numeric columns found in {csv_path}")
        ser = df[num_cols[0]].astype(float)
        if not ser.index.is_unique:
            ser = ser.groupby(level=0).mean()
        return ser
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.Series(dtype=float)

def compute_lead_day(init_str, ts_str):
    """Compute lead day from init and target timestamps."""
    t0 = datetime.strptime(init_str, "%Y%m%d%H%M")
    t1 = datetime.strptime(ts_str, "%Y%m%d%H%M")
    delta_days = (t1 - t0).total_seconds() / 86400.0
    if delta_days <= 0:
        return None
    lead = int(np.ceil(delta_days))
    return lead if 1 <= lead <= MAX_LEAD_DAYS else None

def _cycle_overlaps_window(series_index_as_str):
    """Return True if at least one timestamp in the series index is inside the DATE window."""
    if not DATE_START and not DATE_END:
        return True
    for ts in series_index_as_str:
        if _in_window(ts):
            return True
    return False

def process_forecast_data(base_path, obs_series, date_start=None, date_end=None):
    """Process both OrgP and BestP forecast data and return error DataFrames and raw data."""
    org_files, best_files = [], []
    for root, _, files in os.walk(base_path):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(root, fname)
            if "OrgP" in fname and f"_{ELEMENT.lower()}_" in fname:
                org_files.append(fpath)
            elif "BestP" in fname and f"_{ELEMENT.lower()}_" in fname:
                best_files.append(fpath)

    org_dict = {os.path.basename(f).split("_")[0]: f for f in org_files}
    best_dict = {os.path.basename(f).split("_")[0]: f for f in best_files}
    common_inits = sorted(set(org_dict) & set(best_dict))
    print(f"Found {len(common_inits)} forecast cycles with both OrgP and BestP files.")

    org_errors, best_errors, raw_rows = [], [], []

    for init in common_inits:
        try:
            org_ser = load_value_series(org_dict[init])
            best_ser = load_value_series(best_dict[init])

            if date_start or date_end:
                if not _cycle_overlaps_window(org_ser.index):
                    continue
                if not _cycle_overlaps_window(best_ser.index):
                    continue

            common_ts = sorted(set(obs_series.index) & set(org_ser.index) & set(best_ser.index))
            if date_start or date_end:
                common_ts = [ts for ts in common_ts if _in_window(ts)]

            for ts in common_ts:
                lead = compute_lead_day(init, ts)
                if lead is None:
                    continue

                obs_val = float(obs_series.loc[ts])
                org_val = float(org_ser.loc[ts])
                best_val = float(best_ser.loc[ts])

                org_errors.append({
                    "ForecastCycle": init, "Datetime": ts, "LeadDay": lead, "Error": org_val - obs_val
                })
                best_errors.append({
                    "ForecastCycle": init, "Datetime": ts, "LeadDay": lead, "Error": best_val - obs_val
                })
                raw_rows.append({
                    "ForecastCycle": init, "Datetime": ts, "LeadDay": lead,
                    "Obs": obs_val, "OrgP": org_val, "BestP": best_val
                })
        except Exception as e:
            print(f"Skipping cycle {init} due to error: {e}")

    raw_df = pd.DataFrame(raw_rows)
    return pd.DataFrame(org_errors), pd.DataFrame(best_errors), raw_df

def compute_metrics(raw_df):
    """Compute NSE, KGE, RMSE, PBIAS metrics for each lead day."""
    def compute_nse(obs, sim):
        obs = np.array(obs); sim = np.array(sim)
        if obs.size == 0:
            return None
        if np.all(obs == obs[0]):
            return 1.0 if np.allclose(sim, obs) else float("-inf")
        ss_res = np.sum((sim - obs)**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        return 1 - ss_res/ss_tot

    def compute_kge(obs, sim):
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
        beta = mu_s / mu_o if mu_o != 0 else np.nan
        if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
            return None
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    def compute_rmse(obs, sim):
        obs = np.array(obs); sim = np.array(sim)
        if obs.size == 0:
            return None
        return float(np.sqrt(np.mean((sim - obs)**2)))

    def compute_pbias(obs, sim):
        """Percent bias: + overestimation, - underestimation."""
        obs = np.array(obs, dtype=float); sim = np.array(sim, dtype=float)
        if obs.size == 0 or np.isclose(np.sum(obs), 0):
            return None
        return float(100.0 * (np.sum(sim - obs) / np.sum(obs)))

    metrics_records = []
    for lead in range(1, MAX_LEAD_DAYS + 1):
        df_lead = raw_df[raw_df["LeadDay"] == lead]
        obs_vals = df_lead["Obs"].to_numpy()
        org_vals = df_lead["OrgP"].to_numpy()
        best_vals = df_lead["BestP"].to_numpy()

        metrics_records.append({
            "Lead": lead,
            "NSE_Org":  compute_nse(obs_vals, org_vals),
            "NSE_Best": compute_nse(obs_vals, best_vals),
            "KGE_Org":  compute_kge(obs_vals, org_vals),
            "KGE_Best": compute_kge(obs_vals, best_vals),
            "RMSE_Org": compute_rmse(obs_vals, org_vals),
            "RMSE_Best":compute_rmse(obs_vals, best_vals),
            "PBIAS_Org": compute_pbias(obs_vals, org_vals),
            "PBIAS_Best":compute_pbias(obs_vals, best_vals),
        })
    return pd.DataFrame(metrics_records)

def create_combined_error_plot(error_df_before, error_df_after, title, save_path):
    """Create combined forecasting error plot with mean, Q1, Q3 lines, and vertical IQR lines."""
    plot_data_before = error_df_before[error_df_before["LeadDay"] >= 1].copy()
    plot_data_after  = error_df_after [error_df_after ["LeadDay"] >= 1].copy()

    stats_before = plot_data_before.groupby("LeadDay")["Error"].agg(
        mean='mean', q1=lambda x: np.quantile(x, 0.25), q3=lambda x: np.quantile(x, 0.75)
    ).reset_index()

    stats_after = plot_data_after.groupby("LeadDay")["Error"].agg(
        mean='mean', q1=lambda x: np.quantile(x, 0.25), q3=lambda x: np.quantile(x, 0.75)
    ).reset_index()

    if stats_before.empty and stats_after.empty:
        print("No data available for plotting the combined error lines.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Before DA
    if not stats_before.empty:
        ax.plot(stats_before["LeadDay"], stats_before["mean"], marker='o', linestyle='-', color='red', linewidth=1.5, label=f'{rain} Mean Error')
        ax.plot(stats_before["LeadDay"], stats_before["q1"], linestyle='--', color='red', linewidth=1.0, alpha=0.5)
        ax.plot(stats_before["LeadDay"], stats_before["q3"], linestyle='--', color='red', linewidth=1.0, alpha=0.5)
        ax.vlines(stats_before["LeadDay"], stats_before["q1"], stats_before["q3"], color='black', linestyle='-', linewidth=1.5, zorder=1)
        for _, row in stats_before.iterrows():
            lead = row["LeadDay"]
            ax.annotate(f"{row['mean']:.2f}", (lead, row['mean']), textcoords="offset points", xytext=(-15,10), ha='center', color='red', fontsize=10)

    # After DA
    if not stats_after.empty:
        ax.plot(stats_after["LeadDay"], stats_after["mean"], marker='s', linestyle='-', color='blue', linewidth=1.5, label='PF-DA Mean Error')
        ax.plot(stats_after["LeadDay"], stats_after["q1"], linestyle=':', color='blue', linewidth=1.0, alpha=0.5)
        ax.plot(stats_after["LeadDay"], stats_after["q3"], linestyle=':', color='blue', linewidth=1.0, alpha=0.5)
        ax.vlines(stats_after["LeadDay"], stats_after["q1"], stats_after["q3"], color='black', linestyle='-', linewidth=1.5, zorder=1)
        for _, row in stats_after.iterrows():
            lead = row["LeadDay"]
            ax.annotate(f"{row['mean']:.2f}", (lead, row['mean']), textcoords="offset points", xytext=(-15,-20), ha='center', color='blue', fontsize=10)

    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_ylabel("Error of Water Level [m]" if ELEMENT.lower()=="wl" else "Error of Discharge [m³/s]", fontsize=14)
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    day_labels = [f"{i}st Day" if i==1 else f"{i}nd Day" if i==2 else f"{i}rd Day" if i==3 else f"{i}th Day" for i in range(1, MAX_LEAD_DAYS + 1)]
    ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
    ax.set_xticklabels(day_labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='both', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.tick_params(axis='both', which='both', direction='in', length=6, width=1)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches='tight')
    plt.show()
    print(f"Combined error plot saved: {save_path}")

def create_metrics_plot(metrics_df, save_path):
    """Create metrics comparison plot."""
    mp = metrics_df.replace({None: np.nan, float("-inf"): np.nan})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # NSE
    axs[0,0].plot(mp["Lead"], mp["NSE_Org"], marker="o", label=rain, color='red', linewidth=1, markersize=5)
    axs[0,0].plot(mp["Lead"], mp["NSE_Best"], marker="s", label="PF-DA", color='blue', linewidth=1, markersize=5)
    axs[0,0].set_title("Nash-Sutcliffe Efficiency vs Lead Day", fontsize=12, weight='bold')
    axs[0,0].set_xlabel("Lead Day", fontsize=12); axs[0,0].set_ylabel("NSE", fontsize=12)
    axs[0,0].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[0,0].grid(True, alpha=0.4); axs[0,0].legend(fontsize=10)

    # KGE
    axs[0,1].plot(mp["Lead"], mp["KGE_Org"], marker="o", label=rain, color='red', linewidth=1, markersize=5)
    axs[0,1].plot(mp["Lead"], mp["KGE_Best"], marker="s", label="PF-DA", color='blue', linewidth=1, markersize=5)
    axs[0,1].set_title("Kling-Gupta Efficiency vs Lead Day", fontsize=12, weight='bold')
    axs[0,1].set_xlabel("Lead Day", fontsize=12); axs[0,1].set_ylabel("KGE", fontsize=12)
    axs[0,1].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[0,1].grid(True, alpha=0.4); axs[0,1].legend(fontsize=10)

    # RMSE
    axs[1,0].plot(mp["Lead"], mp["RMSE_Org"], marker="o", label=rain, color='red', linewidth=1, markersize=5)
    axs[1,0].plot(mp["Lead"], mp["RMSE_Best"], marker="s", label="PF-DA", color='blue', linewidth=1, markersize=5)
    axs[1,0].set_title("Root Mean Square Error vs Lead Day", fontsize=12, weight='bold')
    axs[1,0].set_xlabel("Lead Day", fontsize=12); axs[1,0].set_ylabel("RMSE", fontsize=12)
    axs[1,0].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[1,0].grid(True, alpha=0.4); axs[1,0].legend(fontsize=10)

    # PBIAS (replaced Bias)
    axs[1,1].plot(mp["Lead"], mp["PBIAS_Org"], marker="o", label=rain, color='red', linewidth=1, markersize=5)
    axs[1,1].plot(mp["Lead"], mp["PBIAS_Best"], marker="s", label="PF-DA", color='blue', linewidth=1, markersize=5)
    axs[1,1].set_title("Percent Bias vs Lead Day", fontsize=12, weight='bold')
    axs[1,1].set_xlabel("Lead Day", fontsize=12); axs[1,1].set_ylabel("PBIAS [%]", fontsize=12)
    axs[1,1].set_xticks(range(1, MAX_LEAD_DAYS + 1)); axs[1,1].grid(True, alpha=0.4); axs[1,1].legend(fontsize=10)

    plt.suptitle(f'Forecast Performance Metrics: Before {rain} vs After PF-DA, {ELEMENT.upper()}, N1, {date}',
                 fontsize=14, weight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches='tight')
    plt.show()
    print(f"Metrics plot saved: {save_path}")

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    # Set paths based on element type
    if ELEMENT.lower() == "wl":
        base_path = BASE_PATH_WL; obs_file = OBS_FILE_WL; out_dir = OUT_DIR_WL
    else:
        base_path = BASE_PATH_Q;  obs_file = OBS_FILE_Q;  out_dir = OUT_DIR_Q

    os.makedirs(out_dir, exist_ok=True)

    print("Loading observations...")
    try:
        obs_df = pd.read_csv(obs_file, dtype={"Datetime": str}).dropna()
        if "Obs" in obs_df.columns:
            obs_series = obs_df.set_index("Datetime")["Obs"].astype(float)
        else:
            col = obs_df.select_dtypes(include="number").columns[0]
            obs_series = obs_df.set_index("Datetime")[col].astype(float)

        if not obs_series.index.is_unique:
            obs_series = obs_series.groupby(level=0).mean()

        if DATE_START or DATE_END:
            obs_series = obs_series.loc[[ts for ts in obs_series.index if _in_window(ts)]]

        print(f"Loaded {len(obs_series)} observation points")

        print("Processing OrgP (Before DA) and BestP (After DA) data...")
        error_df_before, error_df_after, raw_df = process_forecast_data(base_path, obs_series, DATE_START, DATE_END)
        print(f"Before DA (OrgP): {len(error_df_before)} error points")
        print(f"After  DA (BestP): {len(error_df_after)} error points")
        print(f"Raw data points for metrics: {len(raw_df)}")

        if DATE_START or DATE_END:
            print(f"Date window applied: start={DATE_START or '-inf'}, end={DATE_END or '+inf'}")
        else:
            print("Date window: not applied (full range).")

        title_combined = f"Forecasting Error Comparison: Before ({rain}) vs After PF-DA, {ELEMENT.upper()}, N1, {date}"
        save_path_combined = os.path.join(out_dir, f"forecast_error_{ELEMENT}.png")
        create_combined_error_plot(error_df_before, error_df_after, title_combined, save_path_combined)

        print("\nComputing metrics...")
        metrics_df = compute_metrics(raw_df)
        print("Metrics (rounded preview):")
        print(metrics_df.round(3))

        metrics_plot_path = os.path.join(out_dir, f"forecast_metrics_{ELEMENT}.png")
        create_metrics_plot(metrics_df, metrics_plot_path)

        excel_path = os.path.join(out_dir, f"forecast_evaluation_{ELEMENT}.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)
            raw_df.to_excel(writer,    sheet_name="Raw_Data",        index=False)
            error_df_before.to_excel(writer, sheet_name="Errors_Before_DA", index=False)
            error_df_after.to_excel(writer,  sheet_name="Errors_After_DA",  index=False)
        print(f"Excel file saved: {excel_path}")

    except Exception as e:
        print(f"Error processing forecast data: {e}")

    print("\nAll tasks completed!")
    print("\nGenerated files:")
    print(f"- Combined Annotated Line Error Plot: forecast_error_plot_{ELEMENT}.png")
    print(f"- Metrics Comparison: forecast_metrics_comparison_{ELEMENT}.png")
    print(f"- Excel Summary: forecast_evaluation_{ELEMENT}.xlsx")


#%%

#(OrgP only)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
ELEMENT = "wl"  # "wl" (waterlevel) or "qr" (discharge)

BASE_DIR = r"C:/Users/bikra/Desktop/DA_PF_RRI/without_DA/daily/2022/nan_wl_00"

# Water Level paths
BASE_PATH_WL = os.path.join(BASE_DIR, "Results")
OBS_FILE_WL  = os.path.join(BASE_DIR, "ObsData/WaterLevel/ObsWL.csv")
OUT_DIR_WL   = os.path.join(BASE_DIR, "Results/error_plots_orgp")

# Discharge paths
BASE_PATH_Q = os.path.join(BASE_DIR, "Results")
OBS_FILE_Q  = os.path.join(BASE_DIR, "ObsData/Discharge/ObsQ.csv")
OUT_DIR_Q   = os.path.join(BASE_DIR, "Results/error_plots_orgp")

MAX_LEAD_DAYS = 10

# =========================
# HELPER FUNCTIONS
# =========================
def load_value_series(csv_path):
    """Load CSV and return a Series indexed by Datetime."""
    df = pd.read_csv(csv_path, dtype={"Datetime": str}).dropna()
    df = df.set_index("Datetime")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError(f"No numeric columns in {csv_path}")
    ser = df[num_cols[0]].astype(float)
    if not ser.index.is_unique:
        ser = ser.groupby(level=0).mean()
    return ser

def compute_lead_day(init_str, ts_str):
    """Compute lead day from init and target timestamps."""
    t0 = datetime.strptime(init_str, "%Y%m%d%H%M")
    t1 = datetime.strptime(ts_str, "%Y%m%d%H%M")
    delta_days = (t1 - t0).total_seconds() / 86400.0
    if delta_days <= 0:
        return None
    lead = int(np.ceil(delta_days))
    return lead if 1 <= lead <= MAX_LEAD_DAYS else None

def process_orgp_data(base_path, obs_series):
    """Process OrgP forecast data and return error DataFrame and raw data."""
    org_files = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            if fname.endswith(".csv") and ("OrgP" in fname) and (f"_{ELEMENT.lower()}_" in fname):
                org_files.append(os.path.join(root, fname))

    org_dict = {os.path.basename(f).split("_")[0]: f for f in org_files}
    print(f"Found {len(org_dict)} OrgP forecast cycles.")

    org_errors, raw_rows = [], []
    for init in sorted(org_dict):
        try:
            org_ser = load_value_series(org_dict[init])
            common_ts = sorted(set(obs_series.index) & set(org_ser.index))
            for ts in common_ts:
                lead = compute_lead_day(init, ts)
                if lead is None:
                    continue
                obs_val  = float(obs_series.loc[ts])
                org_val  = float(org_ser.loc[ts])
                org_error = org_val - obs_val

                org_errors.append({
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Error": org_error
                })

                raw_rows.append({
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Obs": obs_val,
                    "OrgP": org_val
                })
        except Exception as e:
            print(f"Skipping cycle {init} due to error: {e}")

    return pd.DataFrame(org_errors), pd.DataFrame(raw_rows)

def compute_orgp_metrics(raw_df):
    """Compute NSE, KGE, RMSE, Bias metrics for OrgP per lead day."""
    def compute_nse(obs, sim):
        obs, sim = np.array(obs), np.array(sim)
        if obs.size == 0: return None
        if np.all(obs == obs[0]):
            return 1.0 if np.allclose(sim, obs) else float("-inf")
        return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)

    def compute_kge(obs, sim):
        obs, sim = np.array(obs), np.array(sim)
        if obs.size < 2: return None
        mu_o, mu_s = np.mean(obs), np.mean(sim)
        std_o, std_s = np.std(obs, ddof=0), np.std(sim, ddof=0)
        if std_o == 0 or std_s == 0:
            r = 1.0 if np.allclose(sim, obs) else 0.0
        else:
            r = np.corrcoef(obs, sim)[0, 1]
        alpha = std_s / std_o if std_o != 0 else np.nan
        beta  = mu_s / mu_o if mu_o != 0 else np.nan
        if np.isnan(r) or np.isnan(alpha) or np.isnan(beta): return None
        return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

    def compute_rmse(obs, sim):
        obs, sim = np.array(obs), np.array(sim)
        return None if obs.size == 0 else float(np.sqrt(np.mean((sim - obs)**2)))

    def compute_bias(obs, sim):
        obs, sim = np.array(obs), np.array(sim)
        return None if obs.size == 0 else float(np.mean(sim - obs))

    metrics_records = []
    for lead in range(1, MAX_LEAD_DAYS + 1):
        df_lead = raw_df[raw_df["LeadDay"] == lead]
        obs_vals = df_lead["Obs"].to_numpy()
        org_vals = df_lead["OrgP"].to_numpy()
        metrics_records.append({
            "Lead": lead,
            "NSE_Org": compute_nse(obs_vals, org_vals),
            "KGE_Org": compute_kge(obs_vals, org_vals),
            "RMSE_Org": compute_rmse(obs_vals, org_vals),
            "Bias_Org": compute_bias(obs_vals, org_vals)
        })
    return pd.DataFrame(metrics_records)

def create_orgp_error_plot(error_df, title, save_path):
    """Create OrgP error plot with mean, Q1, Q3 lines, IQR bars, and annotations."""
    plot_data = error_df[error_df["LeadDay"] >= 1].copy()
    stats = plot_data.groupby("LeadDay")["Error"].agg(
        mean='mean',
        q1=lambda x: np.quantile(x, 0.25),
        q3=lambda x: np.quantile(x, 0.75)
    ).reset_index()

    if stats.empty:
        print("No data for OrgP error plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats["LeadDay"], stats["mean"], "o-", color="red", lw=1.5, label="Mean")
    ax.plot(stats["LeadDay"], stats["q1"], "_--", color="red", lw=1.2, alpha=0.8, label="Q1")
    ax.plot(stats["LeadDay"], stats["q3"], "^--", color="red", lw=1.2, alpha=0.8, label="Q3")
    ax.vlines(stats["LeadDay"], stats["q1"], stats["q3"], color="black", lw=1.2)

    # Annotations
    for _, row in stats.iterrows():
        lead = row["LeadDay"]
        ax.annotate(f"{row['mean']:.2f}", (lead, row['mean']), textcoords="offset points", xytext=(-15,10), ha='center', color='red', fontsize=12)
        ax.annotate(f"{row['q1']:.2f}", (lead, row['q1']), textcoords="offset points", xytext=(-15,5), ha='center', color='red', fontsize=12)
        ax.annotate(f"{row['q3']:.2f}", (lead, row['q3']), textcoords="offset points", xytext=(15,5), ha='center', color='red', fontsize=12)

    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_ylabel("Error of Water Level [m]" if ELEMENT.lower()=="wl" else "Error of Discharge [m³/s]", fontsize=14)
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
    day_labels = [f"{i}st Day" if i==1 else f"{i}nd Day" if i==2 else f"{i}rd Day" if i==3 else f"{i}th Day" for i in range(1, MAX_LEAD_DAYS+1)]
    ax.set_xticklabels(day_labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches='tight')
    plt.show()
    print(f"OrgP error plot saved: {save_path}")

def create_orgp_metrics_plot(metrics_df, save_path):
    """Create OrgP metrics plot."""
    mp = metrics_df.replace({None: np.nan, float("-inf"): np.nan})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0,0].plot(mp["Lead"], mp["NSE_Org"], "o-", color="red"); axs[0,0].set_title("NSE vs Lead Day", weight='bold'); axs[0,0].grid(True, alpha=0.4); axs[0,0].set_ylabel("NSE", fontsize=12)
    axs[0,1].plot(mp["Lead"], mp["KGE_Org"], "o-", color="red"); axs[0,1].set_title("KGE vs Lead Day", weight='bold'); axs[0,1].grid(True, alpha=0.4); axs[0,1].set_ylabel("KGE", fontsize=12)
    axs[1,0].plot(mp["Lead"], mp["RMSE_Org"], "o-", color="red"); axs[1,0].set_title("RMSE vs Lead Day", weight='bold'); axs[1,0].grid(True, alpha=0.4); axs[1,0].set_ylabel("RMSE", fontsize=12)
    axs[1,1].plot(mp["Lead"], mp["Bias_Org"], "o-", color="red"); axs[1,1].set_title("Bias vs Lead Day", weight='bold'); axs[1,1].grid(True, alpha=0.4); axs[1,1].set_ylabel("Bias", fontsize=12)
    for ax in axs.ravel():
        ax.set_xlabel("Lead Day"); ax.set_xticks(range(1, MAX_LEAD_DAYS+1))
    plt.suptitle(f"Forecast Performance Metrics (GFS vs RID) — {ELEMENT.upper()}, 2024", fontsize=14, weight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches='tight')
    plt.show()
    print(f"OrgP metrics plot saved: {save_path}")


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    if ELEMENT.lower() == "wl":
        base_path, obs_file, out_dir = BASE_PATH_WL, OBS_FILE_WL, OUT_DIR_WL
    else:
        base_path, obs_file, out_dir = BASE_PATH_Q, OBS_FILE_Q, OUT_DIR_Q

    os.makedirs(out_dir, exist_ok=True)

    print("Loading observations...")
    obs_df = pd.read_csv(obs_file, dtype={"Datetime": str}).dropna()
    if "Obs" in obs_df.columns:
        obs_series = obs_df.set_index("Datetime")["Obs"].astype(float)
    else:
        col = obs_df.select_dtypes(include="number").columns[0]
        obs_series = obs_df.set_index("Datetime")[col].astype(float)
    if not obs_series.index.is_unique:
        obs_series = obs_series.groupby(level=0).mean()
    print(f"Loaded {len(obs_series)} observation points")

    # Process OrgP only
    error_df_orgp, raw_df_orgp = process_orgp_data(base_path, obs_series)
    print(f"OrgP errors: {len(error_df_orgp)} | raw data: {len(raw_df_orgp)}")

    # Error plot
    title = f"Forecasting Error Plot (GFS vs RID) — {ELEMENT.upper()}, 2024"
    save_error = os.path.join(out_dir, f"forecast_error_orgp_{ELEMENT}.png")
    create_orgp_error_plot(error_df_orgp, title, save_error)

    # Metrics
    print("\nComputing OrgP metrics...")
    metrics_df = compute_orgp_metrics(raw_df_orgp)
    print(metrics_df.round(3))
    save_metrics = os.path.join(out_dir, f"forecast_metrics_orgp_{ELEMENT}.png")
    create_orgp_metrics_plot(metrics_df, save_metrics)

    # Excel export
    excel_path = os.path.join(out_dir, f"forecast_evaluation_orgp_{ELEMENT}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        metrics_df.to_excel(writer, sheet_name="OrgP_Metrics", index=False)
        raw_df_orgp.to_excel(writer, sheet_name="OrgP_Raw", index=False)
        error_df_orgp.to_excel(writer, sheet_name="OrgP_Errors", index=False)
    print(f"Excel saved: {excel_path}")

    print("\nAll tasks completed (OrgP only)!")

#%%

#KGE

# ============================================================
# Taylor Diagram (Reference-Style) – Multi-Scenario
# Generates FOUR plots:
#   1) GFS
#   2) GFS + BC-GFS
#   3) GFS + BC-GFS + GFS(PF-DA)
#   4) ALL (GFS, BC-GFS, GFS(PF-DA), BC-GFS(PF-DA))
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------- USER CONFIG --------------------------

ELEMENT       = "wl"      # "wl" or "qr"
MAX_LEAD_DAYS = 10

# Optional date window (inclusive)
#DATE_START = "202407100000"
#DATE_END   = "202409200000"
DATE_START = None
DATE_END   = None

BASE_DIR = r"C:/Users/bikra/Desktop/Master's Thesis/Objective 2 and 3/latest/02. Bias_Forecast/Forecast/GFS_16_0.25_2025"
OUT_DIR  = os.path.join(BASE_DIR, "error_plots_wl_bc") if ELEMENT.lower() == "wl" \
           else os.path.join(BASE_DIR, "error_plots_spc_bc")
EXCEL_PATH = os.path.join(OUT_DIR, f"forecast_evaluation_GFS_vs_BCGFS_{ELEMENT}.xlsx")
PLOT_DIR = os.path.join(OUT_DIR, "Taylor_plot")
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------- HELPERS ------------------------------

def _parse_dt(s):
    return pd.to_datetime(s, format="%Y%m%d%H%M") if s else None

def load_raw(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    xls = pd.ExcelFile(path)
    gfs = pd.read_excel(xls, "Raw_Data_GFS");   gfs["Forcing"] = "GFS"
    bc  = pd.read_excel(xls, "Raw_Data_BCGFS"); bc["Forcing"]  = "BCGFS"
    df = pd.concat([gfs, bc], ignore_index=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    s, e = _parse_dt(DATE_START), _parse_dt(DATE_END)
    if s is not None: df = df[df["Datetime"] >= s]
    if e is not None: df = df[df["Datetime"] <= e]
    cols = ["Forcing","ForecastCycle","Datetime","LeadDay","Obs","OrgP","BestP"]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["Obs"])
    return df

def _valid(obs, sim):
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]

def taylor_metrics(obs, sim):
    y, yhat = _valid(obs, sim)
    if y.size < 2:
        return [np.nan]*5

    mu_o = float(np.mean(y))
    mu_s = float(np.mean(yhat))

    syo = float(np.std(y, ddof=1))
    sys = float(np.std(yhat, ddof=1))

    # ---- Compute KGE ----
    r = float(np.corrcoef(y, yhat)[0,1]) if syo>0 and sys>0 else np.nan
    alpha = sys / syo if syo > 0 else np.nan
    beta  = mu_s / mu_o if mu_o != 0 else np.nan

    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = float(1.0 - np.sqrt((r-1.0)**2 + (alpha-1.0)**2 + (beta-1.0)**2))
    else:
        kge = np.nan

    crmsd = float(np.sqrt(np.mean(((y - mu_o) - (yhat - mu_s))**2)))
    bias  = float((yhat - y).mean())

    # IMPORTANT: return KGE instead of corr
    return syo, sys, kge, crmsd, bias

def collect_stats(df: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    recs = []
    scens = [
        ("GFS",   "OrgP",  "GFS"),
        ("GFS",   "BestP", "GFS(PF-DA)"),
        ("BCGFS", "OrgP",  "BC-GFS"),
        ("BCGFS", "BestP", "BC-GFS(PF-DA)"),
    ]
    for forcing, col, label in scens:
        sub = df[df["Forcing"] == forcing]
        for ld in range(1, max_lead+1):
            s = sub[sub["LeadDay"] == ld]
            if s.empty: continue
            syo, sys, corr, crmsd, bias = taylor_metrics(s["Obs"], s[col])
            recs.append(dict(scenario=label, forcing=forcing, kind=col, lead=ld,
                             std_obs=syo, std_sim=sys, corr=corr, crmsd=crmsd, bias=bias))
    return pd.DataFrame.from_records(recs)

# --------------------- BACKGROUND ----------------------------

def draw_background(ax, r_max=2.0):
    ref = 1.0
    ax.set_xticklabels([])

    # CRMSD contours (green)
    rs = np.linspace(0, r_max, 260)
    ts = np.linspace(0, np.pi/2, 260)
    RS, TS = np.meshgrid(rs, ts)
    RMS = np.sqrt(ref**2 + RS**2 - 2*ref*RS*np.cos(TS))
    cs = ax.contour(TS, RS, RMS, levels=[0.3, 0.5, 0.7, 1.0, 1.2, 1.5], colors="green", linewidths=0.6)
    plt.clabel(cs, fmt="%.1f", fontsize=12, inline=True, colors=["green"])

    # Correlation rays
    for c in [0.99,0.95,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10]:
        th = np.arccos(np.clip(c, -1, 1))
        ax.plot([th, th], [0, r_max], linestyle="dashed", color="black", linewidth=0.8, alpha=0.6)

    # Std-dev rings
    for v in [0.5, 1.0, 1.5]:
        ax.plot(np.linspace(0, np.pi/2, 300), np.full(300, v),
                color="darkgreen", linestyle="dotted", linewidth=1.0)
    ax.plot(np.linspace(0, np.pi/2, 300), np.full(300, 1.0),
            color="black", linewidth=1.5)

    # Reference observation
    ax.plot(0.02, 1.0, marker="*", color="black", markersize=20)

    # Frame and ticks
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_rticks([0.5, 1.0, 1.5])
    ax.set_rlabel_position(92)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(False)

    # Correlation ticks
    tick_vals = [round(v,2) for v in np.arange(0.10, 0.91, 0.10)] + [0.95, 0.99]
    for c in tick_vals:
        th = np.arccos(np.clip(c, -1, 1))
        ax.plot([th, th], [r_max-0.035, r_max], color="black", linewidth=1.0)
        ax.text(th, r_max+0.06, f"{c:.2f}", ha="center", va="bottom", fontsize=14, color="black")

    ax.text(np.deg2rad(40), r_max + 0.18, "KGE",
            ha="center", va="bottom", fontsize=14, color="black")

# --------------------- PLOTTING -----------------------------

COLOR_FOR = {"GFS":"red", "GFS(PF-DA)":"blue", "BC-GFS":"blue", "BC-GFS(PF-DA)":"navy"}
LEAD_MARK = {1:"o",2:"s",3:"^",4:"D",5:"v",6:"P",7:"X",8:"<",9:">",10:"*"}

def plot_taylor_subset(df_stats, element, out_dir, scenarios_to_include, title_text, filename_suffix):
    df = df_stats[df_stats["scenario"].isin(scenarios_to_include)].copy()
    if df.empty:
        print(f"[WARN] No data for {scenarios_to_include}")
        return

    df["r"] = df["std_sim"] / df["std_obs"]
    df["theta"] = np.arccos(np.clip(df["corr"], -1, 1))
    df = df[np.isfinite(df["r"]) & np.isfinite(df["theta"])]
    r_max = max(1.5, min(2.0, float(np.nanmax(df["r"]) * 1.15)))

    fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(12, 8), dpi=160)
    draw_background(ax, r_max=r_max)

    # Data points
    for _, row in df.iterrows():
        ax.plot(row["theta"], row["r"], linestyle="None",
                marker=LEAD_MARK.get(int(row["lead"]), "o"),
                markersize=12, markeredgewidth=0.8, markeredgecolor="white",
                color=COLOR_FOR.get(row["scenario"], "black"))

    # Reference obs handle
    obs_handle = Line2D([0],[0], marker="*", linestyle="None", markersize=12,
                        markerfacecolor="black", markeredgecolor="black", label="Observation")

    # Scenario legend
    scen_handles = []
    for scen in ["GFS","GFS(PF-DA)","BC-GFS","BC-GFS(PF-DA)"]:
        if scen in df["scenario"].unique():
            scen_handles.append(Line2D([0],[0], marker="o", linestyle="None",
                                       markersize=9, markerfacecolor=COLOR_FOR[scen],
                                       markeredgecolor=COLOR_FOR[scen], label=scen))
    scen_handles.append(obs_handle)

    # Lead legend
    lead_handles = []
    for ld in sorted(df["lead"].unique()):
        lead_handles.append(Line2D([0],[0], marker=LEAD_MARK.get(int(ld),"o"),
                                   linestyle="None", markersize=9, markerfacecolor="none",
                                   markeredgecolor="black", label=f"Day {int(ld)}"))

    leg1 = ax.legend(handles=scen_handles, title="Scenarios",
                     loc="center left", bbox_to_anchor=(0.92, 0.95),
                     frameon=False, fontsize=10, title_fontsize=12)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=lead_handles, title="Lead Days",
                     loc="center left", bbox_to_anchor=(1.0, 0.60),
                     frameon=False, fontsize=10, title_fontsize=12)

    ax.set_title(title_text, fontsize=16, pad=50)
    fig.text(0.46, -0.03, "Standard deviation (Normalized)", ha="center", va="bottom", fontsize=16)
    fig.text(0.13, 0.48, "Standard deviation (Normalized)", ha="center", va="center",
             rotation=90, fontsize=16)

    out_png = os.path.join(out_dir, f"{filename_suffix}_taylor.png")
    plt.tight_layout(rect=[0.04, 0.04, 0.82, 0.98])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"[TAYLOR] saved: {out_png}")

# --------------------- MAIN --------------------------------

def main():
    print(f"Reading: {EXCEL_PATH}")
    raw = load_raw(EXCEL_PATH)
    stats = collect_stats(raw, MAX_LEAD_DAYS)
    stats.to_csv(os.path.join(PLOT_DIR, f"taylor_ALL_leads_stats_{ELEMENT}.csv"), index=False)

    base = "Taylor Diagram – WL(N1, 2025)"

    # 1️⃣ GFS
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS"], f"{base}: GFS", "1.GFS_only")

    # 2️⃣ GFS + BC-GFS
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS","BC-GFS"], f"{base}: GFS vs BC-GFS", "2.GFS_vs_BCGFS")

    # 3️⃣ GFS + BC-GFS + GFS(PF-DA)
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS","BC-GFS","GFS(PF-DA)"],
        f"{base}: GFS, BC-GFS, and GFS(PF-DA)", "3.GFS_vs_BC_GFS_vs_GFS-PFDA")

    # 4️⃣ ALL
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS","GFS(PF-DA)","BC-GFS","BC-GFS(PF-DA)"],
        f"{base}: Before and After PF-DA", "4.ALL")

if __name__ == "__main__":
    main()
