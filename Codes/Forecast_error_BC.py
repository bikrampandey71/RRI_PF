
# -*- coding: utf-8 -*-

# ============================================================
# Forecast Error & Metrics Plotter — Styled Multi-Set Version
# - Generates 4 forecasting-error plots (with point labels, vertical day lines)
# - Generates 4 metrics plots (2×2: NSE, KGE, RMSE, PBIAS)
# - Saves raw data, error data, AND metrics + mean error summaries into Excel
# - Safe with missing subsets (skips gracefully)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

# Choose element type: "wl" (water level) or "qr" (discharge)
ELEMENT = "wl"

# ---- Base directory that contains Results and Results_BC ----
BASE_DIR = r"C:/Users/bikra/Desktop/DA_PF_RRI/with_DA/Daily/Metrics/2022/GFS_BC_single_2022"

# Water level paths (N1)
BASE_PATH_GFS_WL   = os.path.join(BASE_DIR, "Results")
BASE_PATH_BC_WL    = os.path.join(BASE_DIR, "Results_BC")
OBS_FILE_WL        = os.path.join(BASE_DIR, "ObsData/WaterLevel/ObsWL.csv")
OUT_DIR_WL         = os.path.join(BASE_DIR, "error_plots_wl_bc_org/error_loop_new")

# Discharge paths (N1)
BASE_PATH_GFS_Q    = os.path.join(BASE_DIR, "Results")
BASE_PATH_BC_Q     = os.path.join(BASE_DIR, "Results_BC")
OBS_FILE_Q         = os.path.join(BASE_DIR, "ObsData/Discharge/ObsQ.csv")
OUT_DIR_Q          = os.path.join(BASE_DIR, "error_plots_spc_bc/error_loop")

# Maximum forecast lead time in days
MAX_LEAD_DAYS = 10

# --- Optional date window (inclusive). Use 'YYYYMMDDHHMM' or set to None.
#DATE_START = "202207010000"
#DATE_END   = "202209100000"
DATE_START = None
DATE_END   = None


# =========================
# HELPER FUNCTIONS
# =========================

def _in_window(ts_str: str) -> bool:
    """Check if timestamp string is within the inclusive window."""
    if DATE_START and ts_str < DATE_START:
        return False
    if DATE_END and ts_str > DATE_END:
        return False
    return True

def _cycle_overlaps_window(index_as_str) -> bool:
    """True if at least one timestamp in a series index is within the window."""
    if not DATE_START and not DATE_END:
        return True
    for ts in index_as_str:
        if _in_window(ts):
            return True
    return False

def load_value_series(csv_path: str) -> pd.Series:
    """
    Load a CSV with columns [Datetime, <numeric>] and return the first numeric
    column as a Series indexed by Datetime (string).
    """
    try:
        df = pd.read_csv(csv_path, dtype={"Datetime": str}).dropna()
        df = df.set_index("Datetime")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError(f"No numeric columns in {csv_path}")
        ser = df[num_cols[0]].astype(float)
        if not ser.index.is_unique:
            ser = ser.groupby(level=0).mean()
        return ser
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.Series(dtype=float)

def compute_lead_day(init_str: str, ts_str: str):
    """
    Lead day = ceil((timestamp - init)/1 day).
    Returns None if lead <=0 or > MAX_LEAD_DAYS.
    """
    t0 = datetime.strptime(init_str, "%Y%m%d%H%M")
    t1 = datetime.strptime(ts_str, "%Y%m%d%H%M")
    d_days = (t1 - t0).total_seconds() / 86400.0
    if d_days <= 0:
        return None
    ld = int(np.ceil(d_days))
    return ld if 1 <= ld <= MAX_LEAD_DAYS else None

def process_forecast_data(base_path: str, obs_series: pd.Series, forcing_label: str):
    """
    Process one forcing dataset (RAW GFS or BC-GFS).

    Returns:
        errors_org  : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Error]
        errors_best : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Error]
        raw_df      : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Obs, OrgP, BestP]
    """
    org_files, best_files = [], []

    for root, _, files in os.walk(base_path):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            full = os.path.join(root, fname)
            if "OrgP" in fname and f"_{ELEMENT.lower()}_" in fname:
                org_files.append(full)
            elif "BestP" in fname and f"_{ELEMENT.lower()}_" in fname:
                best_files.append(full)

    org_dict  = {os.path.basename(f).split("_")[0]: f for f in org_files}
    best_dict = {os.path.basename(f).split("_")[0]: f for f in best_files}

    common_inits = sorted(set(org_dict) & set(best_dict))
    print(f"[{forcing_label}] cycles with OrgP & BestP: {len(common_inits)}")

    errors_org, errors_best, raw_rows = [], [], []

    for init in common_inits:
        try:
            org_ser  = load_value_series(org_dict[init])
            best_ser = load_value_series(best_dict[init])

            if not _cycle_overlaps_window(org_ser.index) and not _cycle_overlaps_window(best_ser.index):
                continue

            common_ts = sorted(set(obs_series.index) & set(org_ser.index) & set(best_ser.index))
            common_ts = [ts for ts in common_ts if _in_window(ts)]

            for ts in common_ts:
                lead = compute_lead_day(init, ts)
                if lead is None:
                    continue
                obs = float(obs_series.loc[ts])
                org = float(org_ser.loc[ts])
                bes = float(best_ser.loc[ts])

                errors_org.append({
                    "Forcing": forcing_label,
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Error": org - obs
                })
                errors_best.append({
                    "Forcing": forcing_label,
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Error": bes - obs
                })
                raw_rows.append({
                    "Forcing": forcing_label,
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Obs": obs,
                    "OrgP": org,
                    "BestP": bes
                })
        except Exception as e:
            print(f"[{forcing_label}] skip {init}: {e}")

    return pd.DataFrame(errors_org), pd.DataFrame(errors_best), pd.DataFrame(raw_rows)


# =========================
# METRICS
# =========================

def compute_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NSE, KGE, RMSE, PBIAS for each lead day per forcing (Org vs Best)."""
    def nse(o, s):
        o = np.asarray(o); s = np.asarray(s)
        if o.size == 0 or np.isclose(np.var(o), 0):
            return None
        return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

    def kge(o, s):
        o = np.asarray(o); s = np.asarray(s)
        if o.size < 2:
            return None
        r = np.corrcoef(o, s)[0, 1] if np.std(o) > 0 and np.std(s) > 0 else 0.0
        alpha = np.std(s) / np.std(o) if np.std(o) != 0 else np.nan
        beta  = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
        if np.isnan(alpha) or np.isnan(beta):
            return None
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    def rmse(o, s):
        o = np.asarray(o); s = np.asarray(s)
        return float(np.sqrt(np.mean((s - o) ** 2))) if o.size > 0 else None

    def pbias(o, s):
        o = np.asarray(o); s = np.asarray(s)
        return float(100.0 * np.sum(s - o) / np.sum(o)) if o.size > 0 and not np.isclose(np.sum(o), 0) else None

    rec = []
    for forcing, sub in raw_df.groupby("Forcing"):
        for lead in range(1, MAX_LEAD_DAYS + 1):
            d = sub[sub["LeadDay"] == lead]
            o = d["Obs"].to_numpy()
            op = d["OrgP"].to_numpy()
            bp = d["BestP"].to_numpy()
            rec.append({
                "Forcing": forcing, "Lead": lead,
                "NSE_Org": nse(o, op), "NSE_Best": nse(o, bp),
                "KGE_Org": kge(o, op), "KGE_Best": kge(o, bp),
                "RMSE_Org": rmse(o, op), "RMSE_Best": rmse(o, bp),
                "PBIAS_Org": pbias(o, op), "PBIAS_Best": pbias(o, bp),
            })
    return pd.DataFrame(rec)


# =========================
# PLOTTING — FORECAST ERROR (mean only)
# =========================

def _agg_mean(df):
    """Mean error per lead day; returns empty with right columns if df empty."""
    if df.empty:
        return pd.DataFrame(columns=["LeadDay", "mean"])
    return (
        df.groupby("LeadDay", dropna=True)["Error"]
          .mean()
          .reset_index()
          .rename(columns={"Error": "mean"})
    )

def _draw_vertical_day_lines(ax, x_min=1, x_max=10, alpha=0.85):
    """Vertical dashed lines at each day across slightly expanded y-range."""
    ymin, ymax = ax.get_ylim()
    margin = 0.12 * (ymax - ymin)
    y0 = ymin - margin
    y1 = ymax + margin
    for x in range(x_min, x_max + 1):
        ax.vlines(x, y0, y1, color="black", linestyle="--", linewidth=1.0, alpha=alpha)
    ax.set_ylim(y0, y1)

def _day_labels():
    labs = []
    for i in range(1, MAX_LEAD_DAYS + 1):
        if i == 1: labs.append("1st Day")
        elif i == 2: labs.append("2nd Day")
        elif i == 3: labs.append("3rd Day")
        else: labs.append(f"{i}th Day")
    return labs

def _annotate_series(ax, df, color, offset):
    """Add numeric labels at each point with a small, consistent offset (points)."""
    if df.empty:
        return
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['mean']:.2f}",
            (row["LeadDay"], row["mean"]),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            color=color,
            fontsize=12
        )

def create_forecast_plot(eog, ebg, eob, ebb, include, title, save_path):
    """
    Styled forecast error plot. 'include' is a list among:
    ["GFS", "GFS(PF-DA)", "BC-GFS", "BC-GFS(PF-DA)"]
    """

    # aggregate
    gfs_org  = _agg_mean(eog)
    gfs_best = _agg_mean(ebg)
    bc_org   = _agg_mean(eob)
    bc_best  = _agg_mean(ebb)

    # if truly everything empty, skip
    if all(d.empty for d in [gfs_org, gfs_best, bc_org, bc_best]):
        print(f"[WARN] No data for {os.path.basename(save_path)} — skipped.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # stylistic plotting (lines + markers)
    if "GFS" in include and not gfs_org.empty:
        ax.plot(gfs_org["LeadDay"], gfs_org["mean"], "-o", color="red", lw=1.5, ms=4, label="GFS")
        _annotate_series(ax, gfs_org, "red", offset=(-20, 10))

    if "GFS(PF-DA)" in include and not gfs_best.empty:
        ax.plot(gfs_best["LeadDay"], gfs_best["mean"], "--s", color="red", lw=1.5, ms=4, label="GFS(PF-DA)")
        _annotate_series(ax, gfs_best, "red", offset=(20, 10))

    if "BC-GFS" in include and not bc_org.empty:
        ax.plot(bc_org["LeadDay"], bc_org["mean"], "-o", color="blue", lw=1.5, ms=4, label="BC-GFS")
        _annotate_series(ax, bc_org, "blue", offset=(-20, -15))

    if "BC-GFS(PF-DA)" in include and not bc_best.empty:
        ax.plot(bc_best["LeadDay"], bc_best["mean"], "--s", color="blue", lw=1.5, ms=4, label="BC-GFS(PF-DA)")
        _annotate_series(ax, bc_best, "blue", offset=(20, -15))

    # axes, grid, verticals
    ax.set_title(title, fontsize=14, weight="bold", pad=20)
    ax.set_ylabel(
        "Forecast Error of Water Level [m]" if ELEMENT.lower() == "wl"
        else "Forecast Error of Discharge [m³/s]",
        fontsize=14
    )
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
    ax.set_xticklabels(_day_labels(), fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="both", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.tick_params(axis="both", which="both", direction="in", length=6, width=1)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    # vertical day lines (like your reference)
    _draw_vertical_day_lines(ax, 1, MAX_LEAD_DAYS, alpha=0.9)

    ax.legend(loc="upper left", fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[PLOT] {os.path.basename(save_path)} saved.")


# =========================
# PLOTTING — METRICS 2×2 (styled like your 4th plot)
# =========================

def create_metrics_plot(metrics_df: pd.DataFrame, include, title, save_path):
    """
    2×2 panel plot: NSE, KGE, RMSE, PBIAS.
    'include' controls which series appear (same strings as forecast plot).
    """
    if metrics_df.empty:
        print(f"[WARN] Empty metrics dataset for {os.path.basename(save_path)}")
        return

    gfs   = metrics_df[metrics_df["Forcing"] == "GFS"]
    bcgfs = metrics_df[metrics_df["Forcing"] == "BCGFS"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    def draw_one(ax, met, title_txt):
        # GFS
        if "GFS" in include and not gfs.empty and f"{met}_Org" in gfs.columns:
            ax.plot(gfs["Lead"], gfs[f"{met}_Org"], "-o", color="red", lw=1.2, ms=3, label="GFS")
        if "GFS(PF-DA)" in include and not gfs.empty and f"{met}_Best" in gfs.columns:
            ax.plot(gfs["Lead"], gfs[f"{met}_Best"], "--s", color="red", lw=1.2, ms=3, label="GFS(PF-DA)")
        # BC-GFS
        if "BC-GFS" in include and not bcgfs.empty and f"{met}_Org" in bcgfs.columns:
            ax.plot(bcgfs["Lead"], bcgfs[f"{met}_Org"], "-o", color="blue", lw=1.2, ms=3, label="BC-GFS")
        if "BC-GFS(PF-DA)" in include and not bcgfs.empty and f"{met}_Best" in bcgfs.columns:
            ax.plot(bcgfs["Lead"], bcgfs[f"{met}_Best"], "--s", color="blue", lw=1.2, ms=3, label="BC-GFS(PF-DA)")

        ax.set_title(title_txt, fontsize=12, weight="bold")
        ax.set_xlabel("Lead Day", fontsize=12)
        ax.set_ylabel(met, fontsize=12)
        ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=10)

    draw_one(axs[0, 0], "NSE",   "Nash-Sutcliffe Efficiency vs Lead Day")
    draw_one(axs[0, 1], "KGE",   "Kling-Gupta Efficiency vs Lead Day")
    draw_one(axs[1, 0], "RMSE",  "Root Mean Square Error vs Lead Day")
    draw_one(axs[1, 1], "PBIAS", "Percent Bias vs Lead Day")

    plt.suptitle(title, fontsize=14, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[METRIC] {os.path.basename(save_path)} saved.")


# =========================
# MULTI-SET DRIVER
# =========================

def run_all_sets(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                 raw_gfs, raw_bc, element, out_dir):

    # --- Base titles for consistency ---
    base_error   = f"Forecast Mean Error – {element.upper()}(N1, 2022)"
    base_metrics = f"Forecast Metrics – {element.upper()}(N1, 2022)"

    # --- Define all plot sets ---
    sets = [
        # tag, include list
        ("GFS_only",
         ["GFS"],
         f"{base_error}: GFS",
         f"{base_metrics}: GFS"),

        ("GFS_vs_BCGFS",
         ["GFS", "BC-GFS"],
         f"{base_error}: GFS and BC-GFS",
         f"{base_metrics}: GFS and BC-GFS"),

        ("GFS_BCGFS_GFSPFDA",
         ["GFS", "BC-GFS", "GFS(PF-DA)"],
         f"{base_error}: GFS, BC-GFS, and GFS(PF-DA)",
         f"{base_metrics}: GFS, BC-GFS, and GFS(PF-DA)"),

        ("ALL",
         ["GFS", "BC-GFS", "GFS(PF-DA)", "BC-GFS(PF-DA)"],
         f"{base_error}: Before and After PF-DA",
         f"{base_metrics}: Before and After PF-DA"),
    ]

    for tag, include, err_title, met_title in sets:
        print(f"\n--- Generating {tag} ---")

        # Forecast error plot
        save_err = os.path.join(out_dir, f"forecast_error_{tag}_{element}.png")
        create_forecast_plot(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                             include, err_title, save_err)

        # Metrics: build raw subset needed
        raw = pd.DataFrame()
        if any(x in include for x in ["GFS", "GFS(PF-DA)"]):
            raw = pd.concat([raw, raw_gfs], ignore_index=True)
        if any(x in include for x in ["BC-GFS", "BC-GFS(PF-DA)"]):
            raw = pd.concat([raw, raw_bc], ignore_index=True)
        if not raw.empty:
            metrics = compute_metrics(raw)
            save_met = os.path.join(out_dir, f"forecast_metrics_{tag}_{element}.png")
            create_metrics_plot(metrics, include, met_title, save_met)
        else:
            print(f"[WARN] No raw data for metrics in {tag}.")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    # Resolve paths by element
    if ELEMENT.lower() == "wl":
        base_path_gfs = BASE_PATH_GFS_WL
        base_path_bc  = BASE_PATH_BC_WL
        obs_file      = OBS_FILE_WL
        out_dir       = OUT_DIR_WL
    else:
        base_path_gfs = BASE_PATH_GFS_Q
        base_path_bc  = BASE_PATH_BC_Q
        obs_file      = OBS_FILE_Q
        out_dir       = OUT_DIR_Q

    os.makedirs(out_dir, exist_ok=True)

    try:
        # Load observations
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

        print(f"Observations loaded: {len(obs_series)} points.")

        # Process both forcings
        print("\nProcessing RAW GFS (Results)...")
        errors_org_gfs, errors_best_gfs, raw_gfs = process_forecast_data(
            base_path_gfs, obs_series, forcing_label="GFS"
        )

        print("\nProcessing Bias-Corrected GFS (Results_BC)...")
        errors_org_bc, errors_best_bc, raw_bc = process_forecast_data(
            base_path_bc, obs_series, forcing_label="BCGFS"
        )

        if DATE_START or DATE_END:
            print(f"\nDate window applied: start={DATE_START or '-inf'}, end={DATE_END or '+inf'}")
        else:
            print("\nDate window: not applied (full range).")

        # Generate all plot sets
        run_all_sets(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                     raw_gfs, raw_bc, ELEMENT, out_dir)

        # ===== Metrics & Mean Error summaries for Excel =====
        raw_all = pd.concat([raw_gfs, raw_bc], ignore_index=True)
        metrics_all = compute_metrics(raw_all)

        # Mean forecast error summary (OrgP & BestP)
        raw_all_errors = raw_all.copy()
        raw_all_errors["Error_Org"] = raw_all_errors["OrgP"] - raw_all_errors["Obs"]
        raw_all_errors["Error_Best"] = raw_all_errors["BestP"] - raw_all_errors["Obs"]
        mean_error_all = (
            raw_all_errors
            .groupby(["Forcing", "LeadDay"])
            .agg(
                MeanError_Org=("Error_Org", "mean"),
                MeanError_Best=("Error_Best", "mean"),
            )
            .reset_index()
        )

        # Save an Excel summary for reuse (e.g., Taylor script)
        excel_path = os.path.join(out_dir, f"forecast_evaluation_GFS_vs_BCGFS_{ELEMENT}.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            # 1st sheet: metrics summary
            metrics_all.to_excel(writer, "Metrics_All", index=False)
            # 2nd sheet: mean forecast error summary
            mean_error_all.to_excel(writer, "MeanError_All", index=False)
            # Other sheets
            raw_gfs.to_excel(writer, "Raw_Data_GFS", index=False)
            raw_bc.to_excel(writer, "Raw_Data_BCGFS", index=False)
            errors_org_gfs.to_excel(writer, "Errors_GFS_OrgP", index=False)
            errors_best_gfs.to_excel(writer, "Errors_GFS_BestP", index=False)
            errors_org_bc.to_excel(writer, "Errors_BCGFS_OrgP", index=False)
            errors_best_bc.to_excel(writer, "Errors_BCGFS_BestP", index=False)

        print(f"\nExcel saved: {excel_path}")

    except Exception as e:
        print(f"\n[ERROR] {e}")

    print("\nDone.")


#%%

# -*- coding: utf-8 -*-

# ============================================================
# Forecast Error & Metrics Plotter — Styled Multi-Set Version
# - Generates 4 forecasting-error plots (with point labels, vertical day lines)
# - Generates 4 metrics plots (2×2: NSE, KGE, RMSE, PBIAS)
# - Safe with missing subsets (skips gracefully)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

# Choose element type: "wl" (water level) or "qr" (discharge)
ELEMENT = "wl"

# ---- Base directory that contains Results and Results_BC ----
BASE_DIR = r"C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/GFS_16_0.25_2022_edit"

# Water level paths (N1)
BASE_PATH_GFS_WL   = os.path.join(BASE_DIR, "Results")
BASE_PATH_BC_WL    = os.path.join(BASE_DIR, "Results_BC")
OBS_FILE_WL        = os.path.join(BASE_DIR, "ObsData/WaterLevel/ObsWL.csv")
OUT_DIR_WL         = os.path.join(BASE_DIR, "error_plots_wl_bc_org/error_loop")

# Discharge paths (N1)
BASE_PATH_GFS_Q    = os.path.join(BASE_DIR, "Results")
BASE_PATH_BC_Q     = os.path.join(BASE_DIR, "Results_BC")
OBS_FILE_Q         = os.path.join(BASE_DIR, "ObsData/Discharge/ObsQ.csv")
OUT_DIR_Q          = os.path.join(BASE_DIR, "error_plots_spc_bc/error_loop")

# Maximum forecast lead time in days
MAX_LEAD_DAYS = 10

# --- Optional date window (inclusive). Use 'YYYYMMDDHHMM' or set to None.
#DATE_START = "202207010000"
#DATE_END   = "202209100000"
DATE_START = None
DATE_END   = None


# =========================
# HELPER FUNCTIONS
# =========================

def _in_window(ts_str: str) -> bool:
    """Check if timestamp string is within the inclusive window."""
    if DATE_START and ts_str < DATE_START:
        return False
    if DATE_END and ts_str > DATE_END:
        return False
    return True

def _cycle_overlaps_window(index_as_str) -> bool:
    """True if at least one timestamp in a series index is within the window."""
    if not DATE_START and not DATE_END:
        return True
    for ts in index_as_str:
        if _in_window(ts):
            return True
    return False

def load_value_series(csv_path: str) -> pd.Series:
    """
    Load a CSV with columns [Datetime, <numeric>] and return the first numeric
    column as a Series indexed by Datetime (string).
    """
    try:
        df = pd.read_csv(csv_path, dtype={"Datetime": str}).dropna()
        df = df.set_index("Datetime")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError(f"No numeric columns in {csv_path}")
        ser = df[num_cols[0]].astype(float)
        if not ser.index.is_unique:
            ser = ser.groupby(level=0).mean()
        return ser
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.Series(dtype=float)

def compute_lead_day(init_str: str, ts_str: str):
    """
    Lead day = ceil((timestamp - init)/1 day).
    Returns None if lead <=0 or > MAX_LEAD_DAYS.
    """
    t0 = datetime.strptime(init_str, "%Y%m%d%H%M")
    t1 = datetime.strptime(ts_str, "%Y%m%d%H%M")
    d_days = (t1 - t0).total_seconds() / 86400.0
    if d_days <= 0:
        return None
    ld = int(np.ceil(d_days))
    return ld if 1 <= ld <= MAX_LEAD_DAYS else None

def process_forecast_data(base_path: str, obs_series: pd.Series, forcing_label: str):
    """
    Process one forcing dataset (RAW GFS or BC-GFS).

    Returns:
        errors_org  : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Error]
        errors_best : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Error]
        raw_df      : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Obs, OrgP, BestP]
    """
    org_files, best_files = [], []

    for root, _, files in os.walk(base_path):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            full = os.path.join(root, fname)
            if "OrgP" in fname and f"_{ELEMENT.lower()}_" in fname:
                org_files.append(full)
            elif "BestP" in fname and f"_{ELEMENT.lower()}_" in fname:
                best_files.append(full)

    org_dict  = {os.path.basename(f).split("_")[0]: f for f in org_files}
    best_dict = {os.path.basename(f).split("_")[0]: f for f in best_files}

    common_inits = sorted(set(org_dict) & set(best_dict))
    print(f"[{forcing_label}] cycles with OrgP & BestP: {len(common_inits)}")

    errors_org, errors_best, raw_rows = [], [], []

    for init in common_inits:
        try:
            org_ser  = load_value_series(org_dict[init])
            best_ser = load_value_series(best_dict[init])

            if not _cycle_overlaps_window(org_ser.index) and not _cycle_overlaps_window(best_ser.index):
                continue

            common_ts = sorted(set(obs_series.index) & set(org_ser.index) & set(best_ser.index))
            common_ts = [ts for ts in common_ts if _in_window(ts)]

            for ts in common_ts:
                lead = compute_lead_day(init, ts)
                if lead is None:
                    continue
                obs = float(obs_series.loc[ts])
                org = float(org_ser.loc[ts])
                bes = float(best_ser.loc[ts])

                errors_org.append({"Forcing": forcing_label, "ForecastCycle": init, "Datetime": ts, "LeadDay": lead, "Error": org - obs})
                errors_best.append({"Forcing": forcing_label, "ForecastCycle": init, "Datetime": ts, "LeadDay": lead, "Error": bes - obs})
                raw_rows.append({"Forcing": forcing_label, "ForecastCycle": init, "Datetime": ts, "LeadDay": lead, "Obs": obs, "OrgP": org, "BestP": bes})
        except Exception as e:
            print(f"[{forcing_label}] skip {init}: {e}")

    return pd.DataFrame(errors_org), pd.DataFrame(errors_best), pd.DataFrame(raw_rows)


# =========================
# METRICS
# =========================

def compute_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NSE, KGE, RMSE, PBIAS for each lead day per forcing (Org vs Best)."""
    def nse(o, s):
        o = np.asarray(o); s = np.asarray(s)
        if o.size == 0 or np.isclose(np.var(o), 0):
            return None
        return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

    def kge(o, s):
        o = np.asarray(o); s = np.asarray(s)
        if o.size < 2:
            return None
        r = np.corrcoef(o, s)[0, 1] if np.std(o) > 0 and np.std(s) > 0 else 0.0
        alpha = np.std(s) / np.std(o) if np.std(o) != 0 else np.nan
        beta  = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
        if np.isnan(alpha) or np.isnan(beta):
            return None
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    def rmse(o, s):
        o = np.asarray(o); s = np.asarray(s)
        return float(np.sqrt(np.mean((s - o) ** 2))) if o.size > 0 else None

    def pbias(o, s):
        o = np.asarray(o); s = np.asarray(s)
        return float(100.0 * np.sum(s - o) / np.sum(o)) if o.size > 0 and not np.isclose(np.sum(o), 0) else None

    rec = []
    for forcing, sub in raw_df.groupby("Forcing"):
        for lead in range(1, MAX_LEAD_DAYS + 1):
            d = sub[sub["LeadDay"] == lead]
            o = d["Obs"].to_numpy()
            op = d["OrgP"].to_numpy()
            bp = d["BestP"].to_numpy()
            rec.append({
                "Forcing": forcing, "Lead": lead,
                "NSE_Org": nse(o, op), "NSE_Best": nse(o, bp),
                "KGE_Org": kge(o, op), "KGE_Best": kge(o, bp),
                "RMSE_Org": rmse(o, op), "RMSE_Best": rmse(o, bp),
                "PBIAS_Org": pbias(o, op), "PBIAS_Best": pbias(o, bp),
            })
    return pd.DataFrame(rec)


# =========================
# PLOTTING — FORECAST ERROR (styled like your 3rd plot)
# =========================

def _agg_mean(df):
    """Mean error per lead day; returns empty with right columns if df empty."""
    if df.empty:
        return pd.DataFrame(columns=["LeadDay", "mean"])
    return (
        df.groupby("LeadDay", dropna=True)["Error"]
          .mean()
          .reset_index()
          .rename(columns={"Error": "mean"})
    )

def _draw_vertical_day_lines(ax, x_min=1, x_max=10, alpha=0.85):
    """Vertical dashed lines at each day across slightly expanded y-range."""
    ymin, ymax = ax.get_ylim()
    margin = 0.12 * (ymax - ymin)
    y0 = ymin - margin
    y1 = ymax + margin
    for x in range(x_min, x_max + 1):
        ax.vlines(x, y0, y1, color="black", linestyle="--", linewidth=1.0, alpha=alpha)
    ax.set_ylim(y0, y1)

def _day_labels():
    labs = []
    for i in range(1, MAX_LEAD_DAYS + 1):
        if i == 1: labs.append("1st Day")
        elif i == 2: labs.append("2nd Day")
        elif i == 3: labs.append("3rd Day")
        else: labs.append(f"{i}th Day")
    return labs

def _annotate_series(ax, df, color, offset):
    """Add numeric labels at each point with a small, consistent offset (points)."""
    if df.empty: return
    for _, row in df.iterrows():
        ax.annotate(f"{row['mean']:.2f}",
                    (row["LeadDay"], row["mean"]),
                    textcoords="offset points",
                    xytext=offset, ha="center",
                    color=color, fontsize=12)

def create_forecast_plot(eog, ebg, eob, ebb, include, title, save_path):
    """
    Styled forecast error plot. 'include' is a list among:
    ["GFS", "GFS(PF-DA)", "BC-GFS", "BC-GFS(PF-DA)"]
    """

    # aggregate
    gfs_org  = _agg_mean(eog)
    gfs_best = _agg_mean(ebg)
    bc_org   = _agg_mean(eob)
    bc_best  = _agg_mean(ebb)

    # if truly everything empty, skip
    if all(d.empty for d in [gfs_org, gfs_best, bc_org, bc_best]):
        print(f"[WARN] No data for {os.path.basename(save_path)} — skipped.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # stylistic plotting (lines + markers)
    if "GFS" in include and not gfs_org.empty:
        ax.plot(gfs_org["LeadDay"], gfs_org["mean"], "-o", color="red", lw=1.5, ms=4, label="GFS")
        _annotate_series(ax, gfs_org, "red", offset=(-20, 10))

    if "GFS(PF-DA)" in include and not gfs_best.empty:
        ax.plot(gfs_best["LeadDay"], gfs_best["mean"], "--s", color="red", lw=1.5, ms=4, label="GFS(PF-DA)")
        _annotate_series(ax, gfs_best, "red", offset=(20, 10))

    if "BC-GFS" in include and not bc_org.empty:
        ax.plot(bc_org["LeadDay"], bc_org["mean"], "-o", color="blue", lw=1.5, ms=4, label="BC-GFS")
        _annotate_series(ax, bc_org, "blue", offset=(-20, -15))

    if "BC-GFS(PF-DA)" in include and not bc_best.empty:
        ax.plot(bc_best["LeadDay"], bc_best["mean"], "--s", color="blue", lw=1.5, ms=4, label="BC-GFS(PF-DA)")
        _annotate_series(ax, bc_best, "blue", offset=(20, -15))

    # axes, grid, verticals
    ax.set_title(title, fontsize=14, weight="bold", pad=20)
    ax.set_ylabel("Forecast Error of Water Level [m]" if ELEMENT.lower()=="wl" else "Forecast Error of Discharge [m³/s]",
                  fontsize=14)
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
    ax.set_xticklabels(_day_labels(), fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="both", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.tick_params(axis="both", which="both", direction="in", length=6, width=1)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    # vertical day lines (like your reference)
    _draw_vertical_day_lines(ax, 1, MAX_LEAD_DAYS, alpha=0.9)

    ax.legend(loc="upper left", fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[PLOT] {os.path.basename(save_path)} saved.")


# =========================
# PLOTTING — METRICS 2×2 (styled like your 4th plot)
# =========================

def create_metrics_plot(metrics_df: pd.DataFrame, include, title, save_path):
    """
    2×2 panel plot: NSE, KGE, RMSE, PBIAS.
    'include' controls which series appear (same strings as forecast plot).
    """
    if metrics_df.empty:
        print(f"[WARN] Empty metrics dataset for {os.path.basename(save_path)}")
        return

    gfs   = metrics_df[metrics_df["Forcing"] == "GFS"]
    bcgfs = metrics_df[metrics_df["Forcing"] == "BCGFS"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    def draw_one(ax, met, title_txt):
        # GFS
        if "GFS" in include and not gfs.empty and f"{met}_Org" in gfs.columns:
            ax.plot(gfs["Lead"], gfs[f"{met}_Org"], "-o", color="red", lw=1.2, ms=3, label="GFS")
        if "GFS(PF-DA)" in include and not gfs.empty and f"{met}_Best" in gfs.columns:
            ax.plot(gfs["Lead"], gfs[f"{met}_Best"], "--s", color="red", lw=1.2, ms=3, label="GFS(PF-DA)")
        # BC-GFS
        if "BC-GFS" in include and not bcgfs.empty and f"{met}_Org" in bcgfs.columns:
            ax.plot(bcgfs["Lead"], bcgfs[f"{met}_Org"], "-o", color="blue", lw=1.2, ms=3, label="BC-GFS")
        if "BC-GFS(PF-DA)" in include and not bcgfs.empty and f"{met}_Best" in bcgfs.columns:
            ax.plot(bcgfs["Lead"], bcgfs[f"{met}_Best"], "--s", color="blue", lw=1.2, ms=3, label="BC-GFS(PF-DA)")

        ax.set_title(title_txt, fontsize=12, weight="bold")
        ax.set_xlabel("Lead Day", fontsize=12)
        ax.set_ylabel(met, fontsize=12)
        ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=10)

    draw_one(axs[0, 0], "NSE",   "Nash-Sutcliffe Efficiency vs Lead Day")
    draw_one(axs[0, 1], "KGE",   "Kling-Gupta Efficiency vs Lead Day")
    draw_one(axs[1, 0], "RMSE",  "Root Mean Square Error vs Lead Day")
    draw_one(axs[1, 1], "PBIAS", "Percent Bias vs Lead Day")

    plt.suptitle(title, fontsize=14, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[METRIC] {os.path.basename(save_path)} saved.")


# =========================
# MULTI-SET DRIVER
# =========================

def run_all_sets(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                 raw_gfs, raw_bc, element, out_dir):

    # --- Base titles for consistency ---
    base_error   = f"Forecast Mean Error – {element.upper()}(N1, 2022)"
    base_metrics = f"Forecast Metrics – {element.upper()}(N1, 2022)"

    # --- Define all plot sets ---
    sets = [
        # tag, include list
        ("GFS_only",
         ["GFS"],
         f"{base_error}: GFS",
         f"{base_metrics}: GFS"),

        ("GFS_vs_BCGFS",
         ["GFS", "BC-GFS"],
         f"{base_error}: GFS and BC-GFS",
         f"{base_metrics}: GFS and BC-GFS"),

        ("GFS_BCGFS_GFSPFDA",
         ["GFS", "BC-GFS", "GFS(PF-DA)"],
         f"{base_error}: GFS, BC-GFS, and GFS(PF-DA)",
         f"{base_metrics}: GFS, BC-GFS, and GFS(PF-DA)"),

        ("ALL",
         ["GFS", "BC-GFS", "GFS(PF-DA)", "BC-GFS(PF-DA)"],
         f"{base_error}: Before and After PF-DA",
         f"{base_metrics}: Before and After PF-DA"),
    ]

    for tag, include, err_title, met_title in sets:
        print(f"\n--- Generating {tag} ---")

        # Forecast error plot
        save_err = os.path.join(out_dir, f"forecast_error_{tag}_{element}.png")
        create_forecast_plot(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                             include, err_title, save_err)

        # Metrics: build raw subset needed
        raw = pd.DataFrame()
        if any(x in include for x in ["GFS", "GFS(PF-DA)"]):
            raw = pd.concat([raw, raw_gfs], ignore_index=True)
        if any(x in include for x in ["BC-GFS", "BC-GFS(PF-DA)"]):
            raw = pd.concat([raw, raw_bc], ignore_index=True)
        if not raw.empty:
            metrics = compute_metrics(raw)
            save_met = os.path.join(out_dir, f"forecast_metrics_{tag}_{element}.png")
            create_metrics_plot(metrics, include, met_title, save_met)
        else:
            print(f"[WARN] No raw data for metrics in {tag}.")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    # Resolve paths by element
    if ELEMENT.lower() == "wl":
        base_path_gfs = BASE_PATH_GFS_WL
        base_path_bc  = BASE_PATH_BC_WL
        obs_file      = OBS_FILE_WL
        out_dir       = OUT_DIR_WL
    else:
        base_path_gfs = BASE_PATH_GFS_Q
        base_path_bc  = BASE_PATH_BC_Q
        obs_file      = OBS_FILE_Q
        out_dir       = OUT_DIR_Q

    os.makedirs(out_dir, exist_ok=True)

    try:
        # Load observations
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

        print(f"Observations loaded: {len(obs_series)} points.")

        # Process both forcings
        print("\nProcessing RAW GFS (Results)...")
        errors_org_gfs, errors_best_gfs, raw_gfs = process_forecast_data(
            base_path_gfs, obs_series, forcing_label="GFS"
        )

        print("\nProcessing Bias-Corrected GFS (Results_BC)...")
        errors_org_bc, errors_best_bc, raw_bc = process_forecast_data(
            base_path_bc, obs_series, forcing_label="BCGFS"
        )

        if DATE_START or DATE_END:
            print(f"\nDate window applied: start={DATE_START or '-inf'}, end={DATE_END or '+inf'}")
        else:
            print("\nDate window: not applied (full range).")

        # Generate all plot sets
        run_all_sets(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                     raw_gfs, raw_bc, ELEMENT, out_dir)

        # Save an Excel summary for reuse (e.g., Taylor script)
        excel_path = os.path.join(out_dir, f"forecast_evaluation_GFS_vs_BCGFS_{ELEMENT}.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            raw_gfs.to_excel(writer, "Raw_Data_GFS", index=False)
            raw_bc.to_excel(writer, "Raw_Data_BCGFS", index=False)
            errors_org_gfs.to_excel(writer, "Errors_GFS_OrgP", index=False)
            errors_best_gfs.to_excel(writer, "Errors_GFS_BestP", index=False)
            errors_org_bc.to_excel(writer, "Errors_BCGFS_OrgP", index=False)
            errors_best_bc.to_excel(writer, "Errors_BCGFS_BestP", index=False)
        print(f"\nExcel saved: {excel_path}")

    except Exception as e:
        print(f"\n[ERROR] {e}")

    print("\nDone.")

#%%

# ============================================================
# Forecast Error & Metrics Plotter — Styled Multi-Set Version
# - Generates forecasting-error plots with Mean, Q1, Q3, IQR bars & annotations
# - Generates metrics plots (2×2: NSE, KGE, RMSE, PBIAS)
# - Safe with missing subsets (skips gracefully)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

# Choose element type: "wl" (water level) or "qr" (discharge)
ELEMENT = "wl"

# ---- Base directory that contains Results and Results_BC ----
BASE_DIR = r"C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/GFS_16_0.25_2022_edit"

# Water level paths (N1)
BASE_PATH_GFS_WL   = os.path.join(BASE_DIR, "Results")
BASE_PATH_BC_WL    = os.path.join(BASE_DIR, "Results_BC")
OBS_FILE_WL        = os.path.join(BASE_DIR, "ObsData/WaterLevel/ObsWL.csv")
OUT_DIR_WL         = os.path.join(BASE_DIR, "error_plots_wl_bc_org/error_loop_new")

# Discharge paths (N1)
BASE_PATH_GFS_Q    = os.path.join(BASE_DIR, "Results")
BASE_PATH_BC_Q     = os.path.join(BASE_DIR, "Results_BC")
OBS_FILE_Q         = os.path.join(BASE_DIR, "ObsData/Discharge/ObsQ.csv")
OUT_DIR_Q          = os.path.join(BASE_DIR, "error_plots_spc_bc/error_loop")

# Maximum forecast lead time in days
MAX_LEAD_DAYS = 10

# --- Optional date window (inclusive). Use 'YYYYMMDDHHMM' or set to None.
#DATE_START = "202207010000"
#DATE_END   = "202209100000"
DATE_START = None
DATE_END   = None


# =========================
# HELPER FUNCTIONS
# =========================

def _in_window(ts_str: str) -> bool:
    """Check if timestamp string is within the inclusive window."""
    if DATE_START and ts_str < DATE_START:
        return False
    if DATE_END and ts_str > DATE_END:
        return False
    return True

def _cycle_overlaps_window(index_as_str) -> bool:
    """True if at least one timestamp in a series index is within the window."""
    if not DATE_START and not DATE_END:
        return True
    for ts in index_as_str:
        if _in_window(ts):
            return True
    return False

def load_value_series(csv_path: str) -> pd.Series:
    """
    Load a CSV with columns [Datetime, <numeric>] and return the first numeric
    column as a Series indexed by Datetime (string).
    """
    try:
        df = pd.read_csv(csv_path, dtype={"Datetime": str}).dropna()
        df = df.set_index("Datetime")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError(f"No numeric columns in {csv_path}")
        ser = df[num_cols[0]].astype(float)
        if not ser.index.is_unique:
            ser = ser.groupby(level=0).mean()
        return ser
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.Series(dtype=float)

def compute_lead_day(init_str: str, ts_str: str):
    """
    Lead day = ceil((timestamp - init)/1 day).
    Returns None if lead <=0 or > MAX_LEAD_DAYS.
    """
    t0 = datetime.strptime(init_str, "%Y%m%d%H%M")
    t1 = datetime.strptime(ts_str, "%Y%m%d%H%M")
    d_days = (t1 - t0).total_seconds() / 86400.0
    if d_days <= 0:
        return None
    ld = int(np.ceil(d_days))
    return ld if 1 <= ld <= MAX_LEAD_DAYS else None

def process_forecast_data(base_path: str, obs_series: pd.Series, forcing_label: str):
    """
    Process one forcing dataset (RAW GFS or BC-GFS).

    Returns:
        errors_org  : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Error]
        errors_best : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Error]
        raw_df      : DataFrame [Forcing, ForecastCycle, Datetime, LeadDay, Obs, OrgP, BestP]
    """
    org_files, best_files = [], []

    for root, _, files in os.walk(base_path):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            full = os.path.join(root, fname)
            if "OrgP" in fname and f"_{ELEMENT.lower()}_" in fname:
                org_files.append(full)
            elif "BestP" in fname and f"_{ELEMENT.lower()}_" in fname:
                best_files.append(full)

    org_dict  = {os.path.basename(f).split("_")[0]: f for f in org_files}
    best_dict = {os.path.basename(f).split("_")[0]: f for f in best_files}

    common_inits = sorted(set(org_dict) & set(best_dict))
    print(f"[{forcing_label}] cycles with OrgP & BestP: {len(common_inits)}")

    errors_org, errors_best, raw_rows = [], [], []

    for init in common_inits:
        try:
            org_ser  = load_value_series(org_dict[init])
            best_ser = load_value_series(best_dict[init])

            if not _cycle_overlaps_window(org_ser.index) and not _cycle_overlaps_window(best_ser.index):
                continue

            common_ts = sorted(set(obs_series.index) & set(org_ser.index) & set(best_ser.index))
            common_ts = [ts for ts in common_ts if _in_window(ts)]

            for ts in common_ts:
                lead = compute_lead_day(init, ts)
                if lead is None:
                    continue
                obs = float(obs_series.loc[ts])
                org = float(org_ser.loc[ts])
                bes = float(best_ser.loc[ts])

                errors_org.append({
                    "Forcing": forcing_label,
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Error": org - obs
                })
                errors_best.append({
                    "Forcing": forcing_label,
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Error": bes - obs
                })
                raw_rows.append({
                    "Forcing": forcing_label,
                    "ForecastCycle": init,
                    "Datetime": ts,
                    "LeadDay": lead,
                    "Obs": obs,
                    "OrgP": org,
                    "BestP": bes
                })
        except Exception as e:
            print(f"[{forcing_label}] skip {init}: {e}")

    return pd.DataFrame(errors_org), pd.DataFrame(errors_best), pd.DataFrame(raw_rows)


# =========================
# METRICS
# =========================

def compute_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NSE, KGE, RMSE, PBIAS for each lead day per forcing (Org vs Best)."""
    def nse(o, s):
        o = np.asarray(o); s = np.asarray(s)
        if o.size == 0 or np.isclose(np.var(o), 0):
            return None
        return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

    def kge(o, s):
        o = np.asarray(o); s = np.asarray(s)
        if o.size < 2:
            return None
        r = np.corrcoef(o, s)[0, 1] if np.std(o) > 0 and np.std(s) > 0 else 0.0
        alpha = np.std(s) / np.std(o) if np.std(o) != 0 else np.nan
        beta  = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
        if np.isnan(alpha) or np.isnan(beta):
            return None
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    def rmse(o, s):
        o = np.asarray(o); s = np.asarray(s)
        return float(np.sqrt(np.mean((s - o) ** 2))) if o.size > 0 else None

    def pbias(o, s):
        o = np.asarray(o); s = np.asarray(s)
        return float(100.0 * np.sum(s - o) / np.sum(o)) if o.size > 0 and not np.isclose(np.sum(o), 0) else None

    rec = []
    for forcing, sub in raw_df.groupby("Forcing"):
        for lead in range(1, MAX_LEAD_DAYS + 1):
            d = sub[sub["LeadDay"] == lead]
            o = d["Obs"].to_numpy()
            op = d["OrgP"].to_numpy()
            bp = d["BestP"].to_numpy()
            rec.append({
                "Forcing": forcing, "Lead": lead,
                "NSE_Org": nse(o, op), "NSE_Best": nse(o, bp),
                "KGE_Org": kge(o, op), "KGE_Best": kge(o, bp),
                "RMSE_Org": rmse(o, op), "RMSE_Best": rmse(o, bp),
                "PBIAS_Org": pbias(o, op), "PBIAS_Best": pbias(o, bp),
            })
    return pd.DataFrame(rec)


# =========================
# PLOTTING — FORECAST ERROR WITH MEAN, Q1, Q3
# =========================

def _agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean, Q1, Q3 per lead day; returns empty with right columns if df empty.
    """
    if df.empty:
        return pd.DataFrame(columns=["LeadDay", "mean", "q1", "q3"])
    stats = df.groupby("LeadDay", dropna=True)["Error"].agg(
        mean="mean",
        q1=lambda x: np.quantile(x, 0.25),
        q3=lambda x: np.quantile(x, 0.75),
    ).reset_index()
    return stats

def _draw_vertical_day_lines(ax, x_min=1, x_max=10, alpha=0.85):
    """Vertical dashed lines at each day across slightly expanded y-range."""
    ymin, ymax = ax.get_ylim()
    margin = 0.12 * (ymax - ymin) if ymax > ymin else 1.0
    y0 = ymin - margin
    y1 = ymax + margin
    for x in range(x_min, x_max + 1):
        ax.vlines(x, y0, y1, color="black", linestyle="--", linewidth=1.0, alpha=alpha)
    ax.set_ylim(y0, y1)

def _day_labels():
    labs = []
    for i in range(1, MAX_LEAD_DAYS + 1):
        if i == 1: labs.append("1st Day")
        elif i == 2: labs.append("2nd Day")
        elif i == 3: labs.append("3rd Day")
        else: labs.append(f"{i}th Day")
    return labs

def _annotate_stat(ax, df: pd.DataFrame, column: str, color: str, offset):
    """Add numeric labels at each point for given column (mean, q1, q3)."""
    if df.empty:
        return
    for _, row in df.iterrows():
        value = row[column]
        ax.annotate(
            f"{value:.2f}",
            (row["LeadDay"], value),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            color=color,
            fontsize=11,
        )

def create_forecast_plot(eog, ebg, eob, ebb, include, title, save_path):
    """
    Styled forecast error plot with Mean, Q1, Q3 (like OrgP script).
    'include' is a list among:
    ["GFS", "GFS(PF-DA)", "BC-GFS", "BC-GFS(PF-DA)"]
    """

    # aggregate stats
    gfs_org  = _agg_stats(eog)
    gfs_best = _agg_stats(ebg)
    bc_org   = _agg_stats(eob)
    bc_best  = _agg_stats(ebb)

    # if truly everything empty, skip
    if all(d.empty for d in [gfs_org, gfs_best, bc_org, bc_best]):
        print(f"[WARN] No data for {os.path.basename(save_path)} — skipped.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- GFS OrgP ----
    if "GFS" in include and not gfs_org.empty:
        # Mean
        ax.plot(
            gfs_org["LeadDay"], gfs_org["mean"],
            "o-", color="red", lw=1.5, ms=4,
            label="Mean (GFS)"
        )
        # Q1
        ax.plot(
            gfs_org["LeadDay"], gfs_org["q1"],
            "_-", color="red", lw=0.8, ms=4, alpha=0.5,
            label="Q1 (GFS)"
        )
        # Q3
        ax.plot(
            gfs_org["LeadDay"], gfs_org["q3"],
            "^-", color="red", lw=0.8, ms=4, alpha=0.5,
            label="Q3 (GFS)"
        )
        # IQR bars
        ax.vlines(
            gfs_org["LeadDay"], gfs_org["q1"], gfs_org["q3"],
            color="red", lw=1.0, alpha=0.7
        )
        # Annotations
        _annotate_stat(ax, gfs_org, "mean", "red", (-18, 10))
      

    # ---- GFS (PF-DA) BestP ----
    if "GFS(PF-DA)" in include and not gfs_best.empty:
        ax.plot(
            gfs_best["LeadDay"], gfs_best["mean"],
            "o--", color="red", lw=1.5, ms=4,
            label="Mean (GFS-PF-DA)"
        )
        ax.plot(
            gfs_best["LeadDay"], gfs_best["q1"],
            "_--", color="red", lw=0.8, ms=4, alpha=0.5,
            label="Q1 (GFS-PF-DA)"
        )
        ax.plot(
            gfs_best["LeadDay"], gfs_best["q3"],
            "^--", color="red", lw=0.8, ms=4, alpha=0.5,
            label="Q3 (GFS-PF-DA)"
        )
        ax.vlines(
            gfs_best["LeadDay"], gfs_best["q1"], gfs_best["q3"],
            color="darkred", lw=1.0, alpha=0.7
        )
        _annotate_stat(ax, gfs_best, "mean", "darkred", (-18, 14))
        

    # ---- BC-GFS OrgP ----
    if "BC-GFS" in include and not bc_org.empty:
        ax.plot(
            bc_org["LeadDay"], bc_org["mean"],
            "o-", color="blue", lw=1.5, ms=4,
            label="Mean (BC-GFS)"
        )
        ax.plot(
            bc_org["LeadDay"], bc_org["q1"],
            "_-", color="blue", lw=0.8, ms=4, alpha=0.5,
            label="Q1 (BC-GFS)"
        )
        ax.plot(
            bc_org["LeadDay"], bc_org["q3"],
            "^-", color="blue", lw=0.8, ms=4, alpha=0.5,
            label="Q3 (BC-GFS)"
        )
        ax.vlines(
            bc_org["LeadDay"], bc_org["q1"], bc_org["q3"],
            color="blue", lw=1.0, alpha=0.7
        )
        _annotate_stat(ax, bc_org, "mean", "blue", (-18, 10))
        
    # ---- BC-GFS (PF-DA) BestP ----
    if "BC-GFS(PF-DA)" in include and not bc_best.empty:
        ax.plot(
            bc_best["LeadDay"], bc_best["mean"],
            "o--", color="blue", lw=1.5, ms=4,
            label="Mean (BC-GFS-PF-DA)"
        )
        ax.plot(
            bc_best["LeadDay"], bc_best["q1"],
            "_--", color="blue", lw=0.8, ms=4, alpha=0.5,
            label="Q1 (BC-GFS-PF-DA)"
        )
        ax.plot(
            bc_best["LeadDay"], bc_best["q3"],
            "^--", color="blue", lw=0.8, ms=4, alpha=0.5,
            label="Q3 (BC-GFS-PF-DA)"
        )
        ax.vlines(
            bc_best["LeadDay"], bc_best["q1"], bc_best["q3"],
            color="navy", lw=1.0, alpha=0.7
        )
        _annotate_stat(ax, bc_best, "mean", "navy", (-18, 14))
        

    # axes, grid, verticals
    ax.set_title(title, fontsize=14, weight="bold", pad=20)
    ax.set_ylabel(
        "Forecast Error of Water Level [m]" if ELEMENT.lower() == "wl"
        else "Forecast Error of Discharge [m³/s]",
        fontsize=14
    )
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
    ax.set_xticklabels(_day_labels(), fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="both", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.tick_params(axis="both", which="both", direction="in", length=6, width=1)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    # vertical day lines (like your reference)
    _draw_vertical_day_lines(ax, 1, MAX_LEAD_DAYS, alpha=0.9)

    ax.legend(loc="upper left", fontsize=9, frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[PLOT] {os.path.basename(save_path)} saved.")


# =========================
# PLOTTING — METRICS 2×2
# =========================

def create_metrics_plot(metrics_df: pd.DataFrame, include, title, save_path):
    """
    2×2 panel plot: NSE, KGE, RMSE, PBIAS.
    'include' controls which series appear (same strings as forecast plot).
    """
    if metrics_df.empty:
        print(f"[WARN] Empty metrics dataset for {os.path.basename(save_path)}")
        return

    gfs   = metrics_df[metrics_df["Forcing"] == "GFS"]
    bcgfs = metrics_df[metrics_df["Forcing"] == "BCGFS"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    def draw_one(ax, met, title_txt):
        # GFS
        if "GFS" in include and not gfs.empty and f"{met}_Org" in gfs.columns:
            ax.plot(gfs["Lead"], gfs[f"{met}_Org"], "-o", color="red", lw=1.2, ms=3, label="GFS")
        if "GFS(PF-DA)" in include and not gfs.empty and f"{met}_Best" in gfs.columns:
            ax.plot(gfs["Lead"], gfs[f"{met}_Best"], "--s", color="red", lw=1.2, ms=3, label="GFS(PF-DA)")
        # BC-GFS
        if "BC-GFS" in include and not bcgfs.empty and f"{met}_Org" in bcgfs.columns:
            ax.plot(bcgfs["Lead"], bcgfs[f"{met}_Org"], "-o", color="blue", lw=1.2, ms=3, label="BC-GFS")
        if "BC-GFS(PF-DA)" in include and not bcgfs.empty and f"{met}_Best" in bcgfs.columns:
            ax.plot(bcgfs["Lead"], bcgfs[f"{met}_Best"], "--s", color="blue", lw=1.2, ms=3, label="BC-GFS(PF-DA)")

        ax.set_title(title_txt, fontsize=12, weight="bold")
        ax.set_xlabel("Lead Day", fontsize=12)
        ax.set_ylabel(met, fontsize=12)
        ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=9)

    draw_one(axs[0, 0], "NSE",   "Nash-Sutcliffe Efficiency vs Lead Day")
    draw_one(axs[0, 1], "KGE",   "Kling-Gupta Efficiency vs Lead Day")
    draw_one(axs[1, 0], "RMSE",  "Root Mean Square Error vs Lead Day")
    draw_one(axs[1, 1], "PBIAS", "Percent Bias vs Lead Day")

    plt.suptitle(title, fontsize=14, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=720, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[METRIC] {os.path.basename(save_path)} saved.")


# =========================
# MULTI-SET DRIVER
# =========================

def run_all_sets(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                 raw_gfs, raw_bc, element, out_dir):

    # --- Base titles for consistency ---
    base_error   = f"Forecast Error – {element.upper()}(N1, 2022)"
    base_metrics = f"Forecast Metrics – {element.upper()}(N1, 2022)"

    # --- Define all plot sets ---
    sets = [
        # tag, include list, error title, metrics title
        ("GFS_only",
         ["GFS"],
         f"{base_error}: GFS",
         f"{base_metrics}: GFS"),

        ("GFS_vs_BCGFS",
         ["GFS", "BC-GFS"],
         f"{base_error}: GFS vs BC-GFS",
         f"{base_metrics}: GFS vs BC-GFS"),

        ("GFS_BCGFS_GFSPFDA",
         ["GFS", "BC-GFS", "GFS(PF-DA)"],
         f"{base_error}: GFS, BC-GFS, and GFS(PF-DA)",
         f"{base_metrics}: GFS, BC-GFS, and GFS(PF-DA)"),

        ("ALL",
         ["GFS", "BC-GFS", "GFS(PF-DA)", "BC-GFS(PF-DA)"],
         f"{base_error}: Before and After PF-DA",
         f"{base_metrics}: Before and After PF-DA"),
    ]

    for tag, include, err_title, met_title in sets:
        print(f"\n--- Generating {tag} ---")

        # Forecast error plot
        save_err = os.path.join(out_dir, f"forecast_error_{tag}_{element}.png")
        create_forecast_plot(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                             include, err_title, save_err)

        # Metrics: build raw subset needed
        raw = pd.DataFrame()
        if any(x in include for x in ["GFS", "GFS(PF-DA)"]):
            raw = pd.concat([raw, raw_gfs], ignore_index=True)
        if any(x in include for x in ["BC-GFS", "BC-GFS(PF-DA)"]):
            raw = pd.concat([raw, raw_bc], ignore_index=True)
        if not raw.empty:
            metrics = compute_metrics(raw)
            save_met = os.path.join(out_dir, f"forecast_metrics_{tag}_{element}.png")
            create_metrics_plot(metrics, include, met_title, save_met)
        else:
            print(f"[WARN] No raw data for metrics in {tag}.")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    # Resolve paths by element
    if ELEMENT.lower() == "wl":
        base_path_gfs = BASE_PATH_GFS_WL
        base_path_bc  = BASE_PATH_BC_WL
        obs_file      = OBS_FILE_WL
        out_dir       = OUT_DIR_WL
    else:
        base_path_gfs = BASE_PATH_GFS_Q
        base_path_bc  = BASE_PATH_BC_Q
        obs_file      = OBS_FILE_Q
        out_dir       = OUT_DIR_Q

    os.makedirs(out_dir, exist_ok=True)

    try:
        # Load observations
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

        print(f"Observations loaded: {len(obs_series)} points.")

        # Process both forcings
        print("\nProcessing RAW GFS (Results)...")
        errors_org_gfs, errors_best_gfs, raw_gfs = process_forecast_data(
            base_path_gfs, obs_series, forcing_label="GFS"
        )

        print("\nProcessing Bias-Corrected GFS (Results_BC)...")
        errors_org_bc, errors_best_bc, raw_bc = process_forecast_data(
            base_path_bc, obs_series, forcing_label="BCGFS"
        )

        if DATE_START or DATE_END:
            print(f"\nDate window applied: start={DATE_START or '-inf'}, end={DATE_END or '+inf'}")
        else:
            print("\nDate window: not applied (full range).")

        # Generate all plot sets
        run_all_sets(errors_org_gfs, errors_best_gfs, errors_org_bc, errors_best_bc,
                     raw_gfs, raw_bc, ELEMENT, out_dir)

        # Save an Excel summary for reuse (e.g., Taylor script)
        excel_path = os.path.join(out_dir, f"forecast_evaluation_GFS_vs_BCGFS_{ELEMENT}.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            raw_gfs.to_excel(writer, "Raw_Data_GFS", index=False)
            raw_bc.to_excel(writer, "Raw_Data_BCGFS", index=False)
            errors_org_gfs.to_excel(writer, "Errors_GFS_OrgP", index=False)
            errors_best_gfs.to_excel(writer, "Errors_GFS_BestP", index=False)
            errors_org_bc.to_excel(writer, "Errors_BCGFS_OrgP", index=False)
            errors_best_bc.to_excel(writer, "Errors_BCGFS_BestP", index=False)
        print(f"\nExcel saved: {excel_path}")

    except Exception as e:
        print(f"\n[ERROR] {e}")

    print("\nDone.")


#%%

# ============================================================
# Taylor Diagram (Reference-Style) – Multi-Scenario (KGE-based)
# Generates FOUR plots:
#   1) GFS
#   2) GFS + BC-GFS
#   3) GFS + BC-GFS + GFS(PF-DA)
#   4) ALL (GFS, BC-GFS, GFS(PF-DA), BC-GFS(PF-DA))
# NOTE: The angular axis now uses KGE instead of correlation.
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

BASE_DIR = r"C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/GFS_16_0.25_2024_edit"
OUT_DIR  = os.path.join(BASE_DIR, "error_plots_wl_bc_org") if ELEMENT.lower() == "wl" \
           else os.path.join(BASE_DIR, "error_plots_spc_bc")
EXCEL_PATH = os.path.join(OUT_DIR, f"forecast_evaluation_GFS_vs_BCGFS_{ELEMENT}.xlsx")
PLOT_DIR = os.path.join(OUT_DIR, "Taylor_plot_new")
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------- HELPERS ------------------------------

def _parse_dt(s):
    return pd.to_datetime(s, format="%Y%m%d%H%M") if s else None

def load_raw(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    xls = pd.ExcelFile(path)

    # Two sheets: raw GFS and raw BC-GFS data
    gfs = pd.read_excel(xls, "Raw_Data_GFS")
    gfs["Forcing"] = "GFS"

    bc  = pd.read_excel(xls, "Raw_Data_BCGFS")
    bc["Forcing"]  = "BCGFS"

    # Combine them in one DataFrame
    df = pd.concat([gfs, bc], ignore_index=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Optional date filtering
    s, e = _parse_dt(DATE_START), _parse_dt(DATE_END)
    if s is not None:
        df = df[df["Datetime"] >= s]
    if e is not None:
        df = df[df["Datetime"] <= e]

    # Keep only needed columns (whatever exists)
    cols = ["Forcing","ForecastCycle","Datetime","LeadDay","Obs","OrgP","BestP"]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["Obs"])
    return df

def _valid(obs, sim):
    """Return finite pairs only."""
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]

def taylor_metrics(obs, sim):
    """
    Compute:
      - std of obs
      - std of sim
      - KGE (instead of correlation)
      - centered RMSD
      - bias
    """
    y, yhat = _valid(obs, sim)
    if y.size < 2:
        return [np.nan]*5

    mean_o = float(np.mean(y))
    mean_s = float(np.mean(yhat))
    syo = float(np.std(y, ddof=1))
    sys = float(np.std(yhat, ddof=1))

    # Pearson correlation (needed inside KGE)
    if syo > 0 and sys > 0:
        r = float(np.corrcoef(y, yhat)[0, 1])
    else:
        r = np.nan

    # Ratio of standard deviations
    alpha = sys / syo if syo > 0 else np.nan

    # Ratio of means (avoid division by zero)
    if mean_o != 0:
        beta = mean_s / mean_o
    else:
        beta = np.nan

    # KGE formula
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    else:
        kge = np.nan

    # Centered RMSD
    crmsd = float(np.sqrt(np.mean(((y - mean_o) - (yhat - mean_s))**2)))

    # Bias (mean error)
    bias  = float((yhat - y).mean())

    return syo, sys, kge, crmsd, bias

def collect_stats(df: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    recs = []

    # (Forcing type, column to use, label for scenario)
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
            if s.empty:
                continue
            std_obs, std_sim, kge, crmsd, bias = taylor_metrics(s["Obs"], s[col])
            recs.append(dict(
                scenario=label,
                forcing=forcing,
                kind=col,
                lead=ld,
                std_obs=std_obs,
                std_sim=std_sim,
                kge=kge,
                crmsd=crmsd,
                bias=bias
            ))

    return pd.DataFrame.from_records(recs)

# --------------------- BACKGROUND ----------------------------

def draw_background(ax, r_max=2.0):
    """
    Draw the Taylor-like diagram background:
    - CRMSD contours
    - KGE rays (using same math as correlation, but labeling as KGE)
    - Std-dev circles
    """
    ref = 1.0
    ax.set_xticklabels([])

    # CRMSD contours (green)
    rs = np.linspace(0, r_max, 260)
    ts = np.linspace(0, np.pi/2, 260)
    RS, TS = np.meshgrid(rs, ts)
    RMS = np.sqrt(ref**2 + RS**2 - 2*ref*RS*np.cos(TS))
    cs = ax.contour(TS, RS, RMS,
                    levels=[0.3, 0.5, 0.7, 1.0, 1.2, 1.5],
                    colors="green", linewidths=0.5)
    plt.clabel(cs, fmt="%.1f", fontsize=12, inline=True, colors=["green"])

    # KGE rays (same construction as correlation rays)
    for c in [0.99,0.95,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10]:
        th = np.arccos(np.clip(c, -1, 1))
        ax.plot([th, th], [0, r_max],
                linestyle="dashed", color="black", linewidth=0.8, alpha=0.6)

    # Std-dev rings (normalized)
    for v in [0.5, 1.0, 1.5]:
        ax.plot(np.linspace(0, np.pi/2, 300),
                np.full(300, v),
                color="darkgreen", linestyle="dotted", linewidth=1.0)

    # Unit circle (std of obs = 1)
    ax.plot(np.linspace(0, np.pi/2, 300),
            np.full(300, 1.0),
            color="black", linewidth=1.5)

    # Reference observation point
    ax.plot(0.02, 1.0, marker="*", color="black", markersize=20)

    # Polar frame settings
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.set_rticks([0.5, 1.0, 1.5])
    ax.set_rlabel_position(92)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(False)

    # KGE tick labels (angle → KGE)
    tick_vals = [round(v, 2) for v in np.arange(0.10, 0.91, 0.10)] + [0.95, 0.99]
    for c in tick_vals:
        th = np.arccos(np.clip(c, -1, 1))
        ax.plot([th, th], [r_max-0.035, r_max],
                color="black", linewidth=1.0)
        ax.text(th, r_max+0.06, f"{c:.2f}",
                ha="center", va="bottom", fontsize=16, color="black")

    # Axis label for KGE
    ax.text(np.deg2rad(40), r_max + 0.18, "KGE",
            ha="center", va="bottom", fontsize=16, color="black")

# --------------------- PLOTTING -----------------------------

COLOR_FOR = {
    "GFS": "red",
    "GFS(PF-DA)": "green",
    "BC-GFS": "blue",
    "BC-GFS(PF-DA)": "orange"
}

LEAD_MARK = {
    1:"o",  2:"s",  3:"^", 4:"D", 5:"v",
    6:"P",  7:"X",  8:"<", 9:">", 10:"*"
}

def plot_taylor_subset(df_stats, element, out_dir,
                       scenarios_to_include, title_text, filename_suffix):
    # Filter only selected scenarios
    df = df_stats[df_stats["scenario"].isin(scenarios_to_include)].copy()
    if df.empty:
        print(f"[WARN] No data for {scenarios_to_include}")
        return

    # Normalized std dev (radial axis)
    df["r"] = df["std_sim"] / df["std_obs"]

    # Angular axis now uses KGE instead of correlation
    df["theta"] = np.arccos(np.clip(df["kge"], -1, 1))

    # Keep only finite values
    df = df[np.isfinite(df["r"]) & np.isfinite(df["theta"])]
    if df.empty:
        print(f"[WARN] No finite r/theta for {scenarios_to_include}")
        return

    # Set radial max based on data
    r_max = max(1.5, min(2.0, float(np.nanmax(df["r"]) * 1.15)))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(12, 8), dpi=160)
    draw_background(ax, r_max=r_max)

    # Scatter points: each scenario x lead day
    for _, row in df.iterrows():
        ax.plot(row["theta"], row["r"],
                linestyle="None",
                marker=LEAD_MARK.get(int(row["lead"]), "o"),
                markersize=12,
                markeredgewidth=0.8,
                markeredgecolor="white",
                color=COLOR_FOR.get(row["scenario"], "black"))

    # Reference obs handle for legend
    obs_handle = Line2D([0],[0],
                        marker="*", linestyle="None",
                        markersize=12,
                        markerfacecolor="black",
                        markeredgecolor="black",
                        label="Observation")

    # Scenario legend (colors)
    scen_handles = []
    for scen in ["GFS", "GFS(PF-DA)", "BC-GFS", "BC-GFS(PF-DA)"]:
        if scen in df["scenario"].unique():
            scen_handles.append(
                Line2D([0],[0],
                       marker="o", linestyle="None",
                       markersize=9,
                       markerfacecolor=COLOR_FOR[scen],
                       markeredgecolor=COLOR_FOR[scen],
                       label=scen)
            )
    scen_handles.append(obs_handle)

    # Lead legend (marker shapes)
    lead_handles = []
    for ld in sorted(df["lead"].unique()):
        lead_handles.append(
            Line2D([0],[0],
                   marker=LEAD_MARK.get(int(ld), "o"),
                   linestyle="None",
                   markersize=9,
                   markerfacecolor="none",
                   markeredgecolor="black",
                   label=f"Day {int(ld)}")
        )

    # Place legends
    leg1 = ax.legend(handles=scen_handles, title="Scenarios",
                     loc="center left", bbox_to_anchor=(0.92, 0.95),
                     frameon=False, fontsize=12, title_fontsize=14)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=lead_handles, title="Lead Days",
                     loc="center left", bbox_to_anchor=(1.0, 0.60),
                     frameon=False, fontsize=12, title_fontsize=14)

    # Titles and axis labels
    ax.set_title(title_text, fontsize=16, pad=50)
    fig.text(0.46, -0.03,
             "Standard deviation (Normalized)",
             ha="center", va="bottom", fontsize=16)
    fig.text(0.13, 0.48,
             "Standard deviation (Normalized)",
             ha="center", va="center",
             rotation=90, fontsize=16)

    # Save
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

    # Compute statistics (std, KGE, etc.) for each scenario & lead day
    stats = collect_stats(raw, MAX_LEAD_DAYS)

    # Save raw stats for debugging / inspection
    stats.to_csv(
        os.path.join(PLOT_DIR, f"taylor_ALL_leads_stats_{ELEMENT}.csv"),
        index=False
    )

    base = "Taylor Diagram – WL(N1, 2024)"

    # 1️⃣ GFS only
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS"],
        f"{base}: GFS",
        "1.GFS_only"
    )

    # 2️⃣ GFS vs BC-GFS
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS","BC-GFS"],
        f"{base}: GFS vs BC-GFS",
        "2.GFS_vs_BCGFS"
    )

    # 3️⃣ GFS, BC-GFS, GFS(PF-DA)
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS","BC-GFS","GFS(PF-DA)"],
        f"{base}: GFS, BC-GFS, and GFS(PF-DA)",
        "3.GFS_vs_BC_GFS_vs_GFS-PFDA"
    )

    # 4️⃣ All four: before & after PF-DA (RAW & BC)
    plot_taylor_subset(stats, ELEMENT, PLOT_DIR,
        ["GFS","GFS(PF-DA)","BC-GFS","BC-GFS(PF-DA)"],
        f"{base}: Before and After PF-DA",
        "4.ALL"
    )

if __name__ == "__main__":
    main()



#%%

from PIL import Image, ImageDraw, ImageFont

# ---- Input images (same size) ----
imgA = Image.open("C:/Users/bikra/Desktop/Progress/1. Final Document/Obj3/2022/6.ALL_taylor.png")
imgB = Image.open("C:/Users/bikra/Desktop/Progress/1. Final Document/Obj3/2024/6.ALL_taylor.png")

# ---- Create canvas (side-by-side + space for labels) ----
w, h = imgA.size
canvas = Image.new("RGB", (w*2, h + 110), "white")

# ---- Paste images ----
canvas.paste(imgA, (0, 0))
canvas.paste(imgB, (w, 0))

# ---- Add labels (a) and (b) ----
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 80)
except:
    font = ImageFont.load_default()

# Center labels under each image
draw.text((w//2 - 20, h + 10), "(a)", font=font, fill="black")
draw.text((w + w//2 - 20, h + 10), "(b)", font=font, fill="black")
# ---- Save output ----
canvas.save("C:/Users/bikra/Desktop/Progress/1. Final Document/Obj3/taylor_diagrams_side_by_side.png")

print("Saved as taylor_diagrams_side_by_side.png")

