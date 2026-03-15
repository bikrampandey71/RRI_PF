# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 15:14:39 2025

@author: bikra
"""

#%%

"""
orgp

# PURPOSE: Read RAW from Sheet 2 (LeadDay, Obs, OrgP), compute metrics/errors,
#          write to Sheet 1 & 3, and plot (metrics 2×2, error: boxplot+IQR+means).

"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG --------
EXCEL_PATH = r"C:/Users/bikra/Desktop/DA_PF_RRI/with_DA/Daily/Metrics/2024/N1_wl_2024/Results/error_plots_wl_edit/forecast_evaluation_wl_edit1.xlsx"
MAX_LEAD_DAYS, ELEMENT, DPI = 10, "wl", 720
OUT_DIR = os.path.dirname(EXCEL_PATH)
SHEET1_NAME, SHEET3_NAME = "OrgP_Metrics", "OrgP_Errors"

def _ylabel(elem): return "Error of Water Level [m]" if elem.lower()=="wl" else "Error of Discharge [m³/s]"

# -------- METRICS / ERRORS --------
def compute_orgp_metrics(raw_df):
    def nse(o,s):
        o,s = np.asarray(o), np.asarray(s)
        if o.size==0: return None
        if np.all(o==o[0]): return 1.0 if np.allclose(s,o) else float("-inf")
        return 1 - np.sum((s-o)**2)/np.sum((o-o.mean())**2)

    def kge(o,s):
        o,s = np.asarray(o), np.asarray(s)
        if o.size<2: return None
        mu_o, mu_s = o.mean(), s.mean()
        std_o, std_s = o.std(ddof=0), s.std(ddof=0)
        r = (1.0 if np.allclose(s,o) else 0.0) if (std_o==0 or std_s==0) else np.corrcoef(o,s)[0,1]
        alpha = std_s/std_o if std_o!=0 else np.nan
        beta  = mu_s/mu_o  if mu_o!=0 else np.nan
        if np.isnan([r,alpha,beta]).any(): return None
        return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

    def rmse(o,s):
        o,s=np.asarray(o),np.asarray(s)
        return None if o.size==0 else float(np.sqrt(np.mean((s-o)**2)))

    def pbias(o,s):
        o,s = np.asarray(o, dtype=float), np.asarray(s, dtype=float)
        if o.size==0 or np.isclose(np.sum(o), 0): return None
        # PBIAS = 100 * sum(s - o) / sum(o); + = overestimation, - = underestimation
        return float(100.0 * (np.sum(s - o) / np.sum(o)))

    rows=[]
    for lead in range(1, MAX_LEAD_DAYS+1):
        d = raw_df[raw_df["LeadDay"]==lead]
        o, s = d["Obs"].to_numpy(), d["OrgP"].to_numpy()
        rows.append({
            "Lead":lead,
            "NSE_Org":nse(o,s),
            "KGE_Org":kge(o,s),
            "RMSE_Org":rmse(o,s),
            "PBIAS_Org":pbias(o,s)
        })
    return pd.DataFrame(rows)

def compute_orgp_errors(raw_df):
    if not {"LeadDay","Obs","OrgP"}.issubset(raw_df.columns):
        raise ValueError("RAW sheet must contain: LeadDay, Obs, OrgP")
    df = raw_df.copy(); df["Error"] = df["OrgP"] - df["Obs"]
    return df[["LeadDay","Error"]]

# -------- PLOTS --------
def plot_metrics_2x2(metrics_df, element, save_path):
    mp = metrics_df.replace({None:np.nan, float("-inf"):np.nan}).sort_values("Lead")
    fig, axs = plt.subplots(2,2, figsize=(12,8))

    axs[0,0].plot(mp["Lead"], mp["NSE_Org"], "o-", color="red", lw=1); axs[0,0].set_title("Nash-Sutcliffe Efficiency vs Lead Day")
    axs[0,1].plot(mp["Lead"], mp["KGE_Org"], "o-", color="red", lw=1); axs[0,1].set_title("Kling-Gupta Efficiency vs Lead Day")
    axs[1,0].plot(mp["Lead"], mp["RMSE_Org"],"o-", color="red", lw=1); axs[1,0].set_title("Root Mean Square Error vs Lead Day")
    axs[1,1].plot(mp["Lead"], mp["PBIAS_Org"],"o-", color="red", lw=1); axs[1,1].set_title("Percent Bias vs Lead Day")

    for ax,y in zip(axs.ravel(),["NSE","KGE","RMSE","PBIAS [%]"]):
        ax.set_xlabel("Lead Day", fontsize=12); ax.set_ylabel(y, fontsize=12)
        ax.grid(True, alpha=.4); ax.set_xticks(range(1, MAX_LEAD_DAYS+1))

    plt.suptitle(f"Forecast Performance Metrics (GFS vs RID), {element.upper()}, N1, 2024",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout(); plt.savefig(save_path, dpi=DPI, bbox_inches="tight"); plt.show()

def plot_error_from_errors_sheet(errors_df, element, save_path):
    d = errors_df.copy(); d["LeadDay"]=d["LeadDay"].astype(int); d=d[d["LeadDay"]>=1]
    if d.empty: print("No stats to plot for OrgP_Errors."); return

    df_long = d.rename(columns={"LeadDay":"Forecast Day"}); df_long["Forecast Day"]=df_long["Forecast Day"].astype(str)
    days = sorted(d["LeadDay"].unique())
    wide = d.pivot_table(index=d.groupby("LeadDay").cumcount(), columns="LeadDay", values="Error")[days]
    q1,q3 = wide.quantile(0.25), wide.quantile(0.75); iqr = q3-q1  # quartiles from original data

    plt.figure(figsize=(12,7))
    cat_order = [str(dy) for dy in range(1, MAX_LEAD_DAYS+1)]
    ax = sns.boxplot(
        data=df_long, x="Forecast Day", y="Error",
        order=cat_order, color="#87CEEB", linewidth=1.2, width=0.6,
        showcaps=False, showfliers=False
    )
    ax.set_xticklabels(_day_labels(MAX_LEAD_DAYS), fontsize = 12)

    mean_vals, x_pos = [], []
    for i, day in enumerate(days):
        y = wide[day].dropna()
        q1v, q3v, iqr_v = float(q1[day]), float(q3[day]), float(iqr[day])
        lo, hi = q1v - 1.5*iqr_v, q3v + 1.5*iqr_v
        y_in = y[(y>=lo)&(y<=hi)]
        y_in = y if y_in.empty else y_in   # fall back to all values if no inliers
        m = float(y.mean())
        ymin, ymax = float(y_in.min()), float(y_in.max()); spread = abs(ymax-ymin)+1e-9
        mean_vals.append(m); x_pos.append(i)

        plt.vlines(i, ymin, ymax, colors="gray", linewidth=1.0, zorder=3)
        plt.hlines([ymin, ymax], i-0.15, i+0.15, colors="gray", linewidth=1.0, zorder=3)
        plt.scatter(i, m, marker="o", color="red", s=50, zorder=5)

        plt.text(i, m-0.07*spread,   f"μ={m:.2f}",    ha="right", fontsize=12, color="red")
        plt.text(i, q3v+0.02*spread, f"Q3={q3v:.2f}", ha="right", fontsize=9, color="blue")
        plt.text(i, q1v-0.03*spread, f"Q1={q1v:.2f}", ha="right", fontsize=9, color="blue")
        plt.text(i, ymin-0.04*spread,f"Min={ymin:.2f}",ha="center", fontsize=9, color="white",
                 bbox=dict(facecolor="black", boxstyle="round,pad=0.3"))
        plt.text(i, ymax+0.03*spread,f"Max={ymax:.2f}",ha="center", fontsize=9, color="black",
                 bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

    plt.plot(x_pos, mean_vals, linestyle="--", color="red", linewidth=0.8, zorder=4)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.title(f"Forecasting Error Plot (GFS vs RID), {element.upper()}, N1, 2024", fontsize=14, weight="bold")
    plt.ylabel(_ylabel(element), fontsize=14); plt.xlabel("Forecast Lead Time", fontsize=14)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tick_params(axis="both", which="both", direction="in", length=6, width=1, bottom=True, left=True, labelsize = 12)
    plt.tight_layout(); plt.savefig(save_path, dpi=DPI, bbox_inches="tight"); plt.show()

# -------- MAIN --------
if __name__ == "__main__":
    xls = pd.ExcelFile(EXCEL_PATH)
    raw_sheet = xls.sheet_names[1] if len(xls.sheet_names)>1 else xls.sheet_names[0]
    raw_df = pd.read_excel(xls, raw_sheet)
    need = {"LeadDay","Obs","OrgP"}; miss = need - set(raw_df.columns)
    if miss: raise SystemExit(f"RAW sheet '{raw_sheet}' missing: {miss}")

    metrics_df = compute_orgp_metrics(raw_df)
    errors_df  = compute_orgp_errors(raw_df)

    OUT_XLSX = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(EXCEL_PATH))[0] + "_UPDATED.xlsx")
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
        metrics_df.to_excel(w, sheet_name=SHEET1_NAME, index=False)
        raw_df.to_excel(w,     sheet_name=raw_sheet,   index=False)
        errors_df.to_excel(w,  sheet_name=SHEET3_NAME, index=False)

    met_png = os.path.join(OUT_DIR, f"forecast_metrics_orgp_{ELEMENT}_upd.png")
    err_png = os.path.join(OUT_DIR, f"forecast_error_orgp_{ELEMENT}_upd1.png")
    plot_metrics_2x2(metrics_df, ELEMENT, met_png)
    plot_error_from_errors_sheet(errors_df, ELEMENT, err_png)

    print("Saved:", OUT_XLSX)
    print("Saved:", met_png)
    print("Saved:", err_png)


#%%

# LANGUAGE: Python 3
# PURPOSE: Read RAW from Sheet 2 (LeadDay, Obs, OrgP, BestP), compute metrics/errors,
#          write metrics to Sheet 1 and errors (long-format) to Sheet 3, and plot.
#          Metrics: NSE, KGE, RMSE, PBIAS.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
EXCEL_PATH = r"C:/Users/bikra/Desktop/DA_PF_RRI/with_DA/Daily/Metrics/2022/N1_wl_2022/Results/error_plots_noDA/forecast_evaluation_wl_edit.xlsx"   # <- set your file
MAX_LEAD_DAYS = 10
ELEMENT = "wl"   # "wl" or "qr"
DPI = 720
OUT_DIR = os.path.dirname(EXCEL_PATH)

SHEET1_NAME = "Metrics_Summary"
SHEET3_NAME = "Errors_Long"   # LeadDay, Product, Error (Product ∈ {OrgP, BestP})

# =========================
# HELPERS
# =========================
def _day_labels(n):
    lab = []
    for i in range(1, n+1):
        if i == 1: lab.append("1st Day")
        elif i == 2: lab.append("2nd Day")
        elif i == 3: lab.append("3rd Day")
        else: lab.append(f"{i}th Day")
    return lab

def _ylabel(elem):
    return "Error of Water Level [m]" if elem.lower() == "wl" else "Error of Discharge [m³/s]"

# =========================
# METRICS / ERRORS
# =========================
def compute_metrics_both(raw_df):
    """RAW must have LeadDay, Obs, OrgP, BestP. Returns metrics per lead for both."""

    def nse(o, s):
        o, s = np.array(o), np.array(s)
        if o.size == 0: return None
        if np.all(o == o[0]):
            return 1.0 if np.allclose(s, o) else float("-inf")
        return 1 - np.sum((s - o)**2) / np.sum((o - np.mean(o))**2)

    def kge(o, s):
        o, s = np.array(o), np.array(s)
        if o.size < 2: return None
        mu_o, mu_s = np.mean(o), np.mean(s)
        std_o, std_s = np.std(o, ddof=0), np.std(s, ddof=0)
        r = (1.0 if np.allclose(s, o) else 0.0) if (std_o == 0 or std_s == 0) else np.corrcoef(o, s)[0, 1]
        alpha = std_s / std_o if std_o != 0 else np.nan
        beta  = mu_s / mu_o if mu_o != 0 else np.nan
        if np.isnan(r) or np.isnan(alpha) or np.isnan(beta): return None
        return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

    def rmse(o, s):
        o, s = np.array(o), np.array(s)
        return None if o.size == 0 else float(np.sqrt(np.mean((s - o)**2)))

    def pbias(o, s):
        o, s = np.array(o, dtype=float), np.array(s, dtype=float)
        if o.size == 0 or np.isclose(np.sum(o), 0): return None
        return float(100.0 * (np.sum(s - o) / np.sum(o)))

    recs = []
    for lead in range(1, MAX_LEAD_DAYS + 1):
        d  = raw_df[raw_df["LeadDay"] == lead]
        o  = d["Obs"].to_numpy()
        s1 = d["OrgP"].to_numpy()
        s2 = d["BestP"].to_numpy()
        recs.append({
            "Lead": lead,
            "NSE_Org":   nse(o, s1),
            "NSE_Best":  nse(o, s2),
            "KGE_Org":   kge(o, s1),
            "KGE_Best":  kge(o, s2),
            "RMSE_Org":  rmse(o, s1),
            "RMSE_Best": rmse(o, s2),
            "PBIAS_Org": pbias(o, s1),
            "PBIAS_Best":pbias(o, s2),
        })
    return pd.DataFrame(recs)

def compute_errors_long(raw_df):
    """Return long-format errors with columns LeadDay, Product, Error for OrgP and BestP."""
    need = {"LeadDay","Obs","OrgP","BestP"}
    if not need.issubset(raw_df.columns):
        raise ValueError(f"RAW sheet must contain: {need}")
    df = raw_df.copy()
    org = df.assign(Product="OrgP",  Error=df["OrgP"]  - df["Obs"])[["LeadDay","Product","Error"]]
    bes = df.assign(Product="BestP", Error=df["BestP"] - df["Obs"])[["LeadDay","Product","Error"]]
    return pd.concat([org, bes], ignore_index=True)

# =========================
# PLOTS
# =========================
def plot_metrics_2x2_both(metrics_df, element, save_path):
    mp = metrics_df.replace({None: np.nan, float("-inf"): np.nan}).sort_values("Lead")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0,0].plot(mp["Lead"], mp["NSE_Org"],  "o-", color="red",  lw=1, label="GFS")
    axs[0,0].plot(mp["Lead"], mp["NSE_Best"], "s-", color="blue", lw=1, label="PF-DA")
    axs[0,0].set_title("Nash-Sutcliffe Efficiency vs Lead Day");  
    axs[0,0].set_xlabel("Lead Day");  axs[0,0].set_ylabel("NSE", fontsize=12);  
    axs[0,0].grid(True, alpha=.4); axs[0,0].legend(fontsize=10)

    axs[0,1].plot(mp["Lead"], mp["KGE_Org"],  "o-", color="red",  lw=1, label="GFS")
    axs[0,1].plot(mp["Lead"], mp["KGE_Best"], "s-", color="blue", lw=1, label="PF-DA")
    axs[0,1].set_title("Kling-Gupta Efficiency vs Lead Day");  
    axs[0,1].set_xlabel("Lead Day");  axs[0,1].set_ylabel("KGE", fontsize=12);  
    axs[0,1].grid(True, alpha=.4); axs[0,1].legend(fontsize=10)

    axs[1,0].plot(mp["Lead"], mp["RMSE_Org"],  "o-", color="red",  lw=1, label="GFS")
    axs[1,0].plot(mp["Lead"], mp["RMSE_Best"], "s-", color="blue", lw=1, label="PF-DA")
    axs[1,0].set_title("Root Mean Square Error vs Lead Day"); 
    axs[1,0].set_xlabel("Lead Day"); axs[1,0].set_ylabel("RMSE", fontsize=12); 
    axs[1,0].grid(True, alpha=.4); axs[1,0].legend(fontsize=10)

    axs[1,1].plot(mp["Lead"], mp["PBIAS_Org"],  "o-", color="red",  lw=1, label="GFS")
    axs[1,1].plot(mp["Lead"], mp["PBIAS_Best"], "s-", color="blue", lw=1, label="PF-DA")
    axs[1,1].set_title("Percent Bias vs Lead Day"); 
    axs[1,1].set_xlabel("Lead Day"); axs[1,1].set_ylabel("PBIAS [%]", fontsize=12); 
    axs[1,1].grid(True, alpha=.4); axs[1,1].legend(fontsize=10)

    for ax in axs.ravel():
        ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))

    plt.suptitle(f"Forecast Performance Metrics: Before (GFS) vs After PF-DA, {element.upper()}, N1, 2022", 
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()

def plot_combined_error_long(errors_long_df, element, save_path):
    """Mean/Q1/Q3 + IQR for OrgP (red) and BestP (blue) from long-format errors."""
    if errors_long_df.empty:
        print("No errors to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    day_ticks = range(1, MAX_LEAD_DAYS + 1)

    for product, color, marker, ls in [("OrgP","red","o","-"), ("BestP","blue","s","-")]:
        sub = errors_long_df[errors_long_df["Product"] == product]
        sub = sub[sub["LeadDay"].astype(int) >= 1]
        if sub.empty: 
            continue

        st = (
            sub.groupby("LeadDay")["Error"]
               .agg(mean="mean", q1=lambda x: np.quantile(x,0.25), q3=lambda x: np.quantile(x,0.75))
               .reset_index().sort_values("LeadDay")
        )

        ax.plot(st["LeadDay"], st["mean"], marker=marker, ms=6, ls=ls, color=color, lw=1.5,
        label=f"{'GFS' if product=='OrgP' else 'PF-DA'} Mean Error")
        ax.plot(st["LeadDay"], st["q1"], marker="v", ms=5, ls="--", color=color, lw=1.0, alpha=0.5,
        label=f"{'GFS' if product=='OrgP' else 'PF-DA'} Q1 (25%)")
        ax.plot(st["LeadDay"], st["q3"], marker="^", ms=5, ls="--", color=color, lw=1.0, alpha=0.5,
        label=f"{'GFS' if product=='OrgP' else 'PF-DA'} Q3 (75%)")

        ax.vlines(st["LeadDay"], st["q1"], st["q3"], color="black", linestyle="-", linewidth=1.5, zorder=1)

        for _, r in st.iterrows():
            ld = int(r["LeadDay"])
            y = r["mean"]
            dy = 10 if product=="OrgP" else -20
            ax.annotate(f"{y:.2f}", (ld, y), textcoords="offset points", xytext=(-15,dy),
                        ha='center', color=color, fontsize=10)

    ax.set_title(f"Forecasting Error Plot: Before (GFS) vs After PF-DA, {element.upper()}, N1, 2022", fontsize=14, weight='bold', pad=20)
    ax.set_ylabel(_ylabel(element), fontsize=14)
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    ax.set_xticks(day_ticks); ax.set_xticklabels(_day_labels(MAX_LEAD_DAYS), fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='both', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=DPI, bbox_inches='tight'); plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    xls = pd.ExcelFile(EXCEL_PATH)
    raw_sheet = xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]
    raw_df = pd.read_excel(xls, raw_sheet)

    need = {"LeadDay","Obs","OrgP","BestP"}
    miss = need - set(raw_df.columns)
    if miss: raise SystemExit(f"RAW sheet '{raw_sheet}' missing: {miss}")

    # Compute
    metrics_df = compute_metrics_both(raw_df)
    errors_long = compute_errors_long(raw_df)

    # Write new workbook
    OUT_XLSX = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(EXCEL_PATH))[0] + "_UPDATED.xlsx")
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
        metrics_df.to_excel(w, sheet_name=SHEET1_NAME, index=False)
        raw_df.to_excel(w,     sheet_name=raw_sheet,   index=False)
        errors_long.to_excel(w, sheet_name=SHEET3_NAME, index=False)

    # Plots
    met_png = os.path.join(OUT_DIR, f"forecast_metrics_comparison_{ELEMENT}_DA1.png")
    err_png = os.path.join(OUT_DIR, f"forecast_error_combined_{ELEMENT}_DA1.png")
    plot_metrics_2x2_both(metrics_df, ELEMENT, met_png)
    plot_combined_error_long(errors_long, ELEMENT, err_png)

    print("Saved:", OUT_XLSX)
    print("Saved:", met_png)
    print("Saved:", err_png)

#%%
# LANGUAGE: Python 3
# PURPOSE: Compare 2022 vs 2024 WITHOUT DA (GFS/OrgP only) for 10-day lead.
#          - Read RAW from each Excel (sheet with LeadDay, Obs, OrgP)
#          - Compute metrics (NSE, KGE, RMSE, PBIAS) and long-format errors
#          - Save updated per-year Excel files
#          - Plot side-by-side style comparisons (2022 vs 2024) for metrics and errors
#          - ERROR plot shows Mean + Q1/Q3 with IQR sticks; Q1 has triangle-down markers, Q3 triangle-up

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
# Set your two input files (RAW must have LeadDay, Obs, OrgP)
EXCEL_PATH_2022 = r"C:/Users/bikra/Desktop/DA_PF_RRI/with_DA/Daily/Metrics/2022/N1_wl_2022/Results/error_plots_edit/forecast_evaluation_wl_edit5.xlsx"
EXCEL_PATH_2024 = r"C:/Users/bikra/Desktop/DA_PF_RRI/with_DA/Daily/Metrics/2024/N1_wl_2024/Results/error_plots_wl_edit/forecast_evaluation_wl_edit1.xlsx"

ELEMENT = "wl"     # "wl" or "qr" (for y-axis label units only)
MAX_LEAD_DAYS = 10
DPI = 720

SHEET_METRICS = "Metrics_Summary_OrgP"
SHEET_ERRORS  = "Errors_OrgP_Long"

# Output directory (default: folder of the 2024 file)
OUT_DIR = os.path.join(os.path.dirname(EXCEL_PATH_2024), "compare_2022_2024_orgp")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def _ylabel(elem):
    return "Error of Water Level [m]" if elem.lower() == "wl" else "Error of Discharge [m³/s]"

def _day_labels(n):
    lab = []
    for i in range(1, n+1):
        if i == 1: lab.append("1st Day")
        elif i == 2: lab.append("2nd Day")
        elif i == 3: lab.append("3rd Day")
        else: lab.append(f"{i}th Day")
    return lab

def _load_raw_from_excel(xlsx_path):
    """Load the RAW sheet (2nd if exists, otherwise 1st). Returns DataFrame and sheet name."""
    xls = pd.ExcelFile(xlsx_path)
    raw_sheet = xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]
    df = pd.read_excel(xls, raw_sheet)
    return df, raw_sheet

# =========================
# METRICS / ERRORS (OrgP only)
# =========================
def compute_orgp_metrics(raw_df):
    """Return metrics per lead for OrgP only: NSE, KGE, RMSE, PBIAS."""

    def nse(o, s):
        o, s = np.asarray(o), np.asarray(s)
        if o.size == 0: return None
        if np.all(o == o[0]):  # constant obs
            return 1.0 if np.allclose(s, o) else float("-inf")
        return 1 - np.sum((s - o)**2) / np.sum((o - o.mean())**2)

    def kge(o, s):
        o, s = np.asarray(o), np.asarray(s)
        if o.size < 2: return None
        mu_o, mu_s = o.mean(), s.mean()
        std_o, std_s = o.std(ddof=0), s.std(ddof=0)
        r = (1.0 if np.allclose(s, o) else 0.0) if (std_o==0 or std_s==0) else np.corrcoef(o, s)[0, 1]
        alpha = std_s / std_o if std_o != 0 else np.nan
        beta  = mu_s / mu_o if mu_o != 0 else np.nan
        if np.isnan(r) or np.isnan(alpha) or np.isnan(beta): return None
        return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

    def rmse(o, s):
        o, s = np.asarray(o), np.asarray(s)
        return None if o.size == 0 else float(np.sqrt(np.mean((s - o)**2)))

    def pbias(o, s):
        o, s = np.asarray(o, dtype=float), np.asarray(s, dtype=float)
        denom = np.sum(o)
        if o.size == 0 or np.isclose(denom, 0): return None
        return float(100.0 * (np.sum(s - o) / denom))

    rows = []
    for lead in range(1, MAX_LEAD_DAYS + 1):
        d = raw_df[raw_df["LeadDay"] == lead]
        o = d["Obs"].to_numpy()
        s = d["OrgP"].to_numpy()
        rows.append({
            "Lead": lead,
            "NSE_Org":   nse(o, s),
            "KGE_Org":   kge(o, s),
            "RMSE_Org":  rmse(o, s),
            "PBIAS_Org": pbias(o, s),
        })
    return pd.DataFrame(rows)

def compute_orgp_errors_long(raw_df):
    """Return long-format errors for OrgP: LeadDay, Error (OrgP - Obs)."""
    need = {"LeadDay", "Obs", "OrgP"}
    if not need.issubset(raw_df.columns):
        raise ValueError(f"RAW sheet must contain: {need}")
    df = raw_df.copy()
    df["Error"] = df["OrgP"] - df["Obs"]
    return df[["LeadDay", "Error"]]

# =========================
# PLOTS (comparison 2022 vs 2024)
# =========================
def plot_metrics_compare_2022_2024(m2022, m2024, element, save_path):
    """2×2 metrics panels; each panel shows 2022 vs 2024 for OrgP."""
    m2022 = m2022.replace({None: np.nan, float("-inf"): np.nan}).sort_values("Lead")
    m2024 = m2024.replace({None: np.nan, float("-inf"): np.nan}).sort_values("Lead")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # NSE
    axs[0,0].plot(m2022["Lead"], m2022["NSE_Org"],  "o-", color="red",  lw=1, label="2022 (GFS)")
    axs[0,0].plot(m2024["Lead"], m2024["NSE_Org"],  "s-", color="green", lw=1, label="2024 (GFS)")
    axs[0,0].set_title("Nash-Sutcliffe Efficiency vs Lead Day")
    axs[0,0].set_xlabel("Lead Day"); axs[0,0].set_ylabel("NSE")
    axs[0,0].grid(True, alpha=.4); axs[0,0].legend(fontsize=10)

    # KGE
    axs[0,1].plot(m2022["Lead"], m2022["KGE_Org"],  "o-", color="red",  lw=1, label="2022 (GFS)")
    axs[0,1].plot(m2024["Lead"], m2024["KGE_Org"],  "s-", color="green", lw=1, label="2024 (GFS)")
    axs[0,1].set_title("Kling-Gupta Efficiency vs Lead Day")
    axs[0,1].set_xlabel("Lead Day"); axs[0,1].set_ylabel("KGE")
    axs[0,1].grid(True, alpha=.4); axs[0,1].legend(fontsize=10)

    # RMSE
    axs[1,0].plot(m2022["Lead"], m2022["RMSE_Org"], "o-", color="red",  lw=1, label="2022 (GFS)")
    axs[1,0].plot(m2024["Lead"], m2024["RMSE_Org"], "s-", color="green", lw=1, label="2024 (GFS)")
    axs[1,0].set_title("Root Mean Square Error vs Lead Day")
    axs[1,0].set_xlabel("Lead Day"); axs[1,0].set_ylabel("RMSE")
    axs[1,0].grid(True, alpha=.4); axs[1,0].legend(fontsize=10)

    # PBIAS
    axs[1,1].plot(m2022["Lead"], m2022["PBIAS_Org"], "o-", color="red",  lw=1, label="2022 (GFS)")
    axs[1,1].plot(m2024["Lead"], m2024["PBIAS_Org"], "s-", color="green", lw=1, label="2024 (GFS)")
    axs[1,1].set_title("Percent Bias vs Lead Day")
    axs[1,1].set_xlabel("Lead Day"); axs[1,1].set_ylabel("PBIAS [%]")
    axs[1,1].grid(True, alpha=.4); axs[1,1].legend(fontsize=10)

    for ax in axs.ravel():
        ax.set_xticks(range(1, MAX_LEAD_DAYS + 1))

    plt.suptitle(f"Forecast Performance Metrics (GFS vs RID): 2022 vs 2024, {element.upper()}, N1", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()

def plot_errors_compare_2022_2024(e2022, e2024, element, save_path):
    """Mean/Q1/Q3 + IQR for 2022 (red) and 2024 (green), OrgP only, with markers for Q1 & Q3."""
    if e2022.empty and e2024.empty:
        print("No errors to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    day_ticks = range(1, MAX_LEAD_DAYS + 1)

    for label, df, color, mean_marker, ls in [
        ("2022", e2022, "red",  "o", "-"),
        ("2024", e2024, "green", "s", "-"),
    ]:
        sub = df[df["LeadDay"].astype(int) >= 1]
        if sub.empty:
            continue

        st = (
            sub.groupby("LeadDay")["Error"]
               .agg(mean="mean",
                    q1=lambda x: np.quantile(x, 0.25),
                    q3=lambda x: np.quantile(x, 0.75))
               .reset_index().sort_values("LeadDay")
        )

        # Mean line
        ax.plot(st["LeadDay"], st["mean"], marker=mean_marker, linestyle=ls, color=color,
                linewidth=2.0, label=f"{label} Mean Error")

        # Q1 line with triangle-down markers
        ax.plot(st["LeadDay"], st["q1"], marker="v", markersize = 5, linestyle="--", color=color,
                linewidth=1.0, alpha=0.5, label=f"{label} Q1 (25th %)")

        # Q3 line with triangle-up markers
        ax.plot(st["LeadDay"], st["q3"], marker="^", markersize = 5, linestyle="--", color=color,
                linewidth=1.0, alpha=0.5, label=f"{label} Q3 (75th %)")

        # IQR vertical lines
        ax.vlines(st["LeadDay"], st["q1"], st["q3"], color="black", linestyle="-", linewidth=1.5, zorder=1)

        # Annotate means (offset differently for the two years)
        for _, r in st.iterrows():
            ld = int(r["LeadDay"])
            y  = r["mean"]
            dy = 10 if "2022" in label else -20
            ax.annotate(f"{y:.2f}", (ld, y), textcoords="offset points", xytext=(-15, dy),
                        ha='center', color=color, fontsize=12)

    ax.set_title(f"Forecasting Error Plot (GFS vs RID): 2022 vs 2024, {element.upper()}, N1", fontsize=14, weight='bold', pad=20)
    ax.set_ylabel(_ylabel(element), fontsize=14)
    ax.set_xlabel("Forecast Lead Time", fontsize=14)
    ax.set_xticks(day_ticks); ax.set_xticklabels(_day_labels(MAX_LEAD_DAYS), fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='both', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    ax.legend(loc='best', fontsize=10, ncol=2)
    plt.tight_layout(); plt.savefig(save_path, dpi=DPI, bbox_inches='tight'); plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # --- Load 2022 ---
    raw2022, raw_sheet_2022 = _load_raw_from_excel(EXCEL_PATH_2022)
    need2022 = {"LeadDay","Obs","OrgP"}
    miss2022 = need2022 - set(raw2022.columns)
    if miss2022:
        raise SystemExit(f"2022 RAW sheet '{raw_sheet_2022}' missing: {miss2022}")

    # --- Load 2024 ---
    raw2024, raw_sheet_2024 = _load_raw_from_excel(EXCEL_PATH_2024)
    need2024 = {"LeadDay","Obs","OrgP"}  # BestP may exist; ignored here
    miss2024 = need2024 - set(raw2024.columns)
    if miss2024:
        raise SystemExit(f"2024 RAW sheet '{raw_sheet_2024}' missing: {miss2024}")

    # Ensure LeadDay ints
    raw2022["LeadDay"] = raw2022["LeadDay"].astype(int)
    raw2024["LeadDay"] = raw2024["LeadDay"].astype(int)

    # --- Compute metrics & errors for each year ---
    metrics_2022 = compute_orgp_metrics(raw2022)
    metrics_2024 = compute_orgp_metrics(raw2024)
    errors_2022  = compute_orgp_errors_long(raw2022)
    errors_2024  = compute_orgp_errors_long(raw2024)

    # --- Save UPDATED per-year Excel files (with metrics + errors) ---
    out_2022 = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(EXCEL_PATH_2022))[0] + "_UPDATED.xlsx")
    with pd.ExcelWriter(out_2022, engine="xlsxwriter") as w:
        metrics_2022.to_excel(w, sheet_name=SHEET_METRICS, index=False)
        raw2022.to_excel(w,     sheet_name=raw_sheet_2022, index=False)
        errors_2022.to_excel(w, sheet_name=SHEET_ERRORS,   index=False)

    out_2024 = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(EXCEL_PATH_2024))[0] + "_UPDATED.xlsx")
    with pd.ExcelWriter(out_2024, engine="xlsxwriter") as w:
        metrics_2024.to_excel(w, sheet_name=SHEET_METRICS, index=False)
        raw2024.to_excel(w,     sheet_name=raw_sheet_2024, index=False)
        errors_2024.to_excel(w, sheet_name=SHEET_ERRORS,   index=False)

    print("Saved per-year UPDATED workbooks:")
    print("  -", out_2022)
    print("  -", out_2024)

    # --- Plots: side-by-side comparisons (same style) ---
    met_png = os.path.join(OUT_DIR, f"metrics_compare_orgp_{ELEMENT}_2022_vs_2024.png")
    err_png = os.path.join(OUT_DIR, f"errors_compare_orgp_{ELEMENT}_2022_vs_2024.png")

    plot_metrics_compare_2022_2024(metrics_2022, metrics_2024, ELEMENT, met_png)
    plot_errors_compare_2022_2024(errors_2022, errors_2024, ELEMENT, err_png)

    print("Saved plots:")
    print("  -", met_png)
    print("  -", err_png)
