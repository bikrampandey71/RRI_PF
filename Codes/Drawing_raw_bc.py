
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:18:21 2025

@author: bikra
"""
# -*- coding: utf-8 -*-

"""
Batch hydrograph plotting for RRI/PF results.

This version plots:
- Observed water level / discharge
- Offline HII simulation
- Forecasts driven by RAW GFS (directory: Results)
- Forecasts driven by Bias-Corrected GFS (directory: Results_BC)

For each forecast cycle (PresentTimeTxt) it overlays:
- All particles
- Best particle
- Original particle
- Weight-mean particle

RAW and BC series use different colors and legend labels so they can be compared.

Rainfall:
- Past + forecast rainfall (RAW & BC) are plotted as solid step lines
  on the secondary axis:
    - RAW GFS rainfall: red line
    - Bias-corrected GFS rainfall: purple/blue line
- Rainfall series have no legend labels.
"""

import os, sys, glob, copy, re, statistics, datetime, configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# ------------------------- helpers -------------------------

def _safe_first(glob_pattern):
    """Return first match for a glob pattern, or None if nothing matches."""
    hits = glob.glob(glob_pattern)
    return hits[0] if hits else None

def _file_exists(path):
    """Return True if path is a non-empty string and points to an existing file."""
    return (path is not None) and os.path.exists(path)

def _safe_font(path_hint):
    """
    Try to use a user-provided TTF file; otherwise fall back to Times New Roman,
    and finally to matplotlib default if needed.
    """
    if path_hint and os.path.exists(path_hint):
        return font_manager.FontProperties(fname=path_hint)
    try:
        return font_manager.FontProperties(family='Times New Roman')
    except Exception:
        return font_manager.FontProperties()

def _safe_loadtxt(path, **kwargs):
    """Wrapper around np.loadtxt that prints a message and returns None on failure."""
    if not _file_exists(path):
        print(f'[skip] not found: {path}')
        return None
    try:
        return np.loadtxt(path, **kwargs)
    except Exception as e:
        print(f'[skip] failed to read {path}: {e}')
        return None

def _safe_read_csv(path, **kwargs):
    """Wrapper around pd.read_csv that prints a message and returns None on failure."""
    if not _file_exists(path):
        print(f'[skip] not found: {path}')
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f'[skip] failed to read {path}: {e}')
        return None

def daterange_ts(start_dt, end_dt, step_minutes):
    """Yield YYYYMMDDHHMM strings from start to end inclusive, stepping step_minutes."""
    dt = start_dt
    while dt <= end_dt:
        yield dt.strftime('%Y%m%d%H%M')
        dt += datetime.timedelta(minutes=step_minutes)

# ------------------------- config & I/O -------------------------

def ConfigPara(f):
    """Read DrawingConfig.ini and unpack the parameters we need."""
    cp = configparser.ConfigParser()
    cp.read(f, encoding='utf-8')

    RiverName   = str(cp.get('BasicInfo', 'RiverName'))
    StationName = str(cp.get('BasicInfo', 'StationName'))   # PredictionPoint
    ElemQorH    = str(cp.get('BasicInfo', 'ElemQorH'))
    GraLanguage = str(cp.get('BasicInfo', 'Language'))

    BeforeDays  = int(cp.get('FigTime', 'BeforeDays'))
    AfterDays   = int(cp.get('FigTime', 'AfterDays'))

    Min_Y1    = float(cp.get('GraphRange', 'Min_Y1'))
    Max_Y1    = float(cp.get('GraphRange', 'Max_Y1'))
    MaxV_Y1   = float(cp.get('GraphRange', 'MaxV_Y1'))
    Tick_Y1   = float(cp.get('GraphRange', 'Tick_Y1'))
    Min_Rain  = int(cp.get('GraphRange', 'Min_Rain'))
    Max_Rain  = int(cp.get('GraphRange', 'Max_Rain'))
    MaxV_Rain = int(cp.get('GraphRange', 'MaxV_Rain'))
    Tick_Rain = float(cp.get('GraphRange', 'Tick_Rain'))

    ForecastHyd_WtMean_Flg = int(cp.get('PFcondition', 'ForecastHyd_WtMean_Flg'))
    ForecastHyd_OrgP_Flg   = int(cp.get('PFcondition', 'ForecastHyd_OrgP_Flg'))
    ForecastHyd_BestP_Flg  = int(cp.get('PFcondition', 'ForecastHyd_BestP_Flg'))
    AllParticleFlg         = int(cp.get('PFcondition', 'AllParticleFlg'))
    ForecastRainFlg        = int(cp.get('PFcondition', 'ForecastRainFlg'))
    EnsembleRainFlg        = int(cp.get('PFcondition', 'EnsembleRainFlg'))
    BackTimeFlg            = int(cp.get('PFcondition', 'BackTimeFlg'))

    nLevel  = int(cp.get('WarningLevel', 'nLevel'))
    ls_Level_Value, ls_Level_Name = [], []
    for i in range(nLevel):
        ls_Level_Value.append(float(cp.get('WarningLevel', f'Level_{i+1}_Value')))
        ls_Level_Name.append(cp.get('WarningLevel', f'Level_{i+1}_Name'))

    RRIofflineCalc = int(cp.get('OfflineCalc', 'RRIofflineCalc'))
    OfflineFile    = str(cp.get('OfflineCalc', 'OfflineFile'))

    RivSecFlag = int(cp.get('RivSection', 'RivSecFlag'))
    RivSecFile = str(cp.get('RivSection', 'RivSecFile'))

    RainJ_dt_min   = int(cp.get('OthersInfo', 'RainJ_dt_min'))
    LegendPosition = int(cp.get('OthersInfo', 'LegendPosition'))

    return (RiverName, StationName, ElemQorH, GraLanguage, BeforeDays, AfterDays,
            Min_Y1, Max_Y1, MaxV_Y1, Tick_Y1, Min_Rain, Max_Rain, MaxV_Rain, Tick_Rain,
            ForecastHyd_WtMean_Flg, ForecastHyd_OrgP_Flg, ForecastHyd_BestP_Flg, AllParticleFlg,
            ForecastRainFlg, EnsembleRainFlg, BackTimeFlg,
            nLevel, ls_Level_Value, ls_Level_Name,
            RRIofflineCalc, OfflineFile, RivSecFlag, RivSecFile, RainJ_dt_min, LegendPosition)

def Read_location(file):
    """Read location.txt and return a list of station names."""
    if not _file_exists(file):
        print(f"[skip] location file not found: {file}")
        return []
    with open(file, 'r') as f:
        lines = f.readlines()
    names = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        vals = re.split(r'\s+', line)
        if vals:
            names.append(vals[0])
    return names

def tail2list(path):
    """Return the last comma-separated line of a file as a list of floats."""
    if not _file_exists(path):
        print(f'[skip] not found: {path}')
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return []
    try:
        return [float(x.strip()) for x in lines[-1].split(',') if x.strip() != '']
    except Exception as e:
        print(f'[skip] failed to parse tail of {path}: {e}')
        return []

# ------------------------- main drawing -------------------------

def DrawHydro(
    PresentTimeTxt,
    ConvQ2H,
    HomeDir,
    Hydro_f_type,
    Hydro_f,
    Hydro_locNo,
    RsltDir_raw,
    RsltDir_bc,
    ObsData,
    HQData,
    SecData,
    Pn,
    nStates,
    ls_StatesNo,
    BT_dy,
    FT_dy,
    ForecastType,
    OrgPn,
    RRI_dt_min,
    PF_dt_min,
    MsgBox
):
    """
    Draw one hydrograph panel at PresentTimeTxt, overlaying:
      - Observed WL/Q
      - Offline simulation
      - RAW GFS PF forecasts (All / Best / Org / Mean)
      - Bias-corrected GFS PF forecasts (All / Best / Org / Mean)
      - Rainfall (past + forecast, RAW + BC) as solid step lines on secondary axis
    """
    # config
    config_f = os.path.join(HomeDir, 'PythonCode', 'DrawingConfig.ini')
    if not _file_exists(config_f):
        raise FileNotFoundError(f"Config file not found: {config_f}")

    (RiverName, PredictionPoint, ElemQorH, GraLanguage, BeforeDays, AfterDays,
     Min_Y1, Max_Y1, MaxV_Y1, Tick_Y1, Min_Rain, Max_Rain, MaxV_Rain, Tick_Rain,
     ForecastHyd_WtMean_Flg, ForecastHyd_OrgP_Flg, ForecastHyd_BestP_Flg, AllParticleFlg,
     ForecastRainFlg, EnsembleRainFlg, BackTimeFlg,
     nLevel, ls_Level_Value, ls_Level_Name,
     RRIofflineCalc, OfflineFile, RivSecFlag, RivSecFile, RainJ_dt_min, LegendPosition) = ConfigPara(config_f)

    # style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5

    fig = plt.figure(figsize=(12, 7), dpi=300)
    fp = _safe_font(r'C:\Users\bikra\times new roman.ttf')  # change path if needed

    # stationKey usage
    Location_f = os.path.join(HomeDir, 'Particles', 'Particle00001', 'RRI', 'location.txt')
    _ = Read_location(Location_f)
    StationKey = '_'  # kept simple; name already in PredictionPoint

    # filename suffixes
    if ElemQorH == 'H':
        Result_f_mean = '_WtMean_wl'
        Result_f_org  = '_OrgP???_wl'
        Result_f_best = '_BestP???_wl'
        Result_f_all  = '_AllP_wl'
    else:
        Result_f_mean = '_WtMean_qr'
        Result_f_org  = '_OrgP???_qr'
        Result_f_best = '_BestP???_qr'
        Result_f_all  = '_AllP_qr'

    # time window
    PresentTime = datetime.datetime.strptime(PresentTimeTxt, '%Y%m%d%H%M')
    StartTime   = PresentTime - datetime.timedelta(days=BeforeDays)
    EndTime     = PresentTime + datetime.timedelta(days=AfterDays)
    deltaDay    = int((EndTime - StartTime).total_seconds()/(24*3600)) + 2
    YearName    = StartTime.year
    nSkipBT     = BT_dy * (1440 // RRI_dt_min) - 1

    # axis 1 (hydro)
    ax1 = fig.add_subplot(111)
    xNumberSize, xLabelSize = 15, 18
    y1NumberSize, y1LabelSize = 20, 20
    y2NumberSize, y2LabelSize = 18, 20
    X_major_dt_hr = 24; X_minor_dt_hr = 24
    y1_major_dt = 1.0;  y1_minor_dt = 1.0

    ax1.tick_params(axis='x',labelsize=xNumberSize, which='both', direction='inout')
    ax1.tick_params(axis='y',labelsize=y1NumberSize, which='major', direction='out', length=5, width=1, color='black')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=X_major_dt_hr))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=X_minor_dt_hr))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y1_major_dt))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(y1_minor_dt))
    ax1.minorticks_on()
    ax1.grid(which="major", axis="y", color="gray", alpha=1.0, linestyle="dashed", linewidth=1.0)
    ax1.grid(which="major", axis="x", color="gray", alpha=1.0, linestyle="dashed", linewidth=1.0)
    ax1.grid(which="minor", axis="x", color="gray", alpha=1.0, linestyle="dashed", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax1.set_ylim(Min_Y1, Max_Y1)
    ax1.set_xlim(StartTime, EndTime)
    ax1.set_xlabel(YearName, fontsize=xLabelSize+2)
    ax1.xaxis.set_label_coords(0, -0.100)
    ax1.set_ylabel('Water Level [m]' if ElemQorH=='H' else 'Discharge [$m^3$/s]', fontsize=y1LabelSize, fontweight='bold')
    ax1.set_yticks(np.arange(Min_Y1, MaxV_Y1+1, Tick_Y1))

    # axis 2 (hyeto)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='x', labelsize=xNumberSize)
    ax2.tick_params(axis='y', labelsize=y2NumberSize, which='major', direction='out', length=5, width=1, color='black')
    ax2.set_ylim(Max_Rain, Min_Rain)
    ax2.set_xlim(StartTime, EndTime)
    ax2.set_ylabel('Rainfall [mm/hr]', fontsize=y2LabelSize, fontweight='bold')
    ax2.yaxis.set_label_coords(1.06, 0.80)
    ax2.xaxis.set_ticklabels([])
    ax2.set_yticks(np.arange(Min_Rain, MaxV_Rain+1, Tick_Rain))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=X_major_dt_hr))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=X_minor_dt_hr))
    ax2.grid(which="major", axis="y", color="gray", alpha=0.2, linestyle="solid", linewidth=0.1)

    # river section
    if RivSecFlag == 1:
        RivSec_f = os.path.join(HomeDir, RivSecFile)
        SecElev = _safe_loadtxt(RivSec_f, dtype='float', delimiter=",", skiprows=1)
        if SecElev is not None and SecElev.size > 0:
            ZeroElev = SecElev[0, 2]
            SecZeroH = copy.deepcopy(SecElev)
            SecZeroH[:,1] = SecZeroH[:,1] - ZeroElev
            MinLength = SecZeroH[:,0].min()
            MaxLength = SecZeroH[:,0].max()
            StartTime4Fig = StartTime + datetime.timedelta(hours=6)
            EndTime4Fig   = EndTime   - datetime.timedelta(hours=6)
            SecTimeX, SecZeroY = [], []
            for i in range(SecZeroH.shape[0]):
                dt_s = (EndTime4Fig - StartTime4Fig).total_seconds()/(MaxLength-MinLength)*(SecZeroH[i,0]-MinLength)
                SecTimeX.append(StartTime4Fig + datetime.timedelta(seconds = dt_s))
                SecZeroY.append(SecZeroH[i,1])
            ax1.plot(SecTimeX, SecZeroY, lw=3, alpha=1.0, label='River cross section',
                     linestyle='solid', color='dimgray')

    # observed hydro
    f = _safe_loadtxt(ObsData, dtype='float', delimiter=",", skiprows=1)
    ls_ObsDatetime, ls_ObsVal, ls_ObsDatetimeTxt = [], [], []
    if f is not None:
        f_T = f.T
        ls_ObsDateTxt = f_T[0]
        ls_ObsData    = f_T[1]
        for v, val in zip(ls_ObsDateTxt, ls_ObsData):
            s = str(int(v))
            ls_ObsDatetimeTxt.append(s)
            ls_ObsDatetime.append(datetime.datetime.strptime(s, '%Y%m%d%H%M'))
            ls_ObsVal.append(val)
        label_obs = 'RID Observed W.L.' if ElemQorH=='H' else 'RID Observed discharge'
        ax1.plot(ls_ObsDatetime, ls_ObsVal, lw=2, alpha=0.6, label=label_obs,
                 linestyle="solid", color='black', marker="^",
                 markerfacecolor='black', markersize=6, markeredgewidth=0.5, markeredgecolor='white')

    # offline (hindcast)
    if RRIofflineCalc == 1:
        CalcOfflineFile = os.path.join(HomeDir, OfflineFile)
        df_off = _safe_read_csv(CalcOfflineFile, header=0, parse_dates=[0])
        if df_off is not None and not df_off.empty:
            List_Datetime = df_off['datetime'].tolist()
            yvals = df_off.iloc[:,1:].squeeze("columns")
            ax1.plot(List_Datetime, yvals, lw=1.5, alpha=0.6,
                     label=('HII Sim W.L.' if ElemQorH=='H' else 'HII Sim Discharge'),
                     linestyle='solid', color='purple')

    # warning levels
    ls_LevelColor = ['black','black','black','black','red','darkviolet','gray','gray','gray','gray']
    ls_LevelStyle = ["solid","solid","dashed","dashdot","solid","solid","dotted","dotted","dotted","dotted"]
    ls_LevelWidth = [1.5,1.0,1.0,1.0,1.5,1.0,0.5,0.5,0.5,0.5]
    for iLevel in range(nLevel):
        LevelValue = ls_Level_Value[iLevel]
        LevelName  = ls_Level_Name[iLevel]
        if LevelValue > -10:
            y0 = [LevelValue]*deltaDay
            x0 = pd.date_range(str(StartTime), periods=deltaDay, freq='d')
            ax1.plot(x0, y0, linewidth=ls_LevelWidth[iLevel], alpha=0.8,
                     linestyle=ls_LevelStyle[iLevel], color=ls_LevelColor[iLevel])
            ax1.text(0.02, (LevelValue - Min_Y1)/(Max_Y1 - Min_Y1)+0.01,
                     f"{LevelName} ({LevelValue:.2f}m)", size=15,
                     ha='left', va='baseline', clip_on=False, alpha=0.8, transform=plt.gca().transAxes)

    # -------- rainfall reconstruction helpers (for RAW and BC) --------
    def _collect_past_rain_series(root_dir):
        """
        Reconstruct past rainfall (up to PresentTime) from PF rainfall CSVs
        under root_dir (Results or Results_BC).
        Returns (times, rainfall_values).
        """
        DoTime = StartTime
        nStepJ = max(1.0, float(PF_dt_min / max(1, RainJ_dt_min)))
        ls_TimeJ, ls_RainJ = [], []

        while DoTime <= PresentTime:
            DoTimeTxt_loop = DoTime.strftime('%Y%m%d%H%M')
            RsltDir_Do = os.path.join(
                root_dir,
                DoTime.strftime("%Y"),
                DoTime.strftime("%m"),
                DoTime.strftime("%d")
            )
            patt = os.path.join(
                RsltDir_Do,
                f"{DoTimeTxt_loop}{StationKey}{PredictionPoint}_OrgP???_rain_Pn???_BT??d_FT??d.csv"
            )
            Hyeto_RainJ_f = _safe_first(patt)
            if not _file_exists(Hyeto_RainJ_f):
                DoTime += datetime.timedelta(minutes=PF_dt_min)
                continue

            df_RainJ = _safe_read_csv(Hyeto_RainJ_f)
            if df_RainJ is None or df_RainJ.empty:
                DoTime += datetime.timedelta(minutes=PF_dt_min)
                continue

            ls_HyetoDate_int = df_RainJ.iloc[:, 0].to_list()
            ls_HyetoDatetime = [datetime.datetime.strptime(str(i), '%Y%m%d%H%M') for i in ls_HyetoDate_int]
            ls_HyetoRain = df_RainJ.iloc[:, 1].to_list()

            if int(DoTimeTxt_loop) not in ls_HyetoDate_int:
                DoTime += datetime.timedelta(minutes=PF_dt_min)
                continue

            DotimeRain_idx = ls_HyetoDate_int.index(int(DoTimeTxt_loop))
            for iStep in reversed(range(int(nStepJ))):
                idx = DotimeRain_idx-1-iStep
                if 0 <= idx < len(ls_HyetoDatetime)-1:
                    ls_TimeJ.append(ls_HyetoDatetime[idx])
                    ls_RainJ.append(ls_HyetoRain[idx+1])  # align "edge"

            DoTime += datetime.timedelta(minutes=PF_dt_min)

        return ls_TimeJ, ls_RainJ

    def _collect_forecast_rain_series(root_dir):
        """
        Collect forecast rainfall from PresentTime to EndTime from the current
        PF rainfall CSV at PresentTime under root_dir.
        Returns (times, rainfall_values).
        """
        PresentTimeTxt_local = PresentTime.strftime('%Y%m%d%H%M')
        RsltDir_Present = os.path.join(
            root_dir,
            PresentTime.strftime("%Y"),
            PresentTime.strftime("%m"),
            PresentTime.strftime("%d")
        )
        patt = os.path.join(
            RsltDir_Present,
            f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}_OrgP???_rain_Pn???_BT??d_FT??d.csv"
        )
        Hyeto_RainJ_f = _safe_first(patt)
        if not _file_exists(Hyeto_RainJ_f):
            return [], []

        df_RainJ = _safe_read_csv(Hyeto_RainJ_f)
        if df_RainJ is None or df_RainJ.empty:
            return [], []

        ls_HyetoDate_int = df_RainJ.iloc[:, 0].to_list()
        ls_HyetoDatetime = [datetime.datetime.strptime(str(i), '%Y%m%d%H%M') for i in ls_HyetoDate_int]
        ls_HyetoRain = df_RainJ.iloc[:, 1].to_list()

        times_fcst, vals_fcst = [], []
        for t, r in zip(ls_HyetoDatetime, ls_HyetoRain):
            if t >= PresentTime and t <= EndTime:
                times_fcst.append(t)
                vals_fcst.append(r)
        return times_fcst, vals_fcst

    # collect rainfall time series (past + forecast) for RAW and BC GFS
    times_raw_past, vals_raw_past = _collect_past_rain_series(RsltDir_raw)
    times_bc_past,  vals_bc_past  = _collect_past_rain_series(RsltDir_bc)
    times_raw_fcst, vals_raw_fcst = _collect_forecast_rain_series(RsltDir_raw)
    times_bc_fcst,  vals_bc_fcst  = _collect_forecast_rain_series(RsltDir_bc)

    # merge & sort past+forecast for continuous lines
    def _merge_and_sort(times1, vals1, times2, vals2):
        times = times1 + times2
        vals  = vals1  + vals2
        if not times:
            return [], []
        pairs = sorted(zip(times, vals), key=lambda x: x[0])
        tt, vv = zip(*pairs)
        return list(tt), list(vv)

    times_raw_all, vals_raw_all = _merge_and_sort(times_raw_past, vals_raw_past,
                                                  times_raw_fcst, vals_raw_fcst)
    times_bc_all,  vals_bc_all  = _merge_and_sort(times_bc_past, vals_bc_past,
                                                  times_bc_fcst, vals_bc_fcst)

    # plot rainfall as solid step lines (no legend labels)
    if times_raw_all:
        ax2.step(times_raw_all, vals_raw_all,
                 where='post', color='red',
                 linewidth=1.0, alpha=0.8)

    if times_bc_all:
        ax2.step(times_bc_all, vals_bc_all,
                 where='post', color='blue',
                 linewidth=1.0, alpha=0.8)

    # -------- helper to plot a family of hydro forecasts (RAW or BC) --------
    def _plot_family(root_dir, label_prefix, colors,
                     label_all=None, label_best=None, label_org=None, label_mean=None):
        """
        root_dir: base Results or Results_BC directory
        label_prefix: 'GFS' or 'BC-GFS'
        colors: dict with keys 'all', 'best', 'org', 'mean'
        label_*: legend labels for all/mean/best/org (None or '' to hide)
        """
        PresentTimeTxt_local = PresentTime.strftime('%Y%m%d%H%M')
        RsltDir_Present = os.path.join(root_dir,
                                       PresentTime.strftime("%Y"),
                                       PresentTime.strftime("%m"),
                                       PresentTime.strftime("%d"))

        # All particles
        if AllParticleFlg == 1:
            AllP_hydroFile = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_all}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            All_hydro = _safe_loadtxt(
                AllP_hydroFile,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if All_hydro is not None:
                f_T = All_hydro.T
                ls_AllHydroDatetimeTxt = f_T[0]
                ls_AllHydroData = f_T[1:]
                ls_AllHydroDatetime = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                       for x in ls_AllHydroDatetimeTxt]
                for iPn, series in enumerate(ls_AllHydroData):
                    # Only first particle gets the legend label for "all"
                    lbl = label_all if (iPn == 0 and label_all) else ""
                    ax1.plot(ls_AllHydroDatetime, series, lw=0.5, alpha=0.7,
                             linestyle="solid", color=colors['all'],
                             label=lbl)

        # Best particle
        if ForecastHyd_BestP_Flg != 0:
            patt = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_best}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            hydro_Best_f = _safe_first(patt)
            f = _safe_loadtxt(
                hydro_Best_f,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if f is not None:
                f_T = f.T
                ls_DatetimeTxt_HydBestP = f_T[0]
                ls_Data_HydBestP = f_T[1]
                ls_Datetime_HydBestP = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                        for x in ls_DatetimeTxt_HydBestP]
                ls_Hydro4Fig = list(ls_Data_HydBestP)
                if ForecastHyd_BestP_Flg == 2 and ls_ObsDatetimeTxt:
                    ts_key = PresentTime.strftime('%Y%m%d%H%M')
                    if ts_key in ls_ObsDatetimeTxt:
                        NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(ts_key)]
                        NowCalcVal = ls_Hydro4Fig[nSkipBT] if BackTimeFlg==1 else ls_Hydro4Fig[0]
                        ls_Hydro4Fig = (np.array(ls_Hydro4Fig) + (NowObsVal - NowCalcVal)).tolist()
                ax1.plot(ls_Datetime_HydBestP, ls_Hydro4Fig, lw=2.0, alpha=1.0,
                         label=(label_best or ""),
                         linestyle="dotted", color=colors['best'])

        # OrgP (single particle)
        if ForecastHyd_OrgP_Flg != 0:
            patt = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_org}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            hydro_Org_f = _safe_first(patt)
            f = _safe_loadtxt(
                hydro_Org_f,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if f is not None:
                f_T = f.T
                ls_DatetimeTxt_HydOrgP = f_T[0]
                ls_Data_HydOrgP = f_T[1]
                ls_Datetime_HydOrgP = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                       for x in ls_DatetimeTxt_HydOrgP]
                ls_Hydro4Fig = list(ls_Data_HydOrgP)
                if ForecastHyd_OrgP_Flg == 2 and ls_ObsDatetimeTxt:
                    ts_key = PresentTime.strftime('%Y%m%d%H%M')
                    if ts_key in ls_ObsDatetimeTxt:
                        NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(ts_key)]
                        NowCalcVal = ls_Hydro4Fig[nSkipBT] if BackTimeFlg==1 else ls_Hydro4Fig[0]
                        ls_Hydro4Fig = (np.array(ls_Hydro4Fig) + (NowObsVal - NowCalcVal)).tolist()
                ax1.plot(ls_Datetime_HydOrgP, ls_Hydro4Fig, lw=1.5, alpha=1.0,
                         label=(label_org or ""),
                         linestyle="solid", color=colors['org'])

        # Weight-mean particle
        if ForecastHyd_WtMean_Flg != 0:
            hydro_f = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_mean}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            f = _safe_loadtxt(
                hydro_f,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if f is not None:
                f_T = f.T
                ls_DatetimeTxt_HydWtMean = f_T[0]
                ls_Data_HydWtMean = f_T[1]
                ls_Datetime_HydWtMean = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                         for x in ls_DatetimeTxt_HydWtMean]
                ls_Hydro4Fig = list(ls_Data_HydWtMean)
                if ForecastHyd_WtMean_Flg == 2 and ls_ObsDatetimeTxt:
                    ts_key = PresentTime.strftime('%Y%m%d%H%M')
                    if ts_key in ls_ObsDatetimeTxt:
                        NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(ts_key)]
                        NowCalcVal = ls_Hydro4Fig[nSkipBT] if BackTimeFlg==1 else ls_Hydro4Fig[0]
                        ls_Hydro4Fig = (np.array(ls_Hydro4Fig) + (NowObsVal - NowCalcVal)).tolist()
                ax1.plot(ls_Datetime_HydWtMean, ls_Hydro4Fig, lw=1.5, alpha=1.0,
                         label=(label_mean or ""),
                         linestyle="solid", color=colors['mean'])

    # ==== HERE YOU CONTROL WHICH SERIES GET WHICH LABELS ====
    # plot RAW GFS family
    _plot_family(
        root_dir=RsltDir_raw,
        label_prefix='GFS',
        colors={
            'all':  'orange',
            'best': 'red',
            'org':  'red',
            'mean': 'darkred',
        },
        label_org="GFS: Sim W.L",
        label_best="GFS: Best particle",
        label_all="GFS: All particles",
        label_mean="GFS: Mean particle"
    )

    # plot Bias-Corrected GFS family
    _plot_family(
        root_dir=RsltDir_bc,
        label_prefix='BC-GFS',
        colors={
            'all':  'yellow',
            'best': 'blue',
            'org':  'blue',
            'mean': 'magenta',
        },
        label_org="BC-GFS: Sim W.L",
        label_best="BC-GFS: Best particle",
        label_all="BC-GFS: All particles",
        label_mean="BC-GFS: Mean particle"
    )
    # ========================================================

    # present line
    ax1.plot([PresentTime,PresentTime], [Min_Y1, Max_Y1], lw=3.0, alpha=1.0,
             linestyle="solid", color='black')

    # legend box (solid white background) – only hydro series (no rain labels)
    handles1, labels1 = ax1.get_legend_handles_labels()
    leg = ax1.legend(handles1, labels1, prop=fp, loc='upper right', bbox_to_anchor=(1, 0.95))
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    # title (timestamp)
    plt.text(0.5, 1.05, PresentTime, {'color':'red'}, ha='center', va='center',
             fontproperties=fp, clip_on=False, transform=plt.gca().transAxes)

    return plt

# ------------------------- batch runner (fixed start/end) -------------------------

def run_batch_fixed_range(
    HomeDir,
    RsltDir_raw,
    RsltDir_bc,
    ObsData,
    start_ts,
    end_ts,
    pf_dt_min,
    Pn,
    BT_dy,
    FT_dy,
    RRI_dt_min,
    out_dir
):
    """
    Loop from start_ts to end_ts (inclusive) stepping pf_dt_min minutes.
    For each PresentTimeTxt, call DrawHydro and save a PNG.
    Uses both RAW (RsltDir_raw) and Bias-Corrected (RsltDir_bc) results.
    """
    os.makedirs(out_dir, exist_ok=True)

    # read for naming
    cfg = os.path.join(HomeDir, "PythonCode", "DrawingConfig.ini")
    cp = configparser.ConfigParser()
    cp.read(cfg, encoding="utf-8")
    prediction_point = cp.get('BasicInfo', 'StationName')
    elem = cp.get('BasicInfo', 'ElemQorH')

    start_dt = datetime.datetime.strptime(start_ts, '%Y%m%d%H%M')
    end_dt   = datetime.datetime.strptime(end_ts,   '%Y%m%d%H%M')

    idx = 0
    for ts in daterange_ts(start_dt, end_dt, pf_dt_min):
        idx += 1
        try:
            pl = DrawHydro(
                PresentTimeTxt=ts,
                ConvQ2H=None,
                HomeDir=HomeDir,
                Hydro_f_type=None, Hydro_f=None, Hydro_locNo=None,
                RsltDir_raw=RsltDir_raw,
                RsltDir_bc=RsltDir_bc,
                ObsData=ObsData,
                HQData=None, SecData=None,
                Pn=Pn, nStates=0, ls_StatesNo=[],
                BT_dy=BT_dy, FT_dy=FT_dy,
                ForecastType=None, OrgPn=None,
                RRI_dt_min=RRI_dt_min, PF_dt_min=pf_dt_min,
                MsgBox=None
            )
            out_path = os.path.join(out_dir, f"{prediction_point}_{elem}_{ts}.png")
            pl.savefig(out_path, dpi=300, bbox_inches="tight")
            pl.close()
            print(f"[ok] ({idx}) saved {out_path}")
        except Exception as e:
            print(f"[skip] ({idx}) {ts}: {e}")

# ------------------------- run as script -------------------------

if __name__ == "__main__":
    # ==== EDIT THESE PATHS / SETTINGS ====
    HOME_DIR   = r"C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/GFS_16_0.25_2024"  # has PythonCode/DrawingConfig.ini
    RSLT_DIR_RAW  = os.path.join(HOME_DIR, "Results")       # root with YYYY/MM/DD subfolders (RAW GFS)
    RSLT_DIR_BC   = os.path.join(HOME_DIR, "Results_BC")    # root with YYYY/MM/DD subfolders (Bias-Corrected GFS)
    OBS_FILE      = os.path.join(HOME_DIR, "ObsData/WaterLevel/ObsWL.csv")  # obs CSV with two columns: datetime,value

    START_TS   = "202405200000"  # inclusive; format YYYYMMDDHHMM
    END_TS     = "202410100000"  # inclusive
    PF_DT_MIN  = 1440            # step minutes between successive figures

    PN         = 16
    BT_DY      = 1
    FT_DY      = 10
    RRI_DT_MIN = 60

    OUT_DIR    = os.path.join(HOME_DIR, "Drawing_DA")
    # =====================================

    run_batch_fixed_range(
        HomeDir=HOME_DIR,
        RsltDir_raw=RSLT_DIR_RAW,
        RsltDir_bc=RSLT_DIR_BC,
        ObsData=OBS_FILE,
        start_ts=START_TS,
        end_ts=END_TS,
        pf_dt_min=PF_DT_MIN,
        Pn=PN,
        BT_dy=BT_DY,
        FT_dy=FT_DY,
        RRI_dt_min=RRI_DT_MIN,
        out_dir=OUT_DIR
    )

#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:18:21 2025

@author: bikra
"""
# -*- coding: utf-8 -*-

"""
Batch hydrograph plotting for RRI/PF results.

This version plots:
- Observed water level / discharge
- Offline HII simulation
- Forecasts driven by RAW GFS (directory: Results)
- Forecasts driven by Bias-Corrected GFS (directory: Results_BC)

For each forecast cycle (PresentTimeTxt) it overlays:
- All particles
- Best particle
- Original particle
- Weight-mean particle

RAW and BC series use different colors and legend labels so they can be compared.

Rainfall:
- Past + forecast rainfall (RAW & BC) are plotted as solid step lines
  on the secondary axis:
    - RAW GFS rainfall: red line
    - Bias-corrected GFS rainfall: purple/blue line
- Rainfall series have no legend labels.
"""

import os, sys, glob, copy, re, statistics, datetime, configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D  # <<< NEW

# ------------------------- helpers -------------------------

def _safe_first(glob_pattern):
    """Return first match for a glob pattern, or None if nothing matches."""
    hits = glob.glob(glob_pattern)
    return hits[0] if hits else None

def _file_exists(path):
    """Return True if path is a non-empty string and points to an existing file."""
    return (path is not None) and os.path.exists(path)

def _safe_font(path_hint):
    """
    Try to use a user-provided TTF file; otherwise fall back to Times New Roman,
    and finally to matplotlib default if needed.
    """
    if path_hint and os.path.exists(path_hint):
        return font_manager.FontProperties(fname=path_hint)
    try:
        return font_manager.FontProperties(family='Times New Roman')
    except Exception:
        return font_manager.FontProperties()

def _safe_loadtxt(path, **kwargs):
    """Wrapper around np.loadtxt that prints a message and returns None on failure."""
    if not _file_exists(path):
        print(f'[skip] not found: {path}')
        return None
    try:
        return np.loadtxt(path, **kwargs)
    except Exception as e:
        print(f'[skip] failed to read {path}: {e}')
        return None

def _safe_read_csv(path, **kwargs):
    """Wrapper around pd.read_csv that prints a message and returns None on failure."""
    if not _file_exists(path):
        print(f'[skip] not found: {path}')
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f'[skip] failed to read {path}: {e}')
        return None

def daterange_ts(start_dt, end_dt, step_minutes):
    """Yield YYYYMMDDHHMM strings from start to end inclusive, stepping step_minutes."""
    dt = start_dt
    while dt <= end_dt:
        yield dt.strftime('%Y%m%d%H%M')
        dt += datetime.timedelta(minutes=step_minutes)

# ------------------------- config & I/O -------------------------

def ConfigPara(f):
    """Read DrawingConfig.ini and unpack the parameters we need."""
    cp = configparser.ConfigParser()
    cp.read(f, encoding='utf-8')

    RiverName   = str(cp.get('BasicInfo', 'RiverName'))
    StationName = str(cp.get('BasicInfo', 'StationName'))   # PredictionPoint
    ElemQorH    = str(cp.get('BasicInfo', 'ElemQorH'))
    GraLanguage = str(cp.get('BasicInfo', 'Language'))

    BeforeDays  = int(cp.get('FigTime', 'BeforeDays'))
    AfterDays   = int(cp.get('FigTime', 'AfterDays'))

    Min_Y1    = float(cp.get('GraphRange', 'Min_Y1'))
    Max_Y1    = float(cp.get('GraphRange', 'Max_Y1'))
    MaxV_Y1   = float(cp.get('GraphRange', 'MaxV_Y1'))
    Tick_Y1   = float(cp.get('GraphRange', 'Tick_Y1'))
    Min_Rain  = int(cp.get('GraphRange', 'Min_Rain'))
    Max_Rain  = int(cp.get('GraphRange', 'Max_Rain'))
    MaxV_Rain = int(cp.get('GraphRange', 'MaxV_Rain'))
    Tick_Rain = float(cp.get('GraphRange', 'Tick_Rain'))

    ForecastHyd_WtMean_Flg = int(cp.get('PFcondition', 'ForecastHyd_WtMean_Flg'))
    ForecastHyd_OrgP_Flg   = int(cp.get('PFcondition', 'ForecastHyd_OrgP_Flg'))
    ForecastHyd_BestP_Flg  = int(cp.get('PFcondition', 'ForecastHyd_BestP_Flg'))
    AllParticleFlg         = int(cp.get('PFcondition', 'AllParticleFlg'))
    ForecastRainFlg        = int(cp.get('PFcondition', 'ForecastRainFlg'))
    EnsembleRainFlg        = int(cp.get('PFcondition', 'EnsembleRainFlg'))
    BackTimeFlg            = int(cp.get('PFcondition', 'BackTimeFlg'))

    nLevel  = int(cp.get('WarningLevel', 'nLevel'))
    ls_Level_Value, ls_Level_Name = [], []
    for i in range(nLevel):
        ls_Level_Value.append(float(cp.get('WarningLevel', f'Level_{i+1}_Value')))
        ls_Level_Name.append(cp.get('WarningLevel', f'Level_{i+1}_Name'))

    RRIofflineCalc = int(cp.get('OfflineCalc', 'RRIofflineCalc'))
    OfflineFile    = str(cp.get('OfflineCalc', 'OfflineFile'))

    RivSecFlag = int(cp.get('RivSection', 'RivSecFlag'))
    RivSecFile = str(cp.get('RivSection', 'RivSecFile'))

    RainJ_dt_min   = int(cp.get('OthersInfo', 'RainJ_dt_min'))
    LegendPosition = int(cp.get('OthersInfo', 'LegendPosition'))

    return (RiverName, StationName, ElemQorH, GraLanguage, BeforeDays, AfterDays,
            Min_Y1, Max_Y1, MaxV_Y1, Tick_Y1, Min_Rain, Max_Rain, MaxV_Rain, Tick_Rain,
            ForecastHyd_WtMean_Flg, ForecastHyd_OrgP_Flg, ForecastHyd_BestP_Flg, AllParticleFlg,
            ForecastRainFlg, EnsembleRainFlg, BackTimeFlg,
            nLevel, ls_Level_Value, ls_Level_Name,
            RRIofflineCalc, OfflineFile, RivSecFlag, RivSecFile, RainJ_dt_min, LegendPosition)

def Read_location(file):
    """Read location.txt and return a list of station names."""
    if not _file_exists(file):
        print(f"[skip] location file not found: {file}")
        return []
    with open(file, 'r') as f:
        lines = f.readlines()
    names = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        vals = re.split(r'\s+', line)
        if vals:
            names.append(vals[0])
    return names

def tail2list(path):
    """Return the last comma-separated line of a file as a list of floats."""
    if not _file_exists(path):
        print(f'[skip] not found: {path}')
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return []
    try:
        return [float(x.strip()) for x in lines[-1].split(',') if x.strip() != '']
    except Exception as e:
        print(f'[skip] failed to parse tail of {path}: {e}')
        return []

# ------------------------- main drawing -------------------------

def DrawHydro(
    PresentTimeTxt,
    ConvQ2H,
    HomeDir,
    Hydro_f_type,
    Hydro_f,
    Hydro_locNo,
    RsltDir_raw,
    RsltDir_bc,
    ObsData,
    HQData,
    SecData,
    Pn,
    nStates,
    ls_StatesNo,
    BT_dy,
    FT_dy,
    ForecastType,
    OrgPn,
    RRI_dt_min,
    PF_dt_min,
    MsgBox
):
    """
    Draw one hydrograph panel at PresentTimeTxt, overlaying:
      - Observed WL/Q
      - Offline simulation
      - RAW GFS PF forecasts (All / Best / Org / Mean)
      - Bias-corrected GFS PF forecasts (All / Best / Org / Mean)
      - Rainfall (past + forecast, RAW + BC) as solid step lines on secondary axis
    """
    # config
    config_f = os.path.join(HomeDir, 'PythonCode', 'DrawingConfig.ini')
    if not _file_exists(config_f):
        raise FileNotFoundError(f"Config file not found: {config_f}")

    (RiverName, PredictionPoint, ElemQorH, GraLanguage, BeforeDays, AfterDays,
     Min_Y1, Max_Y1, MaxV_Y1, Tick_Y1, Min_Rain, Max_Rain, MaxV_Rain, Tick_Rain,
     ForecastHyd_WtMean_Flg, ForecastHyd_OrgP_Flg, ForecastHyd_BestP_Flg, AllParticleFlg,
     ForecastRainFlg, EnsembleRainFlg, BackTimeFlg,
     nLevel, ls_Level_Value, ls_Level_Name,
     RRIofflineCalc, OfflineFile, RivSecFlag, RivSecFile, RainJ_dt_min, LegendPosition) = ConfigPara(config_f)

    # style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5

    fig = plt.figure(figsize=(12, 7), dpi=300)
    fp = _safe_font(r'C:\Users\bikra\times new roman.ttf')  # change path if needed

    # stationKey usage
    Location_f = os.path.join(HomeDir, 'Particles', 'Particle00001', 'RRI', 'location.txt')
    _ = Read_location(Location_f)
    StationKey = '_'  # kept simple; name already in PredictionPoint

    # filename suffixes
    if ElemQorH == 'H':
        Result_f_mean = '_WtMean_wl'
        Result_f_org  = '_OrgP???_wl'
        Result_f_best = '_BestP???_wl'
        Result_f_all  = '_AllP_wl'
    else:
        Result_f_mean = '_WtMean_qr'
        Result_f_org  = '_OrgP???_qr'
        Result_f_best = '_BestP???_qr'
        Result_f_all  = '_AllP_qr'

    # time window
    PresentTime = datetime.datetime.strptime(PresentTimeTxt, '%Y%m%d%H%M')
    StartTime   = PresentTime - datetime.timedelta(days=BeforeDays)
    EndTime     = PresentTime + datetime.timedelta(days=AfterDays)
    deltaDay    = int((EndTime - StartTime).total_seconds()/(24*3600)) + 2
    YearName    = StartTime.year
    nSkipBT     = BT_dy * (1440 // RRI_dt_min) - 1

    # axis 1 (hydro)
    ax1 = fig.add_subplot(111)
    xNumberSize, xLabelSize = 15, 18
    y1NumberSize, y1LabelSize = 20, 20
    y2NumberSize, y2LabelSize = 18, 20
    X_major_dt_hr = 24; X_minor_dt_hr = 24
    y1_major_dt = 1.0;  y1_minor_dt = 1.0

    ax1.tick_params(axis='x',labelsize=xNumberSize, which='both', direction='inout')
    ax1.tick_params(axis='y',labelsize=y1NumberSize, which='major', direction='out', length=5, width=1, color='black')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=X_major_dt_hr))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=X_minor_dt_hr))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y1_major_dt))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(y1_minor_dt))
    ax1.minorticks_on()
    ax1.grid(which="major", axis="y", color="gray", alpha=1.0, linestyle="dashed", linewidth=1.0)
    ax1.grid(which="major", axis="x", color="gray", alpha=1.0, linestyle="dashed", linewidth=1.0)
    ax1.grid(which="minor", axis="x", color="gray", alpha=1.0, linestyle="dashed", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax1.set_ylim(Min_Y1, Max_Y1)
    ax1.set_xlim(StartTime, EndTime)
    ax1.set_xlabel(YearName, fontsize=xLabelSize+2)
    ax1.xaxis.set_label_coords(0, -0.100)
    ax1.set_ylabel('Water Level [m]' if ElemQorH=='H' else 'Discharge [$m^3$/s]', fontsize=y1LabelSize, fontweight='bold')
    ax1.set_yticks(np.arange(Min_Y1, MaxV_Y1+1, Tick_Y1))

    # axis 2 (hyeto)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='x', labelsize=xNumberSize)
    ax2.tick_params(axis='y', labelsize=y2NumberSize, which='major', direction='out', length=5, width=1, color='black')
    ax2.set_ylim(Max_Rain, Min_Rain)
    ax2.set_xlim(StartTime, EndTime)
    ax2.set_ylabel('Rainfall [mm/hr]', fontsize=y2LabelSize, fontweight='bold')
    ax2.yaxis.set_label_coords(1.06, 0.80)
    ax2.xaxis.set_ticklabels([])
    ax2.set_yticks(np.arange(Min_Rain, MaxV_Rain+1, Tick_Rain))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=X_major_dt_hr))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=X_minor_dt_hr))
    ax2.grid(which="major", axis="y", color="gray", alpha=0.2, linestyle="solid", linewidth=0.1)

    # river section
    if RivSecFlag == 1:
        RivSec_f = os.path.join(HomeDir, RivSecFile)
        SecElev = _safe_loadtxt(RivSec_f, dtype='float', delimiter=",", skiprows=1)
        if SecElev is not None and SecElev.size > 0:
            ZeroElev = SecElev[0, 2]
            SecZeroH = copy.deepcopy(SecElev)
            SecZeroH[:,1] = SecZeroH[:,1] - ZeroElev
            MinLength = SecZeroH[:,0].min()
            MaxLength = SecZeroH[:,0].max()
            StartTime4Fig = StartTime + datetime.timedelta(hours=6)
            EndTime4Fig   = EndTime   - datetime.timedelta(hours=6)
            SecTimeX, SecZeroY = [], []
            for i in range(SecZeroH.shape[0]):
                dt_s = (EndTime4Fig - StartTime4Fig).total_seconds()/(MaxLength-MinLength)*(SecZeroH[i,0]-MinLength)
                SecTimeX.append(StartTime4Fig + datetime.timedelta(seconds = dt_s))
                SecZeroY.append(SecZeroH[i,1])
            ax1.plot(SecTimeX, SecZeroY, lw=3, alpha=1.0, label='River cross section',
                     linestyle='solid', color='dimgray')

    # observed hydro
    f = _safe_loadtxt(ObsData, dtype='float', delimiter=",", skiprows=1)
    ls_ObsDatetime, ls_ObsVal, ls_ObsDatetimeTxt = [], [], []
    if f is not None:
        f_T = f.T
        ls_ObsDateTxt = f_T[0]
        ls_ObsData    = f_T[1]
        for v, val in zip(ls_ObsDateTxt, ls_ObsData):
            s = str(int(v))
            ls_ObsDatetimeTxt.append(s)
            ls_ObsDatetime.append(datetime.datetime.strptime(s, '%Y%m%d%H%M'))
            ls_ObsVal.append(val)
        label_obs = 'RID Observed W.L.' if ElemQorH=='H' else 'RID Observed discharge'
        ax1.plot(ls_ObsDatetime, ls_ObsVal, lw=2, alpha=0.6, label=label_obs,
                 linestyle="solid", color='black', marker="^",
                 markerfacecolor='black', markersize=6, markeredgewidth=0.5, markeredgecolor='white')

    # offline (hindcast)
    if RRIofflineCalc == 1:
        CalcOfflineFile = os.path.join(HomeDir, OfflineFile)
        df_off = _safe_read_csv(CalcOfflineFile, header=0, parse_dates=[0])
        if df_off is not None and not df_off.empty:
            List_Datetime = df_off['datetime'].tolist()
            yvals = df_off.iloc[:,1:].squeeze("columns")
            ax1.plot(List_Datetime, yvals, lw=1.5, alpha=0.6,
                     label=('HII Sim W.L.' if ElemQorH=='H' else 'HII Sim Discharge'),
                     linestyle='solid', color='purple')

    # warning levels
    ls_LevelColor = ['black','black','black','black','red','darkviolet','gray','gray','gray','gray']
    ls_LevelStyle = ["solid","solid","dashed","dashdot","solid","solid","dotted","dotted","dotted","dotted"]
    ls_LevelWidth = [1.5,1.0,1.0,1.0,1.5,1.0,0.5,0.5,0.5,0.5]
    for iLevel in range(nLevel):
        LevelValue = ls_Level_Value[iLevel]
        LevelName  = ls_Level_Name[iLevel]
        if LevelValue > -10:
            y0 = [LevelValue]*deltaDay
            x0 = pd.date_range(str(StartTime), periods=deltaDay, freq='d')
            ax1.plot(x0, y0, linewidth=ls_LevelWidth[iLevel], alpha=0.8,
                     linestyle=ls_LevelStyle[iLevel], color=ls_LevelColor[iLevel])
            ax1.text(0.02, (LevelValue - Min_Y1)/(Max_Y1 - Min_Y1)+0.01,
                     f"{LevelName} ({LevelValue:.2f}m)", size=15,
                     ha='left', va='baseline', clip_on=False, alpha=0.8, transform=plt.gca().transAxes)

    # -------- rainfall reconstruction helpers (for RAW and BC) --------
    def _collect_past_rain_series(root_dir):
        """
        Reconstruct past rainfall (up to PresentTime) from PF rainfall CSVs
        under root_dir (Results or Results_BC).
        Returns (times, rainfall_values).
        """
        DoTime = StartTime
        nStepJ = max(1.0, float(PF_dt_min / max(1, RainJ_dt_min)))
        ls_TimeJ, ls_RainJ = [], []

        while DoTime <= PresentTime:
            DoTimeTxt_loop = DoTime.strftime('%Y%m%d%H%M')
            RsltDir_Do = os.path.join(
                root_dir,
                DoTime.strftime("%Y"),
                DoTime.strftime("%m"),
                DoTime.strftime("%d")
            )
            patt = os.path.join(
                RsltDir_Do,
                f"{DoTimeTxt_loop}{StationKey}{PredictionPoint}_OrgP???_rain_Pn???_BT??d_FT??d.csv"
            )
            Hyeto_RainJ_f = _safe_first(patt)
            if not _file_exists(Hyeto_RainJ_f):
                DoTime += datetime.timedelta(minutes=PF_dt_min)
                continue

            df_RainJ = _safe_read_csv(Hyeto_RainJ_f)
            if df_RainJ is None or df_RainJ.empty:
                DoTime += datetime.timedelta(minutes=PF_dt_min)
                continue

            ls_HyetoDate_int = df_RainJ.iloc[:, 0].to_list()
            ls_HyetoDatetime = [datetime.datetime.strptime(str(i), '%Y%m%d%H%M') for i in ls_HyetoDate_int]
            ls_HyetoRain = df_RainJ.iloc[:, 1].to_list()

            if int(DoTimeTxt_loop) not in ls_HyetoDate_int:
                DoTime += datetime.timedelta(minutes=PF_dt_min)
                continue

            DotimeRain_idx = ls_HyetoDate_int.index(int(DoTimeTxt_loop))
            for iStep in reversed(range(int(nStepJ))):
                idx = DotimeRain_idx-1-iStep
                if 0 <= idx < len(ls_HyetoDatetime)-1:
                    ls_TimeJ.append(ls_HyetoDatetime[idx])
                    ls_RainJ.append(ls_HyetoRain[idx+1])  # align "edge"

            DoTime += datetime.timedelta(minutes=PF_dt_min)

        return ls_TimeJ, ls_RainJ

    def _collect_forecast_rain_series(root_dir):
        """
        Collect forecast rainfall from PresentTime to EndTime from the current
        PF rainfall CSV at PresentTime under root_dir.
        Returns (times, rainfall_values).
        """
        PresentTimeTxt_local = PresentTime.strftime('%Y%m%d%H%M')
        RsltDir_Present = os.path.join(
            root_dir,
            PresentTime.strftime("%Y"),
            PresentTime.strftime("%m"),
            PresentTime.strftime("%d")
        )
        patt = os.path.join(
            RsltDir_Present,
            f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}_OrgP???_rain_Pn???_BT??d_FT??d.csv"
        )
        Hyeto_RainJ_f = _safe_first(patt)
        if not _file_exists(Hyeto_RainJ_f):
            return [], []

        df_RainJ = _safe_read_csv(Hyeto_RainJ_f)
        if df_RainJ is None or df_RainJ.empty:
            return [], []

        ls_HyetoDate_int = df_RainJ.iloc[:, 0].to_list()
        ls_HyetoDatetime = [datetime.datetime.strptime(str(i), '%Y%m%d%H%M') for i in ls_HyetoDate_int]
        ls_HyetoRain = df_RainJ.iloc[:, 1].to_list()

        times_fcst, vals_fcst = [], []
        for t, r in zip(ls_HyetoDatetime, ls_HyetoRain):
            if t >= PresentTime and t <= EndTime:
                times_fcst.append(t)
                vals_fcst.append(r)
        return times_fcst, vals_fcst

    # collect rainfall time series (past + forecast) for RAW and BC GFS
    times_raw_past, vals_raw_past = _collect_past_rain_series(RsltDir_raw)
    times_bc_past,  vals_bc_past  = _collect_past_rain_series(RsltDir_bc)
    times_raw_fcst, vals_raw_fcst = _collect_forecast_rain_series(RsltDir_raw)
    times_bc_fcst,  vals_bc_fcst  = _collect_forecast_rain_series(RsltDir_bc)

    # merge & sort past+forecast for continuous lines
    def _merge_and_sort(times1, vals1, times2, vals2):
        times = times1 + times2
        vals  = vals1  + vals2
        if not times:
            return [], []
        pairs = sorted(zip(times, vals), key=lambda x: x[0])
        tt, vv = zip(*pairs)
        return list(tt), list(vv)

    times_raw_all, vals_raw_all = _merge_and_sort(times_raw_past, vals_raw_past,
                                                  times_raw_fcst, vals_raw_fcst)
    times_bc_all,  vals_bc_all  = _merge_and_sort(times_bc_past, vals_bc_past,
                                                  times_bc_fcst, vals_bc_fcst)

    # plot rainfall as solid step lines (no legend labels)
    # if times_raw_all:
    #     ax2.step(times_raw_all, vals_raw_all,
    #              where='post', color='red',
    #              linewidth=1.0, alpha=0.8)

    # if times_bc_all:
    #     ax2.step(times_bc_all, vals_bc_all,
    #              where='post', color='blue',
    #              linewidth=1.0, alpha=0.8)

    # -------- helper to plot a family of hydro forecasts (RAW or BC) --------
    def _plot_family(root_dir, label_prefix, colors,
                     label_all=None, label_best=None, label_org=None, label_mean=None):
        """
        root_dir: base Results or Results_BC directory
        label_prefix: 'GFS' or 'BC-GFS'
        colors: dict with keys 'all', 'best', 'org', 'mean'
        label_*: legend labels for all/mean/best/org (None or '' to hide)
        """
        PresentTimeTxt_local = PresentTime.strftime('%Y%m%d%H%M')
        RsltDir_Present = os.path.join(root_dir,
                                       PresentTime.strftime("%Y"),
                                       PresentTime.strftime("%m"),
                                       PresentTime.strftime("%d"))

        # All particles
        if AllParticleFlg == 1:
            AllP_hydroFile = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_all}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            All_hydro = _safe_loadtxt(
                AllP_hydroFile,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if All_hydro is not None:
                f_T = All_hydro.T
                ls_AllHydroDatetimeTxt = f_T[0]
                ls_AllHydroData = f_T[1:]
                ls_AllHydroDatetime = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                       for x in ls_AllHydroDatetimeTxt]
                for iPn, series in enumerate(ls_AllHydroData):
                    lbl = label_all if (iPn == 0 and label_all) else ""
                    ax1.plot(ls_AllHydroDatetime, series, lw=0.5, alpha=0.7,
                             linestyle="solid", color=colors['all'],
                             label=lbl)

        # Best particle
        if ForecastHyd_BestP_Flg != 0:
            patt = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_best}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            hydro_Best_f = _safe_first(patt)
            f = _safe_loadtxt(
                hydro_Best_f,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if f is not None:
                f_T = f.T
                ls_DatetimeTxt_HydBestP = f_T[0]
                ls_Data_HydBestP = f_T[1]
                ls_Datetime_HydBestP = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                        for x in ls_DatetimeTxt_HydBestP]
                ls_Hydro4Fig = list(ls_Data_HydBestP)
                if ForecastHyd_BestP_Flg == 2 and ls_ObsDatetimeTxt:
                    ts_key = PresentTime.strftime('%Y%m%d%H%M')
                    if ts_key in ls_ObsDatetimeTxt:
                        NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(ts_key)]
                        NowCalcVal = ls_Hydro4Fig[nSkipBT] if BackTimeFlg==1 else ls_Hydro4Fig[0]
                        ls_Hydro4Fig = (np.array(ls_Hydro4Fig) + (NowObsVal - NowCalcVal)).tolist()
                ax1.plot(ls_Datetime_HydBestP, ls_Hydro4Fig, lw=2.5, alpha=1.0,
                         label=(label_best or ""),
                         linestyle="dotted", color=colors['best'])

        # OrgP (single particle)
        if ForecastHyd_OrgP_Flg != 0:
            patt = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_org}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            hydro_Org_f = _safe_first(patt)
            f = _safe_loadtxt(
                hydro_Org_f,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if f is not None:
                f_T = f.T
                ls_DatetimeTxt_HydOrgP = f_T[0]
                ls_Data_HydOrgP = f_T[1]
                ls_Datetime_HydOrgP = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                       for x in ls_DatetimeTxt_HydOrgP]
                ls_Hydro4Fig = list(ls_Data_HydOrgP)
                if ForecastHyd_OrgP_Flg == 2 and ls_ObsDatetimeTxt:
                    ts_key = PresentTime.strftime('%Y%m%d%H%M')
                    if ts_key in ls_ObsDatetimeTxt:
                        NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(ts_key)]
                        NowCalcVal = ls_Hydro4Fig[nSkipBT] if BackTimeFlg==1 else ls_Hydro4Fig[0]
                        ls_Hydro4Fig = (np.array(ls_Hydro4Fig) + (NowObsVal - NowCalcVal)).tolist()
                ax1.plot(ls_Datetime_HydOrgP, ls_Hydro4Fig, lw=1.5, alpha=1.0,
                         label=(label_org or ""),
                         linestyle="solid", color=colors['org'])

        # Weight-mean particle
        if ForecastHyd_WtMean_Flg != 0:
            hydro_f = os.path.join(
                RsltDir_Present,
                f"{PresentTimeTxt_local}{StationKey}{PredictionPoint}{Result_f_mean}_"
                f"Pn{Pn:03d}_BT{BT_dy:02d}d_FT{FT_dy:02d}d.csv"
            )
            f = _safe_loadtxt(
                hydro_f,
                dtype='float',
                delimiter=",",
                skiprows=1+(nSkipBT if BackTimeFlg==0 else 0)
            )
            if f is not None:
                f_T = f.T
                ls_DatetimeTxt_HydWtMean = f_T[0]
                ls_Data_HydWtMean = f_T[1]
                ls_Datetime_HydWtMean = [datetime.datetime.strptime(str(int(x)),'%Y%m%d%H%M')
                                         for x in ls_DatetimeTxt_HydWtMean]
                ls_Hydro4Fig = list(ls_Data_HydWtMean)
                if ForecastHyd_WtMean_Flg == 2 and ls_ObsDatetimeTxt:
                    ts_key = PresentTime.strftime('%Y%m%d%H%M')
                    if ts_key in ls_ObsDatetimeTxt:
                        NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(ts_key)]
                        NowCalcVal = ls_Hydro4Fig[nSkipBT] if BackTimeFlg==1 else ls_Hydro4Fig[0]
                        ls_Hydro4Fig = (np.array(ls_Hydro4Fig) + (NowObsVal - NowCalcVal)).tolist()
                ax1.plot(ls_Datetime_HydWtMean, ls_Hydro4Fig, lw=1.5, alpha=1.0,
                         label=(label_mean or ""),
                         linestyle="solid", color=colors['mean'])

    # ==== plot RAW & BC families (unchanged) ====
    _plot_family(
        root_dir=RsltDir_raw,
        label_prefix='GFS',
        colors={
            'all':  'orange',
            'best': 'red',
            'org':  'red',
            'mean': 'darkred',
        },
        label_org="GFS: Sim W.L",
        label_best="GFS: Best particle",
        label_all="GFS: All particles",
        label_mean="GFS: Mean particle"
    )

    _plot_family(
        root_dir=RsltDir_bc,
        label_prefix='BC-GFS',
        colors={
            'all':  'yellow',
            'best': 'blue',
            'org':  'blue',
            'mean': 'magenta',
        },
        label_org="BC-GFS: Sim W.L",
        label_best="BC-GFS: Best particle",
        label_all="BC-GFS: All particles",
        label_mean="BC-GFS: Mean particle"
    )

    # present line
    ax1.plot([PresentTime,PresentTime], [Min_Y1, Max_Y1], lw=2.0, alpha=1.0,
             linestyle="solid", color='black')

    # ============ LEGEND: independent linewidth/alpha just for labels ============

    # Control legend appearance here (ONLY legend, not plot)
    LEGEND_STYLES = {   # <<< EDIT HERE
        "RID Observed W.L.":          {"linewidth": 2.0, "alpha": 1.0},
        "RID Observed discharge":     {"linewidth": 2.0, "alpha": 1.0},
        "HII Sim W.L.":               {"linewidth": 2.0, "alpha": 1.0},
        "HII Sim Discharge":          {"linewidth": 2.0, "alpha": 1.0},
        "GFS: Sim W.L":               {"linewidth": 2.0, "alpha": 1.0},
        "GFS: Best particle":         {"linewidth": 2.0, "alpha": 1.0},
        "GFS: All particles":         {"linewidth": 2.0, "alpha": 1.0},
        "GFS: Mean particle":         {"linewidth": 2.0, "alpha": 1.0},
        "BC-GFS: Sim W.L":            {"linewidth": 2.0, "alpha": 1.0},
        "BC-GFS: Best particle":      {"linewidth": 2.0, "alpha": 1.0},
        "BC-GFS: All particles":      {"linewidth": 2.0, "alpha": 1.0},
        "BC-GFS: Mean particle":      {"linewidth": 2.0, "alpha": 1.0},
        "River cross section":        {"linewidth": 2.0, "alpha": 1.0},
    }

    handles1, labels1 = ax1.get_legend_handles_labels()

    proxy_handles = []
    proxy_labels  = []

    for h, lbl in zip(handles1, labels1):
        style = LEGEND_STYLES.get(lbl, {})
        # fall back to original handle properties if not specified
        color = getattr(h, "get_color", lambda: "black")()
        lw    = style.get("linewidth", getattr(h, "get_linewidth", lambda: 1.5)())
        alpha = style.get("alpha", h.get_alpha() if h.get_alpha() is not None else 1.0)
        ls    = getattr(h, "get_linestyle", lambda: "solid")()
        mk    = getattr(h, "get_marker",   lambda: "None")()
        proxy = Line2D([0], [0],
                       color=color,
                       linewidth=lw,
                       alpha=alpha,
                       linestyle=ls,
                       marker=mk)
        proxy_handles.append(proxy)
        proxy_labels.append(lbl)

    leg = ax1.legend(proxy_handles, proxy_labels,
                     prop=fp, loc='upper right', bbox_to_anchor=(1, 0.95))
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    # title (timestamp)
    plt.text(0.5, 1.05, PresentTime, {'color':'red'}, ha='center', va='center',
             fontproperties=fp, clip_on=False, transform=plt.gca().transAxes)

    return plt

# ------------------------- batch runner (fixed start/end) -------------------------

def run_batch_fixed_range(
    HomeDir,
    RsltDir_raw,
    RsltDir_bc,
    ObsData,
    start_ts,
    end_ts,
    pf_dt_min,
    Pn,
    BT_dy,
    FT_dy,
    RRI_dt_min,
    out_dir
):
    """
    Loop from start_ts to end_ts (inclusive) stepping pf_dt_min minutes.
    For each PresentTimeTxt, call DrawHydro and save a PNG.
    Uses both RAW (RsltDir_raw) and Bias-Corrected (RsltDir_bc) results.
    """
    os.makedirs(out_dir, exist_ok=True)

    # read for naming
    cfg = os.path.join(HomeDir, "PythonCode", "DrawingConfig.ini")
    cp = configparser.ConfigParser()
    cp.read(cfg, encoding="utf-8")
    prediction_point = cp.get('BasicInfo', 'StationName')
    elem = cp.get('BasicInfo', 'ElemQorH')

    start_dt = datetime.datetime.strptime(start_ts, '%Y%m%d%H%M')
    end_dt   = datetime.datetime.strptime(end_ts,   '%Y%m%d%H%M')

    idx = 0
    for ts in daterange_ts(start_dt, end_dt, pf_dt_min):
        idx += 1
        try:
            pl = DrawHydro(
                PresentTimeTxt=ts,
                ConvQ2H=None,
                HomeDir=HomeDir,
                Hydro_f_type=None, Hydro_f=None, Hydro_locNo=None,
                RsltDir_raw=RsltDir_raw,
                RsltDir_bc=RsltDir_bc,
                ObsData=ObsData,
                HQData=None, SecData=None,
                Pn=Pn, nStates=0, ls_StatesNo=[],
                BT_dy=BT_dy, FT_dy=FT_dy,
                ForecastType=None, OrgPn=None,
                RRI_dt_min=RRI_dt_min, PF_dt_min=pf_dt_min,
                MsgBox=None
            )
            out_path = os.path.join(out_dir, f"{prediction_point}_{elem}_{ts}.png")
            pl.savefig(out_path, dpi=300, bbox_inches="tight")
            pl.close()
            print(f"[ok] ({idx}) saved {out_path}")
        except Exception as e:
            print(f"[skip] ({idx}) {ts}: {e}")

# ------------------------- run as script -------------------------

if __name__ == "__main__":
    # ==== EDIT THESE PATHS / SETTINGS ====
    HOME_DIR   = r"C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/GFS_16_0.25_2024"  # has PythonCode/DrawingConfig.ini
    RSLT_DIR_RAW  = os.path.join(HOME_DIR, "Results")       # root with YYYY/MM/DD subfolders (RAW GFS)
    RSLT_DIR_BC   = os.path.join(HOME_DIR, "Results_BC")    # root with YYYY/MM/DD subfolders (Bias-Corrected GFS)
    OBS_FILE      = os.path.join(HOME_DIR, "ObsData/WaterLevel/ObsWL.csv")  # obs CSV with two columns: datetime,value

    START_TS   = "202408170000"  # inclusive; format YYYYMMDDHHMM
    END_TS     = "202408170000"  # inclusive
    PF_DT_MIN  = 1440            # step minutes between successive figures

    PN         = 16
    BT_DY      = 1
    FT_DY      = 10
    RRI_DT_MIN = 60

    OUT_DIR    = os.path.join(HOME_DIR, "Drawing")
    # =====================================

    run_batch_fixed_range(
        HomeDir=HOME_DIR,
        RsltDir_raw=RSLT_DIR_RAW,
        RsltDir_bc=RSLT_DIR_BC,
        ObsData=OBS_FILE,
        start_ts=START_TS,
        end_ts=END_TS,
        pf_dt_min=PF_DT_MIN,
        Pn=PN,
        BT_dy=BT_DY,
        FT_dy=FT_DY,
        RRI_dt_min=RRI_DT_MIN,
        out_dir=OUT_DIR
    )

