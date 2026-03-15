# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt, font_manager
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import datetime
import glob
import sys
import configparser
import copy
from PIL import Image
import re
import statistics

def ConfigPara(f):
    ConfigValSet = configparser.ConfigParser()
    ConfigValSet.read(f, encoding='utf-8')
# --- [BasicInfo] ---
    RiverName   = str(ConfigValSet.get('BasicInfo', 'RiverName'))
    StationName = str(ConfigValSet.get('BasicInfo', 'StationName'))
    ElemQorH = str(ConfigValSet.get('BasicInfo', 'ElemQorH'))
    GraLanguage = str(ConfigValSet.get('BasicInfo', 'Language'))
# --- [FigTime] ---
    BeforeDays  = int(ConfigValSet.get('FigTime', 'BeforeDays'))
    AfterDays   = int(ConfigValSet.get('FigTime', 'AfterDays'))
# --- [GraphRange] ---
    Min_Y1    = float(ConfigValSet.get('GraphRange', 'Min_Y1'))
    Max_Y1    = float(ConfigValSet.get('GraphRange', 'Max_Y1'))
    MaxV_Y1   = float(ConfigValSet.get('GraphRange', 'MaxV_Y1'))
    Tick_Y1   = float(ConfigValSet.get('GraphRange', 'Tick_Y1'))
    Min_Rain  = int(ConfigValSet.get('GraphRange', 'Min_Rain'))
    Max_Rain  = int(ConfigValSet.get('GraphRange', 'Max_Rain'))
    MaxV_Rain = int(ConfigValSet.get('GraphRange', 'MaxV_Rain'))
    Tick_Rain = float(ConfigValSet.get('GraphRange', 'Tick_Rain'))
# --- [PFcondition] ---
    ForecastHyd_WtMean_Flg   = int(ConfigValSet.get('PFcondition', 'ForecastHyd_WtMean_Flg'))
    ForecastHyd_OrgP_Flg   = int(ConfigValSet.get('PFcondition', 'ForecastHyd_OrgP_Flg'))
    ForecastHyd_BestP_Flg   = int(ConfigValSet.get('PFcondition', 'ForecastHyd_BestP_Flg'))
    AllParticleFlg  = int(ConfigValSet.get('PFcondition', 'AllParticleFlg'))
    ForecastRainFlg = int(ConfigValSet.get('PFcondition', 'ForecastRainFlg'))
    EnsembleRainFlg = int(ConfigValSet.get('PFcondition', 'EnsembleRainFlg'))
    BackTimeFlg     = int(ConfigValSet.get('PFcondition', 'BackTimeFlg'))
# --- [WarningLevel] ---
    nLevel  = int(ConfigValSet.get('WarningLevel', 'nLevel'))
    ls_Level_Value, ls_Level_Name = [],[]
    for iLevel in range(nLevel):
        tmp_val1 = 'Level_' + str(iLevel+1) + '_Value'
        tmp_val2 = ConfigValSet.get('WarningLevel',tmp_val1)
        ls_Level_Value.append(float(tmp_val2))
        tmp_name1 = 'Level_' + str(iLevel+1) + '_Name'
        tmp_name2 = ConfigValSet.get('WarningLevel',tmp_name1)
        ls_Level_Name.append(tmp_name2)
# --- [OfflineCalc] ---
    RRIofflineCalc = int(ConfigValSet.get('OfflineCalc', 'RRIofflineCalc'))
    OfflineFile    = str(ConfigValSet.get('OfflineCalc', 'OfflineFile'))
# --- [RivSection] ---
    RivSecFlag = int(ConfigValSet.get('RivSection', 'RivSecFlag'))
    RivSecFile = str(ConfigValSet.get('RivSection', 'RivSecFile'))
# --- [OthersInfo] ---
    RainJ_dt_min = int(ConfigValSet.get('OthersInfo', 'RainJ_dt_min'))
    LegendPosition = int(ConfigValSet.get('OthersInfo', 'LegendPosition'))
    return RiverName, StationName, ElemQorH, GraLanguage, BeforeDays, AfterDays, \
           Min_Y1, Max_Y1, MaxV_Y1, Tick_Y1, Min_Rain, Max_Rain, MaxV_Rain, Tick_Rain, \
           ForecastHyd_WtMean_Flg, ForecastHyd_OrgP_Flg, ForecastHyd_BestP_Flg, AllParticleFlg, \
           ForecastRainFlg, EnsembleRainFlg, BackTimeFlg, \
           nLevel, ls_Level_Value, ls_Level_Name, \
           RRIofflineCalc, OfflineFile, RivSecFlag, RivSecFile, RainJ_dt_min, LegendPosition

def Read_location(file):
    f = open(file, 'r')
    loc_org = f.readlines()
    f.close()
    loc_Name, loc_i, loc_j = [],[],[]
    i = 0
    for loc_iLine in loc_org:
        i += 1
        loc_iLine = loc_iLine.strip()
        loc_Value = re.split(r'\s+', loc_iLine)
        #print loc_Value
        loc_Name.append(loc_Value[0])
        loc_i.append(int(loc_Value[1]))
        loc_j.append(int(loc_Value[2]))
    # print('Output point is ' + str(len(loc_org)) + ': ' + str(loc_Name))
    return loc_Name

def DrawHydro(PresentTimeTxt, ConvQ2H, HomeDir, Hydro_f_type, Hydro_f, Hydro_locNo, RsltDir, ObsData, \
              HQData, SecData, Pn, nStates, ls_StatesNo, BT_dy, FT_dy, ForecastType, OrgPn, \
              RRI_dt_min, PF_dt_min, MsgBox):
    config_f = HomeDir + '/PythonCode/DrawingConfig.ini'
    #######Read for 'DrawingConfig.ini'#################################################
    RiverName, PredictionPoint, ElemQorH, GraLanguage, BeforeDays, AfterDays, \
    Min_Y1, Max_Y1, MaxV_Y1, Tick_Y1, Min_Rain, Max_Rain, MaxV_Rain, Tick_Rain, \
    ForecastHyd_WtMean_Flg, ForecastHyd_OrgP_Flg, ForecastHyd_BestP_Flg, AllParticleFlg, \
    ForecastRainFlg, EnsembleRainFlg, BackTimeFlg, \
    nLevel, ls_Level_Value, ls_Level_Name, \
    RRIofflineCalc, OfflineFile, RivSecFlag, RivSecFile, RainJ_dt_min, LegendPosition \
    = ConfigPara(config_f)

    ## use format
    # plt.style.use('seaborn-whitegrid')
    # plt.style.use('seaborn-colorblind')
    # plt.style.use('seaborn-ticks')

    # Size: X means time-series, y1 means Hydro axis, y2 means Hyeto axis
    xNumberSize  = 15
    xLabelSize   = 18
    y1NumberSize = 20
    y1LabelSize  = 20
    y2NumberSize = 18
    y2LabelSize  = 20

    # label space for x-axis
    X_major_dt_hr = 24   # [hr]
    X_minor_dt_hr = 4  # [hr]
    y1_major_dt   = 1.0 # [m]
    y1_minor_dt   = 0.2 # [m]

    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams["font.size"] = 15
    plt.rcParams['xtick.direction'] = 'inout'   #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'inout'   #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 0.5     #x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 0.5     #y軸主目盛り線の線幅
    plt.rcParams['axes.linewidth'] = 0.5        # 軸の線幅edge linewidth。囲みの太さ
    #plt.rcParams['font.size'] = 18             #Font size (common??)

    ##Size of outout graph
    #fig = plt.figure(figsize=(12,7), dpi=300)
    fig = plt.figure(figsize=(12,9), dpi=300)

    ##日本語表記をするための設定（C:\Windows\Fonts\の中にSourceHanCodeJP-Regular.otfがない場合は書けない．かつwindows）
    fp = font_manager.FontProperties(fname=r'C:\Users\Intel NUC\Desktop\bikram\times new roman.ttf')

    # check the output point in 'location.txt' >>> if 'StationName' do not exist in locations, StationKey change blank.
    # StatoinKey: for the search files, StationName: for the name of the output files
    Location_f = HomeDir + '/Particles/Particle00001/RRI/location.txt'
    if os.path.exists(Location_f) == False:
        print ("'location.txt' file does NOT exist in 'Particle0001'!!")
        sys.exit()
    Loc_Name = Read_location(Location_f)
    if (PredictionPoint in Loc_Name) == True:
        StationKey = '_'
    else:
        StationKey = '_'

    # forecast file with weight mean
    if ElemQorH == 'H':
        Result_f_mean = '_WtMean_wl'
        Result_f_org = '_OrgP???_wl'
        Result_f_best = '_BestP???_wl'
        Result_f_all = '_AllP_wl'
    elif ElemQorH == 'Q':
        Result_f_mean = '_WtMean_qr'
        Result_f_org = '_OrgP???_qr'
        Result_f_best = '_BestP???_qr'
        Result_f_all = '_AllP_qr'

    #Times for fig###
    PresentTime = datetime.datetime.strptime(PresentTimeTxt, '%Y%m%d%H%M')
    StartTime = PresentTime - datetime.timedelta(days = BeforeDays)
    StartTimeTxt = '{:%Y%m%d%H%M}'.format(StartTime)
    EndTime = PresentTime + datetime.timedelta(days = AfterDays)
    EndTimeTxt = '{:%Y%m%d%H%M}'.format(EndTime)
    deltaSec = (EndTime - StartTime).total_seconds()
    deltaDay = int(deltaSec/(24*3600)) + 2
    YearName = StartTime.year
    nSkipBT = BT_dy * (1440 / RRI_dt_min) - 1


    # Configuration of figure
    ax1 = fig.add_subplot(111)
    ##軸目盛の設定，主軸補助線を自動で割り振る
    ax1.tick_params(axis='x',labelsize=xNumberSize,  which='both', direction='inout')
    ax1.tick_params(axis='y',labelsize=y1NumberSize, which='major',direction='out', length=5, width=1, color='black')
    ##X-axis scale: datetime
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval = X_major_dt_hr))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval = X_minor_dt_hr))
    ##Y-axis scale: value
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y1_major_dt))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(y1_minor_dt))
    ax1.minorticks_on() # minor scale: ON
    ##グリッド線の設定
    ax1.grid(which = "major", axis = "y", color = "gray", alpha = 1.0, linestyle = "dashed", linewidth = 1.0)
    ax1.grid(which = "major", axis = "x", color = "gray", alpha = 1.0, linestyle = "dashed", linewidth = 1.0)
    ax1.grid(which = "minor", axis = "x", color = "gray", alpha = 1.0, linestyle = "dashed", linewidth = 0.5)
    ##x軸のメモリの間隔及びラベルの設定
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ##X軸とY軸の範囲設定
    ax1.set_ylim(Min_Y1, Max_Y1)
    ax1.set_xlim(StartTime, EndTime)
    #ax1.set_xlabel('')
    ax1.set_xlabel(YearName, fontsize=xLabelSize+2)
    ax1.xaxis.set_label_coords(0, -0.081)
    if ElemQorH == 'H':
        if GraLanguage == 'Eng':
            ax1.set_ylabel('Water Level [m]', fontsize=y1LabelSize, fontweight = 'bold')
        elif GraLanguage == 'Jpn':
            ax1.set_ylabel(u"水 位（読み値）[m]", fontsize=y1LabelSize, fontweight = 'bold', fontproperties=fp)
    elif ElemQorH == 'Q':
        if GraLanguage == 'Eng':
            ax1.set_ylabel('Discharge [$m^3$/s]', fontsize=y1LabelSize, fontweight = 'bold')
        elif GraLanguage == 'Jpn':
            ax1.set_ylabel(u"流 量 [$m^3$/s]", fontsize=y1LabelSize, fontweight = 'bold', fontproperties=fp)
    ## label of y-axis (range: Min_Y1-MaxV_Y1+1, label: Min_Y1-Tick_Y1)
    ax1.set_yticks(np.arange(Min_Y1, MaxV_Y1+1, Tick_Y1))     # Min~MaxVをTick刻みでラベル目盛作成

    #2nd axis: hyetograph
    ax2 = ax1.twinx()
    ax2.tick_params(axis='x', labelsize=xNumberSize)
    ax2.tick_params(axis='y', labelsize=y2NumberSize, which='major',direction='out', length=5, width=1, color='black')
    ax2.set_ylim(Max_Rain,Min_Rain)
    ax2.set_xlim(StartTime,EndTime)
    #ax2.set_xlabel('')
    if GraLanguage == 'Eng':
        ax2.set_ylabel('Rainfall [mm/hr]', fontsize=y2LabelSize, fontweight = 'bold')
    elif GraLanguage == 'Jpn':
        ax2.set_ylabel(u'降雨強度 [mm/hr]', fontsize=y2LabelSize, fontweight = 'bold', fontproperties=fp)
    ax2.yaxis.set_label_coords(1.06, 0.80)  #ハイエト軸の位置を上にずらす。
    ax2.xaxis.set_ticklabels([])
    #ax2.axes.get_xaxis().set_ticks([])
    ax2.set_yticks(np.arange(Min_Rain, MaxV_Rain+1, Tick_Rain))    # Min~MaxVをTick刻みでラベル目盛作成
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    # X-axis grid & sub-grid for time series
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval = X_major_dt_hr))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval = X_minor_dt_hr))
    ax2.grid(which = "major", axis = "y", color = "gray", alpha = 0.2,linestyle = "solid", linewidth = 0.1)

    #########Draw the river section#################################
    if RivSecFlag == 1:
        RivSec_f = HomeDir + '/' + RivSecFile
        if(os.path.exists(RivSec_f)):
            SecElev = np.loadtxt(RivSec_f, dtype='float', delimiter=",", skiprows=1)
            nPoint = SecElev.shape[0]
            nCol = SecElev.shape[1]
            ZeroElev = SecElev[0, 2]
            SecZeroH = copy.deepcopy(SecElev)               #preparation
            SecZeroH[:,1] = SecZeroH[:,1] - ZeroElev    #elev >>> zeroH
            MinLength = SecZeroH[:,0].min()
            MaxLength = SecZeroH[:,0].max()
            #convert "x-length" to "time-axis"
            StartTime4Fig = StartTime + datetime.timedelta(hours = 6)
            EndTime4Fig = EndTime - datetime.timedelta(hours = 6)
            SecTimeX,SecZeroY = [],[]
            for iPoint in range(nPoint):
                dt_s = (EndTime4Fig - StartTime4Fig).total_seconds()/(MaxLength-MinLength)*(SecZeroH[iPoint,0]-MinLength)
                SecTimeX.append(StartTime4Fig + datetime.timedelta(seconds = dt_s))
                SecZeroY.append(SecZeroH[iPoint,1])
        if GraLanguage == 'Eng':
            ax1.plot(SecTimeX, SecZeroY, lw=3, alpha=1.0, label='River cross section',linestyle='solid',color='dimgray')
        elif GraLanguage == 'Jpn':
            ax1.plot(SecTimeX, SecZeroY, lw=3, alpha=1.0, label='河道横断面',linestyle='solid',color='dimgray')
    #########Draw the river section#################################

    #########ObsWL###################################
    ObsFile = ObsData
    f = np.loadtxt(ObsFile, dtype='float', delimiter=",", skiprows=1)
    f_T = f.T
    ls_ObsDateTxt = f_T[0]
    ls_ObsData = f_T[1]
    ls_ObsVal,ls_ObsDatetimeTxt,ls_ObsDatetime = [],[],[]
    
    for iObs in range(len(ls_ObsDateTxt)):
        DoObsDatetimeTxt = str(int(ls_ObsDateTxt[iObs]))
        ls_ObsDatetimeTxt.append(str(int(ls_ObsDateTxt[iObs])))
        ls_ObsDatetime.append(datetime.datetime.strptime(DoObsDatetimeTxt, '%Y%m%d%H%M'))
        ls_ObsVal.append(ls_ObsData[iObs])
    if ElemQorH == 'H':
        if GraLanguage == 'Eng':
            ax1.plot(ls_ObsDatetime,ls_ObsVal,lw=2,alpha=0.6, label='RID Observed W.L.', linestyle="solid", color='black',
            marker="^",markerfacecolor='black',markersize=6,markeredgewidth=0.5,markeredgecolor='white')
        elif GraLanguage == 'Jpn':
            ax1.plot(ls_ObsDatetime,ls_ObsVal,lw=2,alpha=0.6, label='実績水位', linestyle="",
            marker="^",markerfacecolor='black',markersize=6,markeredgewidth=0.5,markeredgecolor='white')
    elif ElemQorH == 'Q':
        if GraLanguage == 'Eng':
            ax1.plot(ls_ObsDatetime,ls_ObsVal,lw=2,alpha=0.6, label='RID Observed discharge', linestyle="solid", color='black',
            marker="^",markerfacecolor='black',markersize=6,markeredgewidth=0.5,markeredgecolor='white')
        elif GraLanguage == 'Jpn':
            ax1.plot(ls_ObsDatetime,ls_ObsVal,lw=2,alpha=0.6, label='実績流量', linestyle="",
            marker="^",markerfacecolor='black',markersize=6,markeredgewidth=0.5,markeredgecolor='white')
    ##########ObsWL###################################

    #########CalcWL on Offline########################
    if RRIofflineCalc == 1:
        CalcOfflineFile = HomeDir + '/' + OfflineFile
        if os.path.exists(CalcOfflineFile) == True:
            AllFile = pd.read_csv(CalcOfflineFile, header = 0, parse_dates=[0])
            List_Datetime = AllFile['datetime'].tolist()
            #df_Datetime = AllFile['datetime']
            df_Offline = AllFile.iloc[:,1:]
            List_Offline = df_Offline.values.tolist()
            if ElemQorH == 'H':
                if GraLanguage == 'Eng':
                    ax1.plot(List_Datetime, List_Offline, lw=1.5, alpha=0.6, label='HII Sim W.L.',linestyle='solid',color='blue')
                elif GraLanguage == 'Jpn':
                    ax1.plot(List_Datetime, List_Offline, lw=1.5, alpha=0.6, label='再現計算水位',linestyle='solid',color='blue')
            if ElemQorH == 'Q':
                if GraLanguage == 'Eng':
                    ax1.plot(List_Datetime, List_Offline, lw=1.5, alpha=0.6, label='HII Sim Discharge',linestyle='solid',color='blue')
                elif GraLanguage == 'Jpn':
                    ax1.plot(List_Datetime, List_Offline, lw=1.5, alpha=0.6, label='再現計算流量',linestyle='solid',color='blue')
        elif os.path.exists(CalcOfflineFile) == False:
            print(('!!Caution!!! Not found " ' + OfflineFile +' ". \n  >>> Not draw the offline calc.!'))
    #########CalcWL on Offline########################

    #########Other informations#######################
    OtherInfoFile = "./CalcOthers.csv"
    if(os.path.exists(OtherInfoFile)):
        df_AllInfo = pd.read_csv(OtherInfoFile, header=0, parse_dates=[0])
        nInfo = len(df_AllInfo.columns) - 1
        #get for the datetime list
        df_AllInfo_t = df_AllInfo.T
        List_tmp = df_AllInfo['datetime'].tolist()
        List_datetime4other = []
        for StrDatetime in List_tmp[1:]:
            List_datetime4other.append(datetime.datetime.strptime(StrDatetime, '%Y/%m/%d %H:%M'))
        #get for the line type to list
        List_Name = df_AllInfo_t.index.tolist()
        List_Type = df_AllInfo_t[0].tolist()
        #Data for the figure
        for DataName in List_Name[1:]:
            List_tmp = df_AllInfo[DataName].tolist()
            LineType = List_tmp[0]
            LineData = [float(nTmp) for nTmp in List_tmp[1:]]
            if LineType == 'point':
                ax1.scatter(List_datetime4other, LineData, s=1.0, c="silver", marker='o', alpha=0.5, edgecolors='none')
            elif LineType == 'solid' or LineType == 'dashed' or LineType == 'dashdot' or LineType == 'dotted':
                ax1.plot(List_datetime4other,LineData,linewidth = 1.5, alpha = 0.8, linestyle = LineType, color = 'gray')
    #########Other informations#######################

    #########Warning level#################################
    ls_LevelColor = ['black','black','black','black','red','darkviolet','gray','gray','gray','gray']
    ls_LevelStyle = ["solid","solid","dashed","dashdot","solid","solid","dotted","dotted","dotted","dotted"]
    ls_LevelWidth = [1.5,1.0,1.0,1.0,1.5,1.0,0.5,0.5,0.5,0.5]
    for iLevel in range(nLevel):
        LevelValue = ls_Level_Value[iLevel]
        LevelName  = ls_Level_Name[iLevel]
        LevelColor = ls_LevelColor[iLevel]
        LevelStyle = ls_LevelStyle[iLevel]
        LevelWidth = ls_LevelWidth[iLevel]
        if LevelValue > -10:
            y0 = [LevelValue]*deltaDay
            x0 = pd.date_range(str(StartTime),periods=deltaDay,freq='d')
            ax1.plot(x0,y0,linewidth = LevelWidth, alpha = 0.8, linestyle = LevelStyle, color = LevelColor)
            ax1.text(0.02, (LevelValue - Min_Y1)/(Max_Y1 - Min_Y1)+0.01, LevelName + ' (' + format(LevelValue,'.2f') + 'm)', size=15,
                    horizontalalignment = 'left', verticalalignment = 'baseline', clip_on = False, 
                    alpha = 0.8, transform = plt.gca().transAxes)
    #########Warning level#################################

    #########Observed rainfall###################################
    DoTime = StartTime
    nStepJ = float(PF_dt_min/RainJ_dt_min)   #10/30
    if nStepJ < 1: nStepJ = 1
    # PresentTime_r = datetime.datetime.strptime(str(PresentTimeTxt), '%Y%m%d%H%M')
    ls_TimeJ,ls_RainJ = [],[]
    while DoTime <= PresentTime:
        ls_HyetoDate_int,ls_HyetoDate,ls_HyetoDatetime,ls_HyetoRain = [],[],[],[]
        # Result dir
        RsltDir_Do = RsltDir + '/' + DoTime.strftime("%Y") + '/' + DoTime.strftime("%m") + '/' + DoTime.strftime("%d")
        DoTimeTxt = '{:%Y%m%d%H%M}'.format(DoTime)
        Hyeto_RainJ_tmp = RsltDir_Do + '/' + DoTimeTxt + StationKey + PredictionPoint + '_OrgP???_rain_Pn???_BT??d_FT??d.csv'
        Hyeto_RainJ = glob.glob(Hyeto_RainJ_tmp)
        if len(Hyeto_RainJ) == 0:
            DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
            continue
        Hyeto_RainJ_f = Hyeto_RainJ[0]
        if os.path.exists(Hyeto_RainJ_f) == False:
            DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
            continue
        df_RainJ = pd.read_csv(Hyeto_RainJ_f)
        ls_HyetoDate_int = df_RainJ.iloc[:, 0].to_list()
        ls_HyetoDate = [str(i) for i in ls_HyetoDate_int]
        ls_HyetoDatetime = [datetime.datetime.strptime(i, '%Y%m%d%H%M') for i in ls_HyetoDate]
        ls_HyetoRain = df_RainJ.iloc[:, 1].to_list()
        DotimeRain_idx = ls_HyetoDate_int.index(int(DoTimeTxt))
        for iStep in reversed(range(int(nStepJ))):
            ls_TimeJ.append(ls_HyetoDatetime[DotimeRain_idx-1-iStep])   # -1: for bar figure in edge
            ls_RainJ.append(ls_HyetoRain[DotimeRain_idx-iStep])
        DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
    ax2.grid(which = "major", axis = "y", color = "gray", alpha = 0.2,linestyle = "solid", linewidth = 0.1)
    if PF_dt_min <= RainJ_dt_min:
        dWidth = 1/24*(PF_dt_min/60)    # [1] = 1day
    elif PF_dt_min > RainJ_dt_min:
        dWidth = 1/24*(RainJ_dt_min/60) # [1] = 1day
    if GraLanguage == 'Eng':
        ax2.bar(ls_TimeJ,ls_RainJ,label='Rainfall [mm/hr]',width=dWidth, color="Blue", edgecolor = 'white', align="edge", linewidth = 0.1)
    elif GraLanguage == 'Jpn':
        ax2.bar(ls_TimeJ,ls_RainJ,label="降雨強度 [mm/hr]",width=dWidth, color="Blue", edgecolor = 'white', align="edge", linewidth = 0.1)
    #########Observed rainfall###################################

    # ##########Predicted rainfall at present time: bar graph###################################
    # nSkipBT = int(BT_dy * (60 / RainJ_dt_min))
    # ls_HyetoDate_int = []
    # ls_HyetoDate = []
    # ls_HyetoDatetime = []
    # ls_HyetoSec = []
    # ls_HyetoRain = []
    # ls_TimeY = []
    # ls_RainY = []
    # nStepY = float(PF_dt_min/RainY_dt_min)   #10/60
    # if nStepY < 1: nStepY = 1
    # dWidth = 1/24*(RainY_dt_min/60) # [1] = 1day
    # PresentTimeTxt = '{:%Y%m%d%H%M}'.format(PresentTime)
    # RsltDir_Present = RsltDir + '/' + PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
    # Hyeto_RainY_tmp = RsltDir_Present + '/' + PresentTimeTxt + StationKey + PredictionPoint + '_OrgP???_rain_Pn???_BT??h_FT??h.csv'
    # Hyeto_RainY = glob.glob(Hyeto_RainY_tmp)
    # Hyeto_RainY_f = Hyeto_RainY[0]
    # df_RainY = pd.read_csv(Hyeto_RainY_f)
    # ls_HyetoDate_int = df_RainY.iloc[:, 0].to_list()
    # ls_HyetoDate = [str(i) for i in ls_HyetoDate_int]
    # ls_HyetoRain = df_RainY.iloc[:, 1].to_list()
    # PresentTime_idx = ls_HyetoDate_int.index(int(PresentTimeTxt))
    # # YYYYMMDDhhmm >>> second & datetime
    # Ln = len(ls_HyetoDate)
    # for iLn in range(Ln):
    #     Hyeto_Datetime = datetime.datetime.strptime(ls_HyetoDate[iLn], '%Y%m%d%H%M')
    #     Hyeto_InitTime = datetime.datetime.strptime(ls_HyetoDate[0], '%Y%m%d%H%M')
    #     delta_datetime = Hyeto_Datetime - Hyeto_InitTime
    #     delta_sec = delta_datetime.total_seconds()
    #     ls_HyetoSec.append(delta_sec)
    #     ls_HyetoDatetime.append(datetime.datetime.strptime(ls_HyetoDate[iLn], '%Y%m%d%H%M'))
    # Ln = len(ls_HyetoRain) - int(BT_dy * (60/RainJ_dt_min)) - 1
    # for iLn in range(Ln):
    #     ls_TimeY.append(ls_HyetoDatetime[PresentTime_idx+iLn])    # 棒グラフだから実況はなし。また、edgeだから時間は1ステップ戻す（見た目の問題）。
    #     ls_RainY.append(ls_HyetoRain[PresentTime_idx + 1 + iLn])             # 棒グラフだから実況はなし。
    # ax2.bar(ls_TimeY,ls_RainY,label='Forecasted Rainfall',width=dWidth, color="magenta", edgecolor = 'white', align="edge", linewidth = 0.1)
    # ##########Current predicted rainfall: bar graph###################################

    ##########(option)Ensemble rainfall###################################
    if EnsembleRainFlg == 1:
        ls_HyetoDate_int,ls_HyetoDate,ls_HyetoRain_tmp,ls_HyetoSec,ls_HyetoDatetime,ls_HyetoRain = [],[],[],[],[],[]
        PresentTimeTxt = '{:%Y%m%d%H%M}'.format(PresentTime)
        RsltDir_Present = RsltDir + '/' + PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
        Hyeto_RainY_tmp = RsltDir_Present + '/' + PresentTimeTxt + StationKey + PredictionPoint + '_AllP_rain_Pn???_BT??d_FT??d.csv'
        # 202306011200_Tsukinowa_AllP_rain_Pn008_BT03h_FT06h.csv
        Hyeto_RainY = glob.glob(Hyeto_RainY_tmp)
        if len(Hyeto_RainY) != 0:
            Hyeto_RainY_f = Hyeto_RainY[0]
            if os.path.exists(Hyeto_RainY_f) != False:
                df_RainY = pd.read_csv(Hyeto_RainY_f)
                ls_HyetoDate_int = df_RainY.iloc[:, 0].to_list()
                ls_HyetoDate = [str(i) for i in ls_HyetoDate_int]
                ls_HyetoRain_tmp = df_RainY.iloc[:, 1:].to_numpy().tolist()
                ls_HyetoRain = np.array(ls_HyetoRain_tmp).T.tolist()
                # YYYYMMDDhhmm >>> second & datetime
                Ln = len(ls_HyetoDate)
                for iLn in range(Ln):
                    Hyeto_Datetime = datetime.datetime.strptime(ls_HyetoDate[iLn], '%Y%m%d%H%M')
                    Hyeto_InitTime = datetime.datetime.strptime(ls_HyetoDate[0], '%Y%m%d%H%M')
                    delta_datetime = Hyeto_Datetime - Hyeto_InitTime
                    delta_sec = delta_datetime.total_seconds()
                    ls_HyetoSec.append(delta_sec)
                    ls_HyetoDatetime.append(datetime.datetime.strptime(ls_HyetoDate[iLn], '%Y%m%d%H%M'))
                for iPn in range(Pn):
                    ls_TimeY,ls_RainY = [],[]
                    ls_HyetoRain_p = ls_HyetoRain[iPn]
                    Ln = len(ls_HyetoRain_p)
                    for iLn in range(Ln):
                        ls_TimeY.append(ls_HyetoDatetime[iLn])
                        ls_RainY.append(ls_HyetoRain_p[iLn])
                    ax2.plot(ls_TimeY,ls_RainY,lw=0.5, alpha=0.6, label='Ensemble Rainfall',linestyle="solid", marker="",markersize=5,color='orange')
    ##########(option)Ensemble rainfall###################################

    ########## Past predicted rainfall: plot graph ###################################
    if ForecastRainFlg == 1:
        # nSkipBT = int(BT_dy * (60 / RainJ_dt_min))
        DoTime = StartTime
        while DoTime <= PresentTime:
            ls_HyetoDate_int,ls_HyetoDate,ls_HyetoSec,ls_HyetoDatetime,ls_HyetoRain,ls_TimeY,ls_RainY = [],[],[],[],[],[],[]
            DoTimeTxt = '{:%Y%m%d%H%M}'.format(DoTime)
            RsltDir_Do = RsltDir + '/' + DoTime.strftime("%Y") + '/' + DoTime.strftime("%m") + '/' + DoTime.strftime("%d")
            Hyeto_RainY_tmp = RsltDir_Do + '/' + DoTimeTxt + StationKey + PredictionPoint + '_OrgP???_rain_Pn???_BT??d_FT??d.csv'
            Hyeto_RainY = glob.glob(Hyeto_RainY_tmp)
            if len(Hyeto_RainY) == 0:
                DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
                continue
            Hyeto_RainY_f = Hyeto_RainY[0]
            if os.path.exists(Hyeto_RainY_f) == False:
                DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
                continue
            df_RainY = pd.read_csv(Hyeto_RainY_f)
            ls_HyetoDate_int = df_RainY.iloc[:, 0].to_list()
            ls_HyetoDate = [str(i) for i in ls_HyetoDate_int]
            ls_HyetoRain = df_RainY.iloc[:, 1].to_list()
            DotimeRain_idx = ls_HyetoDate_int.index(int(DoTimeTxt))
            # YYYYMMDDhhmm >>> second & datetime
            Ln = len(ls_HyetoDate)
            for iLn in range(Ln):
                Hyeto_Datetime = datetime.datetime.strptime(ls_HyetoDate[iLn], '%Y%m%d%H%M')
                Hyeto_InitTime = datetime.datetime.strptime(ls_HyetoDate[0], '%Y%m%d%H%M')
                delta_datetime = Hyeto_Datetime - Hyeto_InitTime
                delta_sec = delta_datetime.total_seconds()
                ls_HyetoSec.append(delta_sec)
                ls_HyetoDatetime.append(datetime.datetime.strptime(ls_HyetoDate[iLn], '%Y%m%d%H%M'))
            Ln = len(ls_HyetoRain) - int(BT_dy * (1440/RainJ_dt_min))
            for iLn in range(Ln):
                ls_TimeY.append(ls_HyetoDatetime[DotimeRain_idx+iLn])
                ls_RainY.append(ls_HyetoRain[DotimeRain_idx + iLn])
            if DoTime == PresentTime:
                ax2.plot(ls_TimeY,ls_RainY,lw=1.5, alpha=0.5, label='Forecasted Rainfall',linestyle="solid", marker="",markersize=5,color='Red')
            elif DoTime != PresentTime:
                ax2.plot(ls_TimeY,ls_RainY,lw=0.5, alpha=0.5, label='Forecasted Rainfall',linestyle="solid", marker="",markersize=5,color='lightcoral')
            # Update time
            DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
    ########## Past predicted rainfall: plot graph ###################################

    #########Rivebed at present time and prediction time##################################
    if (7 in ls_StatesNo) == True:
        #---Present time: estimated riverbed by paticle filter-------------
        iLs = ls_StatesNo.index(7)
        DoTime = StartTime
        x_TimeSeries,Riverbed_Max,Riverbed_Min,Riverbed_Mid,Riverbed_Mean = [],[],[],[],[]
        while DoTime <= PresentTime:
            ls_HyetoDatetime,ls_HyetoRain,ls_TimeJ,ls_RainJ,x_DoTime = [],[],[],[],[]
            DoTimeTxt = '{:%Y%m%d%H%M}'.format(DoTime)
            RsltDir_Do = RsltDir + '/' + DoTime.strftime("%Y") + '/' + DoTime.strftime("%m") + '/' + DoTime.strftime("%d")
            States_f = RsltDir_Do + '/' + DoTimeTxt + '_' + PredictionPoint + '_ParticleDistribution' + str(iLs+1) + '.log'
            Sediment_J = glob.glob(States_f)
            if len(Sediment_J) == 0:
                DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
                continue
            Sediment_J_f = Sediment_J[0]
            if os.path.exists(Sediment_J_f) == False:
                DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
                continue
            States_Posterior = tail2list(Sediment_J_f)
            Riverbed_p = tail2list(Sediment_J_f) + min(SecZeroY)
            # Particle (scatter)
            x_DoTime = [DoTime - datetime.timedelta(hours = BT_dy)] * Pn
            ax1.scatter(x_DoTime, Riverbed_p, marker='o', s=0.5, c="magenta", alpha = 0.5, edgecolors='none')
            # Max, Min, Medium, Mean (plot)
            x_TimeSeries.append(DoTime - datetime.timedelta(hours = BT_dy))
            Riverbed_Max.append(max(States_Posterior) + min(SecZeroY))
            Riverbed_Min.append(min(States_Posterior) + min(SecZeroY))
            Riverbed_Mid.append(statistics.median(States_Posterior) + min(SecZeroY))
            Riverbed_Mean.append(statistics.mean(States_Posterior) + min(SecZeroY))
            # Update time
            DoTime = DoTime + datetime.timedelta(minutes = PF_dt_min)
        ax1.plot(x_TimeSeries, Riverbed_Max, linewidth = 1.0, alpha = 0.8, linestyle = 'dashed', color = 'gray')
        ax1.plot(x_TimeSeries, Riverbed_Min, linewidth = 1.0, alpha = 0.8, linestyle = 'dashed', color = 'gray')
        #ax1.plot(x_TimeSeries, Riverbed_Mid, linewidth = 1.5, alpha = 0.8, linestyle = 'dashdot', color = 'dimgray')
        ax1.plot(x_TimeSeries, Riverbed_Mean,linewidth = 1.5, alpha = 0.8, linestyle = 'dashdot', color = 'dimgray')
        #---Present time: estimated riverbed by paticle filter-------------

        #---Forecast time: predicted riverbed by phisics model-------------
        # SpecialFlg = 0  # <<< It does not work (homework).
        # if SpecialFlg == 1:
        #     AllP_SedimentFile = RsltDir + '/WL-Sediment_conditions' +'/' + PresentTime.strftime('%Y%m%d%H%M') + '_RiverBedDepth_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'h_FT' + str(BT_dy).zfill(2) + 'h.csv'
        #     if BackTimeFlg == 0:
        #         All_sedim = np.loadtxt(AllP_SedimentFile, dtype='float', delimiter=",", skiprows=1+nSkipBT)
        #     elif BackTimeFlg == 1:
        #         All_sedim = np.loadtxt(AllP_SedimentFile, dtype='float', delimiter=",", skiprows=1)
        #     f_T = All_sedim.T
        #     ls_AllSedimDatetimeTxt = f_T[0]
        #     ls_AllSedimData = f_T[1:]
        #     ls_AllSedimDatetime = []
        #     for iStepAll in range(len(ls_AllSedimDatetimeTxt)):
        #         Do4AllDatetimeTxt = str(int(ls_AllSedimDatetimeTxt[iStepAll]))
        #         ls_AllSedimDatetime.append(datetime.datetime.strptime(Do4AllDatetimeTxt, '%Y%m%d%H%M'))
        #     for iPn in range(Pn):
        #         ls_iPnSedimData = []
        #         for iStepAll in range(len(ls_AllSedimDatetimeTxt)):
        #             ls_iPnSedimData.append(ls_AllSedimData[iPn][iStepAll] + np.min(SecZeroH))
        #         ax1.plot(ls_AllSedimDatetime, ls_iPnSedimData, lw=0.5, alpha=0.5, linestyle="solid", color='hotpink')
        #---Forecast time: predicted riverbed by phisics model-------------
    #########Rivebed at present time and prediction time##################################
	
    #########Forecasted wl/qr for all particle##################################
    if AllParticleFlg == 1:
        ls_AllHydroDatetime, ls_AllHydroData = [],[]
        PresentTimeTxt = '{:%Y%m%d%H%M}'.format(PresentTime)
        RsltDir_Present = RsltDir + '/' + PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
        if ElemQorH == 'H':
            AllP_hydroFile = RsltDir_Present + '/' + PresentTimeTxt + StationKey + PredictionPoint + Result_f_all + '_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        elif ElemQorH == 'Q':
            AllP_hydroFile = RsltDir_Present + '/' + PresentTimeTxt + StationKey + PredictionPoint + Result_f_all + '_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
            #202306011200_Tsukinowa_AllP_wl_Pn008_BT03h_FT06h.csv
        if BackTimeFlg == 0:
            All_hydro = np.loadtxt(AllP_hydroFile, dtype='float', delimiter=",", skiprows=1+nSkipBT)
        elif BackTimeFlg == 1:
            All_hydro = np.loadtxt(AllP_hydroFile, dtype='float', delimiter=",", skiprows=1)
        f_T = All_hydro.T
        ls_AllHydroDatetimeTxt = f_T[0]
        ls_AllHydroData = f_T[1:]
        for iStepAll in range(len(ls_AllHydroDatetimeTxt)):
            Do4AllDatetimeTxt = str(int(ls_AllHydroDatetimeTxt[iStepAll]))
            ls_AllHydroDatetime.append(datetime.datetime.strptime(Do4AllDatetimeTxt, '%Y%m%d%H%M'))
        for iPn in range(Pn):
            ls_iPnHydroData = []
            for iStepAll in range(len(ls_AllHydroDatetimeTxt)):
                ls_iPnHydroData.append(ls_AllHydroData[iPn][iStepAll])
            #ax1.plot(ls_AllHydroDatetime, ls_iPnHydroData, lw=1.0, alpha=0.5, linestyle="solid", color='orange
            ax1.plot(ls_AllHydroDatetime, ls_iPnHydroData,lw=1.0, alpha=0.5, linestyle="solid", color='orange',
             label='GFS All Particle W.L' if iPn == 0 else "")
            # ax1.plot(ls_AllHydroDatetime, ls_iPnHydroData, lw=0.5, alpha=0.5, linestyle="dotted", color='silver')
    #########Forecasted wl/qr for all particle##################################

    #########Forecasted wl/qr for BestP ##################################
    if ForecastHyd_BestP_Flg != 0:
        ls_Datetime_HydBestP,ls_Hydro4Fig = [],[]
        PresentTimeTxt = '{:%Y%m%d%H%M}'.format(PresentTime)
        RsltDir_Present = RsltDir + '/' + PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
        hydro_f_tmp = RsltDir_Present + '/' + DoTimeTxt + StationKey + PredictionPoint + Result_f_best + '_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        hydro_Best = glob.glob(hydro_f_tmp)
        hydro_Best_f = hydro_Best[0]
        # 202306011200_Tsukinowa_BestP002_wl_Pn008_BT03h_FT06h.csv
        if BackTimeFlg == 0:
            f = np.loadtxt(hydro_Best_f, dtype='float', delimiter=",", skiprows=1+nSkipBT)
        elif BackTimeFlg == 1:
            f = np.loadtxt(hydro_Best_f, dtype='float', delimiter=",", skiprows=1)
        f_T = f.T
        ls_DatetimeTxt_HydBestP = f_T[0]
        ls_Data_HydBestP = f_T[1]
        for iStep in range(len(ls_DatetimeTxt_HydBestP)):
            DoDatetimeTxt = str(int(ls_DatetimeTxt_HydBestP[iStep]))
            ls_Datetime_HydBestP.append(datetime.datetime.strptime(DoDatetimeTxt, '%Y%m%d%H%M'))
            ls_Hydro4Fig.append(ls_Data_HydBestP[iStep])
        if ForecastHyd_BestP_Flg == 2:
            NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(PresentTime.strftime('%Y%m%d%H%M'))]
            if BackTimeFlg == 1:
                NowCalcVal = ls_Hydro4Fig[nSkipBT]
            elif BackTimeFlg == 0:
                NowCalcVal = ls_Hydro4Fig[0]
            NowErrVal = NowObsVal - NowCalcVal
            ls_Hydro4Fig = ls_Hydro4Fig + NowErrVal
        ax1.plot(ls_Datetime_HydBestP, ls_Hydro4Fig,lw=1.5,alpha=1.0,label='GFS Best Particle W.L',linestyle="dotted",markersize=3,color='blue')
    #########Forecasted wl/qr for BestP ##################################

    #########Forecasted wl/qr for OrgP ##################################
    if ForecastHyd_OrgP_Flg != 0:
        ls_Datetime_HydOrgP,ls_Hydro4Fig = [],[]
        PresentTimeTxt = '{:%Y%m%d%H%M}'.format(PresentTime)
        RsltDir_Present = RsltDir + '/' + PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
        hydro_f_tmp = RsltDir_Present + '/' + DoTimeTxt + StationKey + PredictionPoint + Result_f_org + '_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        hydro_Org = glob.glob(hydro_f_tmp)
        hydro_Org_f = hydro_Org[0]
        # 202306011200_Tsukinowa_OrgP002_wl_Pn008_BT03h_FT06h.csv
        if BackTimeFlg == 0:
            f = np.loadtxt(hydro_Org_f, dtype='float', delimiter=",", skiprows=1+nSkipBT)
        elif BackTimeFlg == 1:
            f = np.loadtxt(hydro_Org_f, dtype='float', delimiter=",", skiprows=1)
        f_T = f.T
        ls_DatetimeTxt_HydOrgP = f_T[0]
        ls_Data_HydOrgP = f_T[1]
        for iStep in range(len(ls_DatetimeTxt_HydOrgP)):
            DoDatetimeTxt = str(int(ls_DatetimeTxt_HydOrgP[iStep]))
            ls_Datetime_HydOrgP.append(datetime.datetime.strptime(DoDatetimeTxt, '%Y%m%d%H%M'))
            ls_Hydro4Fig.append(ls_Data_HydOrgP[iStep])
        if ForecastHyd_OrgP_Flg == 2:
            NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(PresentTime.strftime('%Y%m%d%H%M'))]
            if BackTimeFlg == 1:
                NowCalcVal = ls_Hydro4Fig[nSkipBT]
            elif BackTimeFlg == 0:
                NowCalcVal = ls_Hydro4Fig[0]
            NowErrVal = NowObsVal - NowCalcVal
            ls_Hydro4Fig = ls_Hydro4Fig + NowErrVal
        ax1.plot(ls_Datetime_HydOrgP, ls_Hydro4Fig,lw=1.5,alpha=1.0,label='GFS Original Particle W.L',linestyle="dashed",markersize=3,color='Red')
    #########Forecasted wl/qr for OrgP ##################################

    #########Forecasted wl/qr for weight mean ##################################
    if ForecastHyd_WtMean_Flg != 0:
        ls_Datetime_HydWtMean,ls_Hydro4Fig = [],[]
        PresentTimeTxt = '{:%Y%m%d%H%M}'.format(PresentTime)
        RsltDir_Present = RsltDir + '/' + PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
        hydro_f = RsltDir_Present + '/' + PresentTimeTxt + StationKey + PredictionPoint + Result_f_mean + '_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        # 202306011200_Tsukinowa_WtMean_wl_Pn100_BT03h_FT06h.csv
        if BackTimeFlg == 0:
            f = np.loadtxt(hydro_f, dtype='float', delimiter=",", skiprows=1+nSkipBT)
        elif BackTimeFlg == 1:
            f = np.loadtxt(hydro_f, dtype='float', delimiter=",", skiprows=1)
        f_T = f.T
        ls_DatetimeTxt_HydWtMean = f_T[0]
        ls_Data_HydWtMean = f_T[1]
        for iStep in range(len(ls_DatetimeTxt_HydWtMean)):
            DoDatetimeTxt = str(int(ls_DatetimeTxt_HydWtMean[iStep]))
            ls_Datetime_HydWtMean.append(datetime.datetime.strptime(DoDatetimeTxt, '%Y%m%d%H%M'))
            ls_Hydro4Fig.append(ls_Data_HydWtMean[iStep])
        if ForecastHyd_WtMean_Flg == 2:
            NowObsVal = ls_ObsVal[ls_ObsDatetimeTxt.index(PresentTime.strftime('%Y%m%d%H%M'))]
            if BackTimeFlg == 1:
                NowCalcVal = ls_Hydro4Fig[nSkipBT]
            elif BackTimeFlg == 0:
                NowCalcVal = ls_Hydro4Fig[0]
            NowErrVal = NowObsVal - NowCalcVal
            ls_Hydro4Fig = ls_Hydro4Fig + NowErrVal
        ax1.plot(ls_Datetime_HydWtMean, ls_Hydro4Fig,lw=1.5,alpha=1.0,label='GFS Mean Particle W.L',linestyle="solid",markersize=5,color='Red')
    #########Forecasted wl/qr for weight mean ##################################

    #########Present line###################################
    ax1.plot([PresentTime,PresentTime], [-9999,9999], lw=3.0, alpha=1.0, linestyle="solid", color='black')
    #########Present line###################################

    ax1.legend(prop=fp)
    #ax2.legend()

    ##テキストボックスでタイトル表示
    TitleName = PresentTime
    plt.text(0.5, 1.02, TitleName, {'color':'red'}, horizontalalignment='center',
            verticalalignment='center', fontproperties=fp, clip_on=False, transform=plt.gca().transAxes)

    return plt

def tail2list(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    Tail_line = lines[-1]
    Tail_List = [float(x.strip()) for x in Tail_line.split(',')]
    return Tail_List
