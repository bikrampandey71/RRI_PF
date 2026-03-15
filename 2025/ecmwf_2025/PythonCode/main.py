# -*- coding: utf-8 -*-

import os
import sys
import shutil
import glob
import multiprocessing as multi
from multiprocessing import Pool
from collections import Counter
import numpy as np
import numpy.random as rd
import datetime
from distutils.dir_util import copy_tree
import pandas as pd
import copy
import BoundaryQH
import Drawing
import EnsembleCalc
import HQeq
import ParticleFilter
import ReadiniFile
import ReadRRI_input
import Sediment
import StateSpace
import SubFunc
import UniformFlow

if __name__ == '__main__':
    DatetimeFile = 'datetime.txt'                   # 現時刻をYYYYMMDDhhmm形式で格納されたファイル
    HyetoFile    = 'BasinAveRain.txt'                  # rainBasin_extraction.exeの出力ファイル名(Hyeto data)
    Resampling_f = 'Resampling.log'                 # 粒子番号の時系列情報 (t=0は事後分布)
    Likekihood_f = 'Likelihood.log'                # 尤度の時系列情報 (t=1～)
    Sigma_f      = 'Sigma.log'                          # 観測誤差の標準偏差
    CalcLog      = 'Calc.log'
    CrossSec_f   = 'CrossSec.csv'
    CorrectedWL  = 'CorrectedWL.txt'                 # 観測水位と計算水位の誤差; HQ式等の補正量
    RainExe      = 'RadarRain_dat2rri.exe'               # datファイルからRRI用のrain.txtに変換するprg
    RainIni      = 'RadarRain_dat2rri4PF.ini'            # datファイルからRRI用のrain.txtに変換するprg
    InitHs       = 'hs_init.out'                          # hs initial condition for hot start
    InitHr       = 'hr_init.out'                          # hr initial condition for hot start
    InitGA       = 'ga_init.out'                          # ga initial condition for hot start
    HQeq_f       = 'HQeq.ini'                           # HQ式のa, bを格納したファイル
    HQtab_f      = 'HQtab.csv'
    RivState_f   = 'RivState_init.txt'              # 等流水深計算用の条件ファイル（河床勾配と土砂堆積深）
    RainTxt      = 'rain.txt'
    RRI_InputTxt = 'RRI_Input.txt'
    BoundaryHr   = 'hr_wlev_bound.txt'            # boundary file name for the water level
    BoundaryQr   = 'qr_bound.txt'                 # boundary file name for the discharge
    SedimentPotential = 0.10                        # The potential of sediment [Unit: m/hr]
    BaseWL = 0.0
    DrawFlaFile  = '/PythonCode/Drawing.flg'         # Activation of the drawing figure

    # === PRESETTING =================================================================
    # [Step0:Read] === 1.Read PF config >>> 2.time management >>> 3.Read RRI parameters =========
    # 1. Read 'RRI-PFconfig.ini' ----------
    Config_PF_f = './../RRI-PFconfig.ini'
    ConvQ2H, HomeDir, PredictionPoint, Hydro_f_type, Hydro_f, Hydro_locNo, CalcDir, InitDir, RsltDir, ArchDir, \
    ObsData, HQData, SecData, RainDir, Rain_extraction_No, BoundHr_func, BoundHr_ini, BoundQr_func, BoundQr_ini,\
    ParticleNum, ResamplingMethod, LikelihoodMethod, nFixP, nStates, ls_StatesNo, ls_StatesOp, \
    ls_RRI_Param, FixedValue, OrgParticle, OrgValue, SequentialConditions_RRI, \
    BT_dy, FT_dy, ForecastType, RRI_dt_min, PF_dt_min, \
    PF_StartTime, PF_EndTime, Mean_SysNoise, SD_SysNoise, SigmaErr_Const, \
    Rslt_Best_all, Rslt_Org_all, Rslt_WtMean_hs, Rslt_WtMean_hr, Rslt_WtMean_ga, Rslt_WtMean_qr, Rslt_OtherSt \
    = ReadiniFile.ReadiniFile(Config_PF_f)
    # SelectConditions_hs, SelectConditions_hr, SelectConditions_ga <<< integrated as RRI on 2024.11.13 by YN
    SelectConditions_hs = SequentialConditions_RRI
    if (0 in ls_StatesNo) == True: SelectConditions_hs = 3
    SelectConditions_hr = SequentialConditions_RRI
    if (2 in ls_StatesNo) == True: SelectConditions_hr = 3
    SelectConditions_ga = SequentialConditions_RRI
    SelectConditions_para = SequentialConditions_RRI
    if (5 in ls_StatesNo) == True: SelectConditions_para = 0    #0: org particle
    # PF conditions
    Pn = ParticleNum
    DA_Obs_threshold = 1.0 # Mean_ObsV or Now_ObsV??
    Tn = int((BT_dy + FT_dy) * 1440 / RRI_dt_min)   # Numbers of step: back-time + forecast-time
    LikelihoodCondition = 1 # Fixed constant
    Sigma_ErrH = SigmaErr_Const
    # x-command [0] >>> Standard PF, [1] >>> like EnsKF
    EnsKF_hs = 0
    EnsKF_hr = 0
    EnsKF_ga = 0    # fix
    EnsKF_para = 0
    EnsKF_rain = 0
    EnsKF_bound = 0
    # (1) Original particle is [1]  Fixed from 2024.12.31 by YN
    OrgPn = OrgParticle
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print(("Original particle >>> [No. " + str(OrgPn) + "] / " + str(Pn)) + " particles")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # (2) Make & Check the directory construction
    msg = SubFunc.CheckDir2024(HomeDir, Pn)
    print(msg)
    # difine directory 
    TmpDir = CalcDir + '/tmp'
    SecCondIni = InitDir + '/' + RivState_f
    # log file of particle distribution accroding to the number of states 
    ParticleDistLog = ['ParticleDistribution1.log'] # list of states/parameters file in timeseries (1st row: μ[Mean_initDist, SD_initDist])
    if nStates > 1:
        for iStates in range(nStates-1):
            ParticleDistLog.append('ParticleDistribution' + str(iStates + 2) + '.log')

    # 2. Time management ----------
    Datetime_f = open(HomeDir + '/' + DatetimeFile, 'r')
    PresentTimeTxt = str(Datetime_f.readline())
    Datetime_f.close()
    if PresentTimeTxt != PF_StartTime:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print("Hot start func.= [OFF] <<< 'PresentTimeTxt' != 'PF_StartTime'\n")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    PresentTime = datetime.datetime.strptime(PresentTimeTxt, '%Y%m%d%H%M')
    # YYYY_Now = PresentTime.strftime("%Y"), MM_Now = PresentTime.strftime("%m"), DD_Now = PresentTime.strftime("%d")
    BackTime = PresentTime - datetime.timedelta(days = BT_dy)
    BackTimeTxt = '{:%Y%m%d%H%M}'.format(BackTime)
    ForecastTime = PresentTime + datetime.timedelta(days = FT_dy)
    ForecastTimeTxt = '{:%Y%m%d%H%M}'.format(ForecastTime)
    EndTime = datetime.datetime.strptime(str(PF_EndTime), '%Y%m%d%H%M')
    NextTime = PresentTime + datetime.timedelta(minutes = PF_dt_min)
    NextTimeTxt = '{:%Y%m%d%H%M}'.format(NextTime)
    PreviousTime = PresentTime - datetime.timedelta(minutes = PF_dt_min)
    PreviousTimeTxt = '{:%Y%m%d%H%M}'.format(PreviousTime)
    NextInitTime = PresentTime - datetime.timedelta(days = BT_dy) + datetime.timedelta(minutes = PF_dt_min)
    NextInitTimeTxt = '{:%Y%m%d%H%M}'.format(NextInitTime)
    print(("PresentTime  : " + str(PresentTime)))
    print(("BackTime     : " + str(BackTime)))
    print(("ForecastTime : " + str(ForecastTime)))
    print(("Sim. EndTime : " + str(EndTime)))
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    if PresentTime > EndTime:
        print('PF experiments was completed!!')
        print("  >>> Stop condition:",PresentTime,">",EndTime)
        sys.exit()
    # [Current time] Make 'Results' & 'Archives' in YYYY/MM/DD
    DateDir_now = PresentTime.strftime("%Y") + '/' + PresentTime.strftime("%m") + '/' + PresentTime.strftime("%d")
    RsltDir_now = RsltDir + '/' + DateDir_now
    os.makedirs(RsltDir_now, exist_ok=True)
    ArchDir_now = ArchDir + '/' + DateDir_now
    ArchDir_now_RRI = ArchDir_now + '/RRI'
    os.makedirs(ArchDir_now_RRI, exist_ok=True)
    ArchDir_now_Riv = ArchDir_now + '/River'
    os.makedirs(ArchDir_now_Riv, exist_ok=True)
    # [Next time] Make 'Results' & 'Archives' in YYYY/MM/DD
    DateDir_next = NextTime.strftime("%Y") + '/' + NextTime.strftime("%m") + '/' + NextTime.strftime("%d")
    ArchDir_next = ArchDir + '/' + DateDir_next
    ArchDir_next_RRI = ArchDir_next + '/RRI'
    os.makedirs(ArchDir_next_RRI, exist_ok=True)
    ArchDir_next_Riv = ArchDir_next + '/River'
    os.makedirs(ArchDir_next_Riv, exist_ok=True)
    # Archive 'RRI-PFconfig.ini' >>> Results
    iniFilePath = "./../RRI-PFconfig.ini"
    shutil.copyfile(iniFilePath, RsltDir_now + '/' + PresentTimeTxt + '_RRI-PFconfig.ini')
    # [Previous time]
    DateDir_prev = PreviousTime.strftime("%Y") + '/' + PreviousTime.strftime("%m") + '/' + PreviousTime.strftime("%d")
    RsltDir_prev = RsltDir + '/' + DateDir_prev

    # 3. Get Obs. data
    df_ObsV = pd.read_csv(ObsData, header = 0, dtype={'Datetime' : 'object', 'Obs' : 'float'})
    Ln = len(df_ObsV.index)
    df_ObsV_sort = SubFunc.df_SortedObs(df_ObsV, PresentTimeTxt, BackTimeTxt)
    Mean_ObsV = df_ObsV_sort['Obs'].mean()
    ls_Obs = df_ObsV_sort['Datetime'].tolist()
    if (PresentTimeTxt in ls_Obs) == True:
        Now_idx = ls_Obs.index(PresentTimeTxt)
        Now_ObsV = df_ObsV_sort['Obs'].iloc[Now_idx]
    elif (PresentTimeTxt in ls_Obs) == False:
        Now_ObsV = Mean_ObsV
    nObs = len(df_ObsV_sort)
    # Obs. datetime
    obs_t = np.zeros([nObs])
    obs_t = df_ObsV_sort['Datetime'].values
    # Obs. value
    obs_v = np.zeros([nObs])
    obs_v = df_ObsV_sort['Obs'].values
    # ******************
    ObsV4DA = "now"
    # ******************
    if ObsV4DA == "mean":
        ObsV4DA = Mean_ObsV
    elif ObsV4DA == "now":
        ObsV4DA = Now_ObsV
    if ObsV4DA >= DA_Obs_threshold:
        DA_flag = 1
    else:
        DA_flag = 0

    # 4. Read settings of RRI model & River model ----------
    # (1) read 'RRI_Input.txt' ----------
    Init_rri_input = InitDir + '/' + RRI_InputTxt
    if os.path.exists(Init_rri_input) == False:
        print("'RRI_Input.txt' file does NOT exist in 'InitialConditions' folder [Required]!!")
        sys.exit()
    with open(Init_rri_input) as f:
        org = f.readlines()
    L3Path_rain, L4Path_dem, L5Path_acc, L6Path_dir, L14rain_xll, L15rain_yll, \
    L18ns_riv, L19Num_LU, L38Val_RivThresh, L39Val_Cw, L40Val_Sw, L41Val_Cd, L42Val_Sd, \
    L43Val_HeightPara, L44Val_Height, L46Flg_RivFile, L47Path_Width, L48Path_Depth, L49Path_Height, \
    L51hs_init_flg, L51hr_init_flg, L51hg_init_flg, L51ga_init_flg, L66Path_LU, L100Path_Loc = ReadRRI_input.Read_RRI_input(org)
    # (2) Check 'location.txt' ----------
    Location_f = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/' + L100Path_Loc 
    if os.path.exists(Location_f) == False:
        print ("'location.txt' file does NOT exist!!")
        sys.exit()
    Loc_Name, Loc_i, Loc_j, nOut = ReadRRI_input.Read_location(Location_f)
    # (3) Read indexes in 'dem.txt' ----------
    DemFile = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/' + L4Path_dem
    ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value = ReadRRI_input.Read_index(DemFile)
    # (4) Remove result (*.out) ----------
    print("Removing results of the RRI Model...")
    msg = SubFunc.RemoveOutfile(CalcDir, TmpDir, Pn)
    print(msg)

    # === PARTICLE FILTER CYCLE =========================================================================
    # [Step1:Preparation] ============
    # 1. Read previous results ----------
    # (1) Read "Prior (initial) distribution" of states/parameters
    LogFileName = []
    for iStates in range(nStates):
        LogFileName.append(RsltDir_prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + ParticleDistLog[iStates])
    StatusList = []
    for iStates in range(nStates):
        if os.path.exists(LogFileName[iStates]) == True: # Case of existing 'ParticleDistribution#.log'
            ls_TailLine = SubFunc.tail2list(LogFileName[iStates])
            StatusList.append(ls_TailLine)
        elif os.path.exists(LogFileName[iStates]) == False:
            print('!!!Caution!!! NOT found the "Prior distribution" >>> file: ParticleDistribution#.log')
            sys.exit()
    # StatusList_Initial: This states distribution is not including updated by "Option func." (StatusList_Prior is including Opt.)
    StatusList_Initial = copy.deepcopy(StatusList)
    for iStates in range(nStates):
        print("StatusList_Initial #" + str(iStates+1) + ":\n",np.round(StatusList_Initial[iStates], 3))
    # (2) Make "Prior distribution" of Particle Number
    Resampled_f = RsltDir_prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + Resampling_f
    k0 = list(range(Pn)) # k0 = [0, 1, 2, ... , Pn-1] (serial number)
    Resampling_Prior = [num + 1 for num in k0]
    print("Resampling_Prior:\n",Resampling_Prior) # Resampling_Prior = [1, 2, 3, ... , Pn] (serial number)
    # (3) Read "Prior distribution" of Particle Number
    LikelihoodFile = RsltDir_prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + Likekihood_f
    Likelihood_Prior = SubFunc.tail2list(LikelihoodFile)
    print("Likelihood_Prior:\n",np.round(Likelihood_Prior, 3))
    # (4) Read "Prior distribution" of Particle Number
    SigmaLogFile = RsltDir_prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + Sigma_f
    SigmaErr_Prior = SubFunc.tail2list(SigmaLogFile)

    # 3. Initial condition for the [RRI model]: change the initial files (hs/hr/ga files) into 'Particle/RRI/init' folder
    # RRI-hs conditions (move file & change the state-space)
    print(' RRI [hs] preparing ... ----------------------------------------')
    if SelectConditions_hs != 3:    #Case: 0(original), 1(best), 2(weight mean), 3(each)
        if SelectConditions_hs == 0:
            hs_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + InitHs
        elif SelectConditions_hs == 1:
            hs_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + InitHs
        elif SelectConditions_hs == 2:
            hs_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + InitHs
        for iPn in range(Pn):
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
            hs_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/init/' + InitHs
            shutil.copy2(hs_rslt_f, hs_seq_p_f)
            if (1 in ls_StatesNo) == True:
                iLs = ls_StatesNo.index(1)
                StatusVal = StatusList[iLs][iPn]
                if EnsKF_hs == 1: StatusVal = rd.normal(1.0, 0.4)
                StatusVal = StateSpace.Change_StateSpace_hs_rate(1, StatusVal, hs_seq_p_f, ls_StatesOp[iLs])
                if ls_StatesOp[iLs] == 1 or EnsKF_hs == 1: StatusList[iLs][iPn] = StatusVal
    elif SelectConditions_hs == 3:
        for iPn in range(Pn):
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
            hs_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + InitHs
            hs_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/init/' + InitHs
            shutil.copy2(hs_rslt_f, hs_seq_p_f)
    print('                               RRI [hs] >>> done!')
    # RRI-hr conditions (move file & change the state-space)
    print(' RRI [hr] preparing ... ----------------------------------------')
    if SelectConditions_hr != 3:    #Case: 1(best), 2(weight mean), 4(original)
        if SelectConditions_hr == 0:
            hr_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + InitHr
        elif SelectConditions_hr == 1:
            hr_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + InitHr
        elif SelectConditions_hr == 2:
            hr_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + InitHr
        for iPn in range(Pn):
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
            hr_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/init/' + InitHr
            shutil.copy2(hr_rslt_f, hr_seq_p_f)
            if (3 in ls_StatesNo) == True:
                iLs = ls_StatesNo.index(3)
                StatusVal = StatusList[iLs][iPn]
                if EnsKF_hr == 1: StatusVal = rd.normal(1.0, 0.4)
                StatusVal = StateSpace.Change_StateSpace_hr_rate(3, StatusVal, hr_seq_p_f, ls_StatesOp[iLs])
                if ls_StatesOp[iLs] == 1 or EnsKF_hr == 1: StatusList[iLs][iPn] = StatusVal
    elif SelectConditions_hr == 3:
        for iPn in range(Pn):
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
            hr_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + InitHr
            hr_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/init/' + InitHr
            shutil.copy2(hr_rslt_f, hr_seq_p_f)
    print('                               RRI [hr] >>> done!')
    # RRI-ga conditions (move file only)
    print(' RRI [ga] preparing ... ----------------------------------------')
    if L51ga_init_flg == 1:
        if SelectConditions_ga != 3:    #Case: 1(best), 2(weight mean), 4(original)
            if SelectConditions_ga == 0:
                ga_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + InitGA
            elif SelectConditions_ga == 1:
                ga_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + InitGA
            elif SelectConditions_ga == 2:
                ga_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + InitGA
            for iPn in range(Pn):
                print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
                ga_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/init/' + InitGA
                shutil.copy2(ga_rslt_f, ga_seq_p_f)
        elif SelectConditions_ga == 3:
            for iPn in range(Pn):
                print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
                ga_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + InitGA
                ga_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/init/' + InitGA
                shutil.copy2(ga_rslt_f, ga_seq_p_f)
        print('                               RRI [ga] >>> done!')
    # RRI_Input conditions (move file & change the state-space)
    print(' RRI [RRI_Input] preparing ... ---------------------------------')
    para_init_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_' + RRI_InputTxt
    for iPn in range(Pn):
        print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        para_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/' + RRI_InputTxt
        shutil.copy2(para_init_f, para_seq_p_f)
        if (5 in ls_StatesNo) == True:
            for iState in range(len(ls_StatesNo)):
                if ls_StatesNo[iState] == 5:
                    # Change state space depend on status list
                    StatusVal = StatusList[iState][iPn]
                    if EnsKF_para == 1: StatusVal = rd.normal(1.0, 0.4)
                    StateSpace.Change_RRI_Input(ls_StatesNo[iState], StatusVal, ls_RRI_Param[iState], para_seq_p_f)
                    if ls_StatesOp[iState] == 1 or EnsKF_para == 1: StatusList[iState][iPn] = StatusVal
        elif (61 in ls_StatesNo) == True or (62 in ls_StatesNo) == True:
            rainBasin_extraction_p = CalcDir+'/Particle'+str(iPn+1).zfill(5)+'/RRI/etc/rainBasin_extraction/rainBasin_extraction.txt'
            shutil.copy2(InitDir + '/rainBasin_extraction.txt', rainBasin_extraction_p)
            for iState in range(len(ls_StatesNo)):
                if ls_StatesNo[iState] == 61:
                    # Change rain position xllconer
                    rain_row = 14
                    rain_lon = L14rain_xll
                    StatusVal = StatusList[iState][iPn] + rain_lon
                    StateSpace.Change_RRI_Input(ls_StatesNo[iState], StatusVal, rain_row, para_seq_p_f)
                    #'C:/RRI/RRI-PF/Minami_5s_No00_20230602Flood/Particles/Particle00002/RRI/etc/rainBasin_extraction.txt'
                    SubFunc.rainBasin_extraction_position(StatusVal, 3, rainBasin_extraction_p)
                if ls_StatesNo[iState] == 62:
                    # Change rain position yllconer
                    rain_row = 15
                    rain_lat = L15rain_yll
                    StatusVal = StatusList[iState][iPn] + rain_lat
                    StateSpace.Change_RRI_Input(ls_StatesNo[iState], StatusVal, rain_row, para_seq_p_f)
                    SubFunc.rainBasin_extraction_position(StatusVal, 4, rainBasin_extraction_p)
        para_rslt_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + RRI_InputTxt
        shutil.copy2(para_seq_p_f, para_rslt_f)
    print('                               RRI [RRI_Input] >>> done!')

    # 4. Boundary condition for the [RRI model]: boundary rain/discharge/water depth(hr)
    # (1) [RRI]rain-ratio conditions
    print("=== [Boundary condition] make rain.txt in '(home)/ObsData/Rainfall' ==========")
    
    """
    # Write over of "RadarRain_dat2rri4PF.ini" ---------------------------
    CombinedFastAR_Minutes = 0  # RRI_Interval, PF_Interval, CombinedFastAR_Minutes	[Unit:min]
    DelayOption = 0             # [1]-> Active, [0] or blank -> Inactive
    FastMinFlag = 0     # [1]-> FastR is used for a minimum times considering delay time (10 - 30 mins)
                        #       If "FastMinFlag" is set by 1, settings of "CombinedFastAR_Minutes" is ignored.
                        # [0] or blank -> Inactive
    RainList = ((PresentTimeTxt, BT_dy, FT_dy, ForecastType, CombinedFastAR_Minutes, DelayOption, FastMinFlag))  # Add, 2020/12/02, For FastAR in PRISM2020
    Rain_ini = open(HomeDir + '/' + 'ObsData/Rainfall' + '/' + RainIni, 'w')
    # 1) Write a ini-file
    for x in RainList:
        Rain_ini.write(str(x) + "\n")
    Rain_ini.close()
    # 2) Make a rain.txt
    os.chdir(RainDir)
    ls_path = glob.glob(RainDir + '/rain_*.txt')
    if len(ls_path) > 0:
        for OldRain in ls_path:
            os.remove(OldRain)  # Erase of old rain files
    result_r = os.system(RainExe)
    """
    os.chdir(RainDir)
    #ls_path = glob.glob(RainDir + '/rain_*.txt')
    ls_path = glob.glob(RainDir + '/rain_' + PresentTimeTxt + '.txt')
    # 3) Archive in 'Result' <<< 'rain_YYYYMMDDhhmmBTxxFTxx.txt'
    Rain_Org_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_' + RainTxt
    shutil.copy2(ls_path[0], Rain_Org_f)
    # 4) Deriver each particle
    for iPn in range(Pn):
        print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        Rain_seq_f = CalcDir + '/' + 'Particle' + str(iPn + 1).zfill(5) + '/RRI/rain/' + RainTxt
        shutil.copy2(ls_path[0], Rain_seq_f)
        # [option] When the state-space must be updated (e.g., rain.txt)
        if (6 in ls_StatesNo) == True:
            # (a) Read the Prior Distribution
            iLs = ls_StatesNo.index(6)
            StatusVal = StatusList[iLs][iPn]
            if EnsKF_rain == 1: StatusVal = rd.normal(1.0, 0.4) # [Exception] accroding the Normal Distribution like as EnsKF
            # (b) Change rain
            StatusVal = StateSpace.Change_RainTxt(6, StatusVal, Rain_seq_f, BT_dy, ls_StatesOp[iLs])
            # Archive folder
            Rain_Arch_f = ArchDir_now_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + RainTxt
            shutil.copy2(Rain_seq_f, Rain_Arch_f)
            if ls_StatesOp[iLs] == 1 or EnsKF_rain == 1: StatusList[iLs][iPn] = StatusVal
    print('                               [rain.txt] >>> done!')
    # (2) RRI-qr conditions (e.g., dam release discharge, Cascade DA from upper)
    if BoundQr_func != 0:   # Case of the boundary conditon using the observed Q and calculated Q.
        print ("+++++++++++++++++++++++++++++++ \nBoundary DISCHARGE >>> ON")
        if os.path.exists(BoundQr_ini) == False:
            print ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n   ""*.ini"" file does NOT exist!! \n    >>>Stop PF! (Comfirm the ini file)  \n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            sys.exit()
        StatusList =BoundaryQH.RRI_BoundQr(HomeDir, DateDir_now, ArchDir_now_RRI, CalcDir, BoundQr_ini, BoundaryQr, \
                    PresentTimeTxt, BackTime, ForecastTime, BT_dy, FT_dy, \
                    Pn, StatusList, ls_StatesNo, ls_StatesOp, EnsKF_bound)
        print('                               RRI [bound_qr] >>> done!')

    # (3) RRI-hr conditions (e.g. Tidal water level)
    if BoundHr_func != 0:   # Case of the boundary conditon using the observed water level and calculated one.
        print ("+++++++++++++++++++++++++++++++ \n  Boundary WATER LEVEL >>> ON \n+++++++++++++++++++++++++++++++")
        if os.path.exists(BoundHr_ini) == False: # CSVがあるのに"BoundaryHr_condition.ini"が存在しない場合
            print ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n   ""*.ini"" file does NOT exist!! \n    >>>Stop PF! (Comfirm the ini file)  \n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            sys.exit()
        # --- (1) header information in 'hr_wlev_bound.txt' ---
        nBound_Hr, BoundFlagH, BoundHrFile, ZeroTPm, loc_i_Hr, loc_j_Hr = ReadiniFile.importBoundaryHCondition(BoundHr_ini)
        # --- (2) read boundary location by 'dem.txt' in the particle00001. ---
        #DemFile = CalcDir + '/Particle00001' + L4Path_dem ***already read to get the index in 'dem.txt'
        #print DemFile
        if nBound_Hr > 0:
            dem_Hr = np.zeros(nBound_Hr)
            depth_Hr = np.zeros(nBound_Hr)
            acc_Hr = np.zeros(nBound_Hr)
            for iBound in range(nBound_Hr):
                # Read dem value
                dem_Hr[iBound] = float(ReadRRI_input.Read_map(DemFile, loc_i_Hr[iBound], loc_j_Hr[iBound]))
                # Read / Calculation depth
                if L46Flg_RivFile == 1:
                    DepthFile = CalcDir + '/Particle00001' + L48Path_Depth
                    depth_Hr[iBound] = float(ReadRRI_input.Read_map(DepthFile, loc_i_Hr[iBound], loc_j_Hr[iBound]))
                elif L46Flg_RivFile == 0:
                    AccFile = CalcDir + '/Particle00001' + L5Path_acc
                    acc_Hr[iBound] = int(ReadRRI_input.Read_map(AccFile, loc_i_Hr[iBound], loc_j_Hr[iBound]))
                    depth_Hr[iBound] = L41Val_Cd * (acc_Hr[iBound] * 28**2 * (cellsize*3600)**2 / 1000 / 1000) ** L42Val_Sd
                print((depth_Hr[iBound]))

    # 5. Initial condition for [River model]: move the sequential files (HQeq/HQtable) into the 'Particle/River' folder
    # [HQeq.ini, HQtab.csv]
    print (" River model preparing ... ------------------------------------------------")
    if ConvQ2H == 1:
        print("  [Method] >>> HQ equation")
        # (1) Read a HQ eq. file
        HQ_prev_f = ArchDir_now_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + HQeq_f
        nHQ, HQ_a, HQ_b, HQ_Hmax, HQ_Qmax = HQeq.importHQeq(HQ_prev_f)
        # (2) Read a corrected wl value
        CorrectedWL_f = ArchDir_now_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + CorrectedWL
        CorrectB = SubFunc.tail2list(CorrectedWL_f)
        # (3) Correct HQeq about height
        New_nHQ = nHQ
        New_HQ_a = HQ_a
        New_HQ_b = HQ_b + CorrectB
        New_HQ_Hmax = HQ_Hmax + CorrectB
        New_HQ_Qmax = New_HQ_a * (New_HQ_Hmax + New_HQ_b) ** 2
        #print nHQ, HQ_a, HQ_b, HQ_Hmax, HQ_Qmax
        # (4) Archive of HQeq each particle folder (into 'ParticleXXXXX/River')
        for iPn in range(Pn):
            print("  [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particles;")
            HQeq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + HQeq_f
            with open(HQeq_p_f, 'w') as f:
                f.write("[HQ_num]" + "\n")
                f.write("nHQ = " + str(New_nHQ) + "\n")
                f.write("[HQ_para]" + "\n")
                for iHQ in range(New_nHQ):
                    f.write("a" + str(iHQ+1) + " = " + str(New_HQ_a[iHQ]) + "\n") 
                    f.write("b" + str(iHQ+1) + " = " + str(New_HQ_b[iHQ]) + "\n") 
                    f.write("Hmax" + str(iHQ+1) + " = " + str(New_HQ_Hmax[iHQ]) + "\n")
        print('                               [HQeq.ini] >>> done!')
    if ConvQ2H == 4:    # added on 2024.10.23
        print("  [Method] >>> HQ table")
        # (1) Read a HQ table file
        HQ_prev_f = ArchDir_now_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + HQtab_f
        HQ_table = np.loadtxt(HQ_prev_f, delimiter=",", skiprows=0)
        # (2) Archive of HQeq each particle folder (into 'ParticleXXXXX/River')
        for iPn in range(Pn):
            print("  [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particles;")
            HQtab_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + HQtab_f
            with open(HQtab_p_f, 'w') as f:
                np.savetxt(f, HQ_table, delimiter = ",", fmt = "%.5f")
        print('                               [HQtab.csv] >>> done!')
    if ConvQ2H == 2 or ConvQ2H == 21 or ConvQ2H == 31:
        print("  [Method] >>> Uniform flow")
        if ((ConvQ2H == 21 or ConvQ2H == 31) and (7 in ls_StatesNo) == False):
            print ("Settings error!!! \n You should set that the states are estimated sediment depthes (set as StatesNo=7).")
            sys.exit()
        for iPn in range(Pn):
            print("  [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particles;")
            # (1) Read a Cross section file
            Sec_Prev_f = ArchDir_now_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + CrossSec_f
            SecXY, ZeroElev = UniformFlow.Read_CrossSection(Sec_Prev_f)
            RivState_Prev_f = ArchDir_now_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + RivState_f
            RivGrad, SedDepth = UniformFlow.Read_RivConditions(RivState_Prev_f)
            # (2) Make H-Q table using previous conditions
            if ConvQ2H == 2:
                SedDepth = 0
            elif ConvQ2H == 21:
                iLs = ls_StatesNo.index(7)
                SedDepth = StatusList[iLs][iPn]
            elif ConvQ2H == 31:
                Mean_initDist = 0
                SD_initDist = SedimentPotential / (60 / PF_dt_min)
                SedDepth = rd.normal(Mean_initDist, SD_initDist)
            HQ_table, NewSecXY = UniformFlow.Sec2HQ_table(SecXY, ZeroElev, RivGrad, SedDepth)
            HQ_table = UniformFlow.ReviseHQ_table(HQ_table)
            print('   *Gradient = 1/ ' + str(int(1/RivGrad)))
            print('   *Sediment depth = ',format(SedDepth,'.3f'), '[m]')
            # (3) Archive of HQ table/CrossSec/RivState each particle folder (into 'ParticleXXXXX/River')
            # [H-Q table]
            HQtab_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + HQtab_f
            with open(HQtab_p_f, 'w') as f:
                np.savetxt(f, HQ_table, delimiter = ",", fmt = "%.5f")
            # [Cross section]
            Sec_p_f = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/River/' + CrossSec_f
            with open(Sec_p_f, 'w') as f:
                f.write("X,Y,Zero/N\n")
                for elem in NewSecXY:
                    f.write(str(elem[0]) + ',' + str(elem[1]-ZeroElev) + ',' + str(elem[2]) + "\n")
            # [River condition, L1: river gradient, L2:sediment depth]
            RivState_p_f = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/River/' + RivState_f
            with open(RivState_p_f, 'w') as f:
                f.write(str(int(1/RivGrad)) + "\n")
                f.write(str(SedDepth) + "\n")
        print('                               [HQtab.csv] >>> done!')

    # StatusList_Initial: This states distribution is not including updated by "Option func." (StatusList_Prior is including Opt.)
    for iStates in range(nStates):
        print("StatusList_Initial-" + str(iStates+1) + ":\n",np.round(StatusList_Initial[iStates], 3))
    print("             ↓↓↓")
    # StatusList_Prior: This states distribution is including updated by "Option func." (StatusList_Initial is not including Opt.)
    StatusList_Prior = copy.deepcopy(StatusList)
    for iStates in range(nStates):
        print("StatusList_Prior-" + str(iStates+1) + ":\n",np.round(StatusList_Prior[iStates], 3))


    # === PREDICTION =========================================================================
    # 1. Run RRI model by ensemble ---
    print("\n\n Next step: PREDICTION \n           ↓ ↓ ↓\n   *** Preparing multi-thread ... ***")
    EnsList = [(iPn, CalcDir, Hydro_f_type) for iPn in range(Pn)] # Preparation for ensemble calc.
    #print(multi.cpu_count())
    output = EnsembleCalc.multi_process(EnsList)    # Execute parallel calc. by each threads
    print("\n\n=== [PREDICTION] ==============================================")
    print("\n   [" + str(Pn) + "-member] Ensemble Calculations completed!!\n")
    print("=================================== RRI & River calculation ===\n")

    # 2. Convert ensemble discharge into ensemble water-level ---
    # (1) summary of the hydro data (qr/hr) for all particles
    print(" - Summarizing [ qr ] files .....")
    # (a) hydro data
    Name_p, hydro_sum = [], []
    for iPn in range(Pn):
        if Hydro_f_type == 1:
            hydro_path = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/' + Hydro_f
            loc_col = Hydro_locNo
        elif Hydro_f_type == 2:
            hydro_path = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/etc/calcHydro/' + Hydro_f
            loc_col = 1
        hydro_p = []
        with open(hydro_path, 'r') as f:
            for ln in f:
                val = ln.split()
                hydro_p.append(val[loc_col])
        Name_p.append('P' + str(iPn+1).zfill(5))
        hydro_sum.append(hydro_p)
    qr_sum = np.array(hydro_sum, dtype='float').T
    # (b) date time
    datetime_col, ls_datetime, ls_datetimeTxt = [], [], []
    if Hydro_f_type == 1:
        with open(CalcDir + '/Particle00001/RRI/' + Hydro_f, 'r') as f:
            for ln in f:
                val = ln.split()
                datetime_col.append(float(val[0]))
        for iTn in range(len(datetime_col)):
            tmp_t = BackTime + datetime.timedelta(seconds=datetime_col[iTn])    #hydro.txt >>> second
            ls_datetime.append(tmp_t)
            ls_datetimeTxt.append(tmp_t.strftime('%Y%m%d%H%M'))
    elif Hydro_f_type == 2:
        with open(CalcDir + '/Particle00001/RRI/etc/calcHydro/' + Hydro_f, 'r') as f:
            for ln in f:
                val = ln.split()
                datetime_col.append(float(val[0]))
        for iTn in range(len(datetime_col)):
            tmp_t = BackTime + datetime.timedelta(seconds=datetime_col[iTn]*RRI_dt_min*60) #hydro_qr/hr_*.txt >>> times
            ls_datetime.append(tmp_t)
            ls_datetimeTxt.append(tmp_t.strftime('%Y%m%d%H%M'))
    TimeSeries = np.array(ls_datetimeTxt)
    df_TimeSeries = pd.DataFrame(ls_datetimeTxt, columns = ['Datetime'])
    print('              >>> done!\n\n')

    # (2) summary of the hyeto data for all particles in DataFrame  *df_r_sum <- etc/rain_hyeto.txt!!
    print(" - Summarizing [rain] files ..........")
    ls_rain_sum, ls_datetime_r, ls_datetimeTxt_r = [], [], []
    # original particle hyeto
    rain_path = CalcDir + '/Particle00001/RRI/etc/rainBasin_extraction/' + HyetoFile
    rain_data = np.loadtxt(rain_path, skiprows=4)
    for iTn in range(len(rain_data)):
        # ls_rain_org.append(rain_data[iTn][Rain_extraction_No])
        tmp_r_t = BackTime + datetime.timedelta(seconds=rain_data[iTn][0])
        tmp_r_tTxt = tmp_r_t.strftime('%Y%m%d%H%M')
        ls_datetime_r.append(tmp_r_t)
        ls_datetimeTxt_r.append(tmp_r_tTxt)
    df_datetimeTxt_r = pd.DataFrame(ls_datetimeTxt_r, columns = ['Datetime'])
    # all particles hyeto
    for iPn in range(Pn):
        # case of 'rainBasin_extraction' (MCC's tool)
        ls_rain_p = []
        rain_path = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/etc/rainBasin_extraction/' + HyetoFile
        rain_data = np.loadtxt(rain_path, skiprows=4)
        for iTn in range(len(rain_data)):
            ls_rain_p.append(rain_data[iTn][Rain_extraction_No])
        ls_rain_sum.append(ls_rain_p)
    rain_sum = np.array(ls_rain_sum).T
    df_rain_d_sum = pd.DataFrame(rain_sum, columns = Name_p)
    df_rain_sum = pd.concat([df_datetimeTxt_r, df_rain_d_sum], axis=1)
    print('              >>> done!\n\n')

    # (3) Convert discharge into WL by the H-Q equations for all particles ---
    # if you want to evaluate likelihood in discharge, you should set 'ConvQ2H = 0'.
    calc_sum = np.zeros([Tn,Pn])
    Sedi_sum = np.zeros([Tn,Pn])
    SediFTorg_sum = np.zeros([Tn,Pn])
    if ConvQ2H == 1:    # Using H-Q equations
        # Case of using common H-Q equation
        for iPn in range(Pn):
            ElementQR = copy.deepcopy(qr_sum[:,iPn])
            HQeq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + HQeq_f
            nHQ, HQ_a, HQ_b, HQ_Hmax, HQ_Qmax = HQeq.importHQeq(HQData)     # HQData <<< 'HQeq.ini'
            Elem_qr2hr = HQeq.ConvQ2H_1x1_HQeq(ElementQR, nHQ, HQ_a, HQ_b, HQ_Qmax)
            calc_sum[:, iPn] = Elem_qr2hr
    elif ConvQ2H == 4:  # Using H-Q table
        calc_sum = UniformFlow.ConvQ2H_all_HQtab(qr_sum, HQ_table)
    elif ConvQ2H == 2:  # Using Uniform flow model (H-Q table <<< Cross sec. + gradient + param) 
        calc_sum = UniformFlow.ConvQ2H_all_HQtab(qr_sum, HQ_table)
    elif ConvQ2H == 21 or ConvQ2H == 31:      # <--- df_calc_sumにConvert to Water Level from discharge by HQ relation using the uniform flowを格納
        Flg_SediFT = 0      # 1: active sediment prediction, 0: inactive
        if Flg_SediFT == 0:
            print ("**************************** \n  Riverbed only current at the back time...  \n****************************")
            for iPn in range(Pn):
                ElementQR = copy.deepcopy(qr_sum[:,iPn])
                HQtab_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + HQtab_f
                HQ_table = np.loadtxt(HQtab_p_f, delimiter=",", skiprows=0)
                conv_qr2hr = UniformFlow.ConvQ2H_1x1_HQtab(ElementQR, HQ_table)
                calc_sum[:, iPn] = conv_qr2hr
        elif Flg_SediFT == 1:
            print ("**************************** \n  Considering riverbed prediction!!  \n****************************")
            Qc, Qi = Sediment.CalcQcQi(HomeDir)
            SediDepth_Sum = []
            for iPn in range(Pn):
                ElementQR = copy.deepcopy(qr_sum[:,iPn])          # 'copy.deepcopy' func is for list
                ElementQR_4Sedi = copy.deepcopy(qr_sum[:,iPn])    # 'copy.deepcopy' func is for list
                # need to check! --------------------------------------------------
                NextQ = ElementQR[int(PF_dt_min/RRI_dt_min)-1]
                if NextQ < Qc:
                    SedimentFlag = 0
                elif NextQ >= Qi and SedimentFlag != 1:
                    SedimentFlag = 1
                # need to check! --------------------------------------------------
                SediDeptd_FT = Sediment.Q2RivDepth_Prediction(HomeDir, ElementQR_4Sedi, Qc, Qi, SedimentFlag) - ZeroElev
                iLs = ls_StatesNo.index(7)
                SediDepth_BT = StatusList[iLs][iPn]
                SediDepth_BTFT = SediDeptd_FT - (SediDeptd_FT[0] - SediDepth_BT)
                conv_qr2hr = UniformFlow.CalcQ2H_RiverbedEvo_UniformFlow(ElementQR, SecXY, ZeroElev, RivGrad, SediDepth_BTFT)
                calc_sum[:, iPn] = conv_qr2hr
                Sedi_sum[:, iPn] = SediDepth_BTFT
                SediFTorg_sum[:,iPn] = SediDeptd_FT
            # Summary of results for sediment
            df_sedi_sum = pd.DataFrame(Sedi_sum)
            df_SediTime_sum = pd.concat([df_TimeSeries, df_sedi_sum], axis=1)
            df_sediFTorg_sum = pd.DataFrame(SediFTorg_sum)
            df_SediFTorgTime_sum = pd.concat([df_TimeSeries, df_sediFTorg_sum], axis=1)
            #SediDepth_Sum.append(SediDepth_BTFT)
    elif ConvQ2H == 0:  # discharge (e.g. dam)
        calc_sum = copy.deepcopy(qr_sum)

    # (4) Add time series column, and Array to DataFrame
    # At this point, "discharge" or "converted water level" is stored in "df_calcTime_sum" 
    df_calcQ_sum = pd.DataFrame(qr_sum, columns = Name_p)
    df_calcQTime_sum = pd.concat([df_TimeSeries, df_calcQ_sum], axis=1)
    if ConvQ2H != 0:  # not discharge
        df_calc_sum = pd.DataFrame(calc_sum, columns = Name_p)
        df_calcTime_sum = pd.concat([df_TimeSeries, df_calc_sum], axis=1)


    # === LIKELIHOOD & RESAMPLING =========================================================================
    # 1. Combined Obs. with Calc. ---
    # [NOTE] Obs. data (obs_t, obs_v) already get early process after "2. Time management".
    if ConvQ2H != 0:  # not discharge
        df_calc_sort = pd.merge(df_ObsV_sort, df_calcTime_sum, on='Datetime')
    else:
        df_calc_sort = pd.merge(df_ObsV_sort, df_calcQTime_sum, on='Datetime')
    calc_v = np.zeros([nObs, Pn])
    calc_v = df_calc_sort.iloc[:, 2:].values

    # 2. Evaluation of likelihoood and Calculation of weight ---
    StatusList_Resampled = []
    Likelihood = np.zeros(Pn)
    pf = ParticleFilter.ParticleFilter(obs_v, obs_t, calc_v, Pn, Sigma_ErrH, ResamplingMethod, LikelihoodMethod, nFixP, OrgPn, DA_flag)
    k, Likelihood, BestPn = pf.simulate()
    Likelihood_Posterior = Likelihood.tolist()
    Resampling_Posterior = [No + 1 for No in k]
    print("Likelihood_Prior    :\n",np.round(Likelihood_Prior, 3))
    print("            ↓↓↓")
    print("Likelihood_Posterior:\n",np.round(Likelihood_Posterior, 3),"\n")
    print("Resampling_Posterior:\n",Resampling_Posterior)
    print('               └≫ Best particle: #' + str(BestPn),"\n")
    # k: index of particle (Not P. No#), BestPn: best particle number, Likelihood: Weights of P. (=normalized likelihood)

    # States/parameters resampling depend on the weights
    for iStates in range(nStates):
        StatusList_tmp = []
        for m in range(Pn - nFixP):  #0,1,2,...,51(=100-48-1)
            if m != OrgPn-1:
                StatusList_tmp.append(StatusList[iStates][k[m]])
            elif m == OrgPn-1:
                StatusList_tmp.append(OrgValue[iStates])
        n = 0
        for m in range(Pn - nFixP, Pn):
            StatusList_tmp.append(FixedValue[iStates][n])
            n += 1
        StatusList_Resampled.append(StatusList_tmp)
        print("StatusList_Resampled " + str(iStates+1) + ":\n",np.round(StatusList_Resampled[iStates], 3),"\n")

    # === SYSTEM NOISE =========================================================================
    StatusList_Posterior = []
    SysNoise = []
    for iStates in range(nStates):
        AveV = Mean_SysNoise[iStates]
        StdV = SD_SysNoise[iStates]
        for iPn in range(Pn - nFixP):
            SysNoise.append(float(rd.normal(AveV, StdV)))
        for m in range(Pn - nFixP, Pn):
            SysNoise.append(float(rd.normal(AveV, 0.00)))
        SysNoise[OrgPn] = 0.00  # For OrgP
        # print("System noise" + str(iStates+1) + ": ",SysNoise)
        StatusList_withSysNoise = [Val + Noise for Val, Noise in zip(StatusList_Resampled[iStates], SysNoise)]
        # print("StatusList_Posterior" + str(iStates+1) + ": ",StatusList_withSysNoise)
        StatusList_Posterior.append(StatusList_withSysNoise)
        # DA switch
        if DA_flag == 0:
            SysNoise = [0.0] * Pn


    # Result of Resampling Process to log file
    CalcLog_f = RsltDir_now + '/' + PresentTimeTxt + '_' + CalcLog
    with open(CalcLog_f, 'w', encoding='utf-8') as f:
        f.write("===========================================================================" + "\n")
        f.write("Calculated datetime :" + PresentTimeTxt + "\n")
        f.write("===========================================================================" + "\n")
        f.write("Prediction station  :" + PredictionPoint + "\n\n")
        f.write("---------------------------------------------------------------------------" + "\n")
        val_str = ', '.join([format(x, '.3f') for x in Likelihood_Prior])
        f.write("Likelihood_Prior    :\n" + val_str + "\n")
        f.write("     ↓↓↓" + "\n")
        val_str = ', '.join([format(x, '.3f') for x in Likelihood_Posterior])
        f.write("Likelihood_Posterior:\n" + val_str + "\n\n")
        f.write("---------------------------------------------------------------------------" + "\n")
        val_str = ', '.join([str(x) for x in Resampling_Posterior])
        f.write("Resampling_Posterior:\n" + val_str + "\n")
        f.write('               └≫ Best particle: #' + str(BestPn) + "\n\n")
        f.write("===========================================================================" + "\n")
        for iStates in range(nStates):
            f.write("[State"+ str(iStates+1) + "] States/Parameters: No." + str(ls_StatesNo[iStates]) + "\n")
            val_str = ', '.join([format(x, '.3f') for x in StatusList_Initial[iStates]])
            f.write(" (1)Initial dis.:\n" + val_str + "\n")
            f.write("       ↓ <<< Option Func." + "\n")
            val_str = ', '.join([format(x, '.3f') for x in StatusList_Prior[iStates]])
            f.write(" (2)Prior distr.:\n" + val_str + "\n")
            f.write("       ↓ <<< Likelihood" + "\n")
            val_str = ', '.join([format(x, '.3f') for x in StatusList_Resampled[iStates]])
            f.write(" (3)Resampled d.:\n" + val_str + "\n")
            f.write("       ↓ <<< System noise" + "\n")
            val_str = ', '.join([format(x, '.3f') for x in StatusList_Posterior[iStates]])
            f.write(" (4)Posterior d.:\n" + val_str + "\n\n")
            f.write("---------------------------------------------------------------------------" + "\n")


    # === Results & Archives =========================================================================
    # --- RESULTS ---
    # 1. Save the files of the particle filter system in "Results"
    # ResultsDir: (HomeDir)/Results/YYYY/MM/DD
    # (1) States/Parameters: YYYY/MM/DD/YYYYMMDDhhmm_ParticleDistribution#.log
    for iStates in range(nStates):
        with open(RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + ParticleDistLog[iStates], 'w') as f:
            f.write(",".join(map(str, StatusList_Prior[iStates])) + "\n")
            f.write(",".join(map(str, StatusList_Posterior[iStates])) + "\n")
    # (2) Resampling: YYYYMMDDhhmm_Resampling.log
    with open(RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + Resampling_f, 'w') as f:
        f.write(",".join(map(str, Resampling_Prior)) + "\n")
        f.write(",".join(map(str, Resampling_Posterior)) + "\n")
    # (3) Likelihood: YYYYMMDDhhmm_Likelihood.log
    with open(RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + Likekihood_f, 'w') as f:
        f.write(",".join(map(str, Likelihood_Posterior)) + "\n")
    # (4) Eval Sigma: YYYYMMDDhhmm_Sigma.log
    with open(RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + Sigma_f, 'w') as f:
        f.write(str(Sigma_ErrH) + "\n")
    
    # 2. Save the forecasted value by the weights (Inner product of ensemble value and likelihood) ---
    # (1) [AllP] Save ensemble qr/qr2wl
    All_calc_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_AllP_qr_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
    df_calcQTime_sum.to_csv(path_or_buf = All_calc_f, header = True, index = False, encoding = "shift_jis")
    if ConvQ2H != 0:
        All_calc_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_AllP_wl_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_calcTime_sum.to_csv(path_or_buf = All_calc_f, header = True, index = False, encoding = "shift_jis")
    # (2) [WtMean] Save forecasted WL/qr with the Weights
    WtMeanQ = np.dot(Likelihood, qr_sum.T)
    df_WtMeanQ = pd.DataFrame(WtMeanQ, columns = ['WtMean'])
    df_WtMeanQTime = pd.concat([df_TimeSeries, df_WtMeanQ], axis=1)
    WtMeanQ_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_WtMean_qr_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
    df_WtMeanQTime.to_csv(path_or_buf = WtMeanQ_f, header = True, index = False, encoding = "shift_jis")
    if ConvQ2H != 0:
        WtMeanH = np.dot(Likelihood, calc_sum.T)
        df_WtMeanH = pd.DataFrame(WtMeanH, columns = ['WtMean'])
        df_WtMeanHTime = pd.concat([df_TimeSeries, df_WtMeanH], axis=1)
        WtMeanH_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_WtMean_wl_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_WtMeanHTime.to_csv(path_or_buf = WtMeanH_f, header = True, index = False, encoding = "shift_jis")
    # (3) [BestP] Save best forecasted WL/qr
    BestQ_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_BestP' + str(BestPn).zfill(3) + '_qr_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
    df_calcQTime_sum.to_csv(path_or_buf=BestQ_f, columns=['Datetime','P' + str(BestPn).zfill(5)], header=True, index=False, encoding="shift_jis")
    if ConvQ2H != 0:
        BestH_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_BestP' + str(BestPn).zfill(3) + '_wl_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_calcTime_sum.to_csv(path_or_buf=BestH_f, columns=['Datetime','P' + str(BestPn).zfill(5)], header=True, index=False, encoding="shift_jis")
    # (4) [OrgP] Save original forecasted WL/qr
    OrgQ_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_OrgP' + str(OrgPn).zfill(3) + '_qr_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
    df_calcQTime_sum.to_csv(path_or_buf=OrgQ_f, columns=['Datetime','P' + str(OrgPn).zfill(5)], header=True, index=False, encoding="shift_jis")
    if ConvQ2H != 0:
        OrgH_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_OrgP' + str(OrgPn).zfill(3) + '_wl_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_calcTime_sum.to_csv(path_or_buf=OrgH_f, columns=['Datetime','P' + str(OrgPn).zfill(5)], header=True, index=False, encoding="shift_jis")
    # (5) [Rain] Save rainfall data (AllP/Org/Best)
    Org_rain_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_OrgP' + str(OrgPn).zfill(3) + '_rain_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
    df_rain_sum.to_csv(path_or_buf=Org_rain_f, columns=['Datetime','P' + str(OrgPn).zfill(5)], header=True, index=False, encoding="shift_jis")
    if (6 in ls_StatesNo) == True or (61 in ls_StatesNo) == True or (62 in ls_StatesNo) == True:
        All_rain_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_AllP_rain_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_rain_sum.to_csv(path_or_buf = All_rain_f, header = True, index = False, encoding = "shift_jis")
        Best_rain_f = RsltDir_now + '/' + PresentTimeTxt + '_' + PredictionPoint + '_BestP' + str(BestPn).zfill(3) + '_rain_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_rain_sum.to_csv(path_or_buf=Best_rain_f, columns=['Datetime','P' + str(BestPn).zfill(5)], header=True, index=False, encoding="shift_jis")
    # (6)[Optional] For non-DA points; Archive result of RRI add in 2020 ---------
    if Rslt_OtherSt == 1 and nOut > 1:
        SubFunc.output_loc_point(Hydro_f_type, Hydro_f, nOut, Loc_Name, Hydro_locNo, Tn, Pn, CalcDir, RsltDir_now, PresentTimeTxt, BT_dy, FT_dy, RRI_dt_min, df_TimeSeries, Likelihood_Posterior)
        # (e.g., YYYYMMDDhhmm_(LocName)_All_P/WtMean_qr_Pn064_BT03d_FT06h.csv)

    # 3. Save the state-spaces (hs/hr/qr & BestP/OrgP/WtMean) for the next 6 days to display the system ---
    # (1)[BestP] hs/hr/qr_BestP_xxxxxx.out
    if Rslt_Best_all == 1:
        print('Best particle: #' + str(BestPn) + '   >>> saved')
        Target_states = ['hs','hr','qr']
        if L51ga_init_flg == 1: Target_states.append('ga')
        for iTarget in Target_states:
            for iFiles in range(int((BT_dy + FT_dy) * 1440 / RRI_dt_min)):
                DoTime = PresentTime - datetime.timedelta(days = BT_dy) + datetime.timedelta(minutes = (1+iFiles)*RRI_dt_min)
                CalcP_f = CalcDir + '/Particle' + str(BestPn).zfill(5) + '/RRI/out/' + iTarget + '_' + str(iFiles+1).zfill(6) + '.out'
                RlstP_f = RsltDir_now + '/' + PresentTimeTxt + '_' + iTarget + '_BestP_FT' + '{:%Y%m%d%H%M}'.format(DoTime) + '.out'   # + str(iFiles+1).zfill(6) + '.out'
                shutil.copyfile(CalcP_f, RlstP_f)
    # (2)[OrgP] hs/hr/qr_OrgP_xxxxxx.out
    if Rslt_Org_all == 1:
        print('Original particle: #' + str(OrgPn) + '   >>> saved')
        Target_states = ['hs','hr','qr']
        for iTarget in Target_states:
            for iFiles in range(int((BT_dy + FT_dy) * 1440 / RRI_dt_min)):
                DoTime = PresentTime - datetime.timedelta(days = BT_dy) + datetime.timedelta(minutes = (1+iFiles)*RRI_dt_min)
                CalcP_f = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/out/' + iTarget + '_' + str(iFiles+1).zfill(6) + '.out'
                RlstP_f = RsltDir_now + '/' + PresentTimeTxt + '_' + iTarget + '_OrgP_FT' + '{:%Y%m%d%H%M}'.format(DoTime) + '.out'    # + str(iFiles+1).zfill(6) + '.out'
                shutil.copyfile(CalcP_f, RlstP_f)
    # (3)[WtMean] hs/hr/qr_WtMean_xxxxxx.out
    Target_states = []
    if Rslt_WtMean_hs == 1: Target_states.append('hs')
    if Rslt_WtMean_hr == 1: Target_states.append('hr')
    if Rslt_WtMean_ga == 1 and L51ga_init_flg == 1: Target_states.append('ga')
    if Rslt_WtMean_qr == 1: Target_states.append('qr')
    if len(Target_states) != 0:
        for iTarget in Target_states:
            print('Weight mean particle: [ ' + iTarget + ' ]   >>> calculating...')
            for iFiles in range(int((BT_dy + FT_dy) * 1440 / RRI_dt_min)):
                DoTime = PresentTime - datetime.timedelta(days = BT_dy) + datetime.timedelta(minutes = (1+iFiles)*RRI_dt_min)
                State_WtM = np.zeros((nrows,ncols))   # nrows & ncols are obtained from dem file.
                for iPn in range(Pn):
                    Wt_v = Likelihood_Posterior[iPn]
                    CalcP_f = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/out/' + iTarget + '_' + str(iFiles+1).zfill(6) + '.out'
                    State_org = np.loadtxt(CalcP_f)
                    State_WtP = np.where(State_org < 0, State_org, State_org * Wt_v)
                    State_WtM = State_WtM + State_WtP
                RlstP_f = RsltDir_now + '/' + PresentTimeTxt + '_' + iTarget + '_WtMean_FT' + '{:%Y%m%d%H%M}'.format(DoTime) + '.out'
                np.savetxt(RlstP_f, State_WtM, fmt='%.3f')
    
    # --- ARCHIVES ---
    # 1. Archives the state-space in RRI for the next step (=NextTimeTxt)
    print ("State-space of the RRI model archiving ... ------------------------------------------------")
    Target_states = ['hs','hr']
    if L51ga_init_flg == 1: Target_states.append('ga')
    # (1) [BEST] Archive the state-space of best particle from 'Results' ---------
    for iTarget in Target_states:
        BestP_states = CalcDir + '/Particle' + str(BestPn).zfill(5) + '/RRI/out/' + iTarget + '_' + str(int(PF_dt_min/RRI_dt_min)).zfill(6) + '.out'
        BestP_seq_f = ArchDir_next_RRI + '/' + NextTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + iTarget + '_init.out'
        shutil.copyfile(BestP_states, BestP_seq_f)
    # (2) [ORG] Archive the state-space of original particle from 'Results' ---------
    for iTarget in Target_states:
        OrgP_states = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/out/' + iTarget + '_' + str(int(PF_dt_min/RRI_dt_min)).zfill(6) + '.out'
        OrgP_seq_f = ArchDir_next_RRI + '/' + NextTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + iTarget + '_init.out'
        shutil.copyfile(OrgP_states, OrgP_seq_f)
    # (3) [WtMean] Archive the state-space of weight mean from 'Results' ---------
    for iTarget in Target_states:
        State_WtM = np.zeros((nrows,ncols))   # nrows & ncols are obtained from dem file.
        for iPn in range(Pn):
            Wt_v = Likelihood_Posterior[iPn]
            CalcP_f = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/out/' + iTarget + '_' + str(int(PF_dt_min/RRI_dt_min)).zfill(6) + '.out'
            State_org = np.loadtxt(CalcP_f)
            State_WtP = np.where(State_org < 0, State_org, State_org * Wt_v)
            State_WtM = State_WtM + State_WtP
        WtMean_seq_f = ArchDir_next_RRI + '/' + NextTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + iTarget + '_init.out'
        np.savetxt(WtMean_seq_f, State_WtM, fmt='%.3f')
    # (4) [EachP] Archive the state-space of each particle from 'Results' ---------
    if SequentialConditions_RRI == 3:
        for iTarget in Target_states:
            for iPn in range(Pn):
                next_states = CalcDir + '/Particle' + str(iPn + 1).zfill(5) + '/RRI/out/' + iTarget + '_' + str(int(PF_dt_min/RRI_dt_min)).zfill(6) + '.out'
                seq_f = ArchDir_next_RRI + '/' + NextTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + iTarget + '_init.out'
                shutil.copy2(next_states, seq_f)
    print('              >>> done!\n')

    # 2. Archives the 'RRI_Input.txt' for the next step (=NextTimeTxt)
    print ("Parameters of the RRI model archiving ... ------------------------------------------------")
    next_para_f = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/RRI_Input.txt'
    seq_para_f = ArchDir_next_RRI + '/' + NextTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_' + RRI_InputTxt
    shutil.copy2(next_para_f, seq_para_f)
    print('              >>> done!\n')

    # 3. Archives the conditions in the River model for the next step (=NextTimeTxt)
    print ("Conditions of the River model archiving ... ------------------------------------------------")
    # "CorrectB" is a tantative procedure.
    CorrectB = 0.0
    CorrectedWL_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_' + CorrectedWL
    with open(CorrectedWL_f, 'w') as f:
        f.write(str(CorrectB) + "\n")
    if ConvQ2H == 1:
        print("  [Method] >>> HQ equation")
        # HQData = '(Homedir)/ObsData/River/HQeq.ini'  by ReadIniFile.py
        Next_HQeq_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_' + HQeq_f
        shutil.copy2(HQData, Next_HQeq_f)
    elif ConvQ2H == 4:
        print("  [Method] >>> HQ table")
        # HQData = '(Homedir)/ObsData/River/HQtab.csv'  by ReadIniFile.py
        Next_HQtab_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_' + HQtab_f
        shutil.copy2(HQData, Next_HQtab_f)
    elif ConvQ2H == 2:
        print("  [Method] >>> Uniform flow")
        # HQData = '(Homedir)/ObsData/River/HQtab.csv'  by ReadIniFile.py
        Next_HQtab_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_' + HQtab_f
        shutil.copy2(HQData, Next_HQtab_f)
    elif ConvQ2H == 2 or ConvQ2H == 21 or ConvQ2H == 31:
        print("  [Method] >>> Uniform flow")
        if ((ConvQ2H == 21 or ConvQ2H == 31) and (7 in ls_StatesNo) == False):
            print ("Settings error!!! \n You should set that the states are estimated sediment depthes (set as StatesNo=7).")
            sys.exit()
        Next_Sec_Org_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_' + CrossSec_f
        shutil.copy2(SecData, Next_Sec_Org_f)
        Next_RivState_Org_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_' + RivState_f
        shutil.copy2(SecCondIni, Next_RivState_Org_f)
        for iPn in range(Pn):
            print("  [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particles;")
            # (1) Copy a H-Q table file ('Particles' >>> 'Archives')
            HQtab_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + HQtab_f
            Next_HQtab_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_P' + str(iPn+1).zfill(5) + '_' + HQtab_f
            shutil.copy2(HQtab_p_f, Next_HQtab_f)
            # (2) Copy a cross section file ('Particles' >>> 'Archives')
            Sec_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + CrossSec_f
            Next_Sec_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_P' + str(iPn+1).zfill(5) + '_' + CrossSec_f
            shutil.copy2(Sec_p_f, Next_Sec_f)
            # (3) Copy a river state file ('Particles' >>> 'Archives')
            RivState_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/River/' + RivState_f
            Next_RivState_f = ArchDir_next_Riv + '/' + NextTimeTxt + '_' + PredictionPoint + '_P' + str(iPn+1).zfill(5) + '_' + RivState_f
            shutil.copy2(RivState_p_f, Next_RivState_f)
    print('              >>> done!\n')
    
    # --- Remove result (*.out) ---
    print("Removing result of the RRI Model...")
    msg = SubFunc.RemoveOutfile(CalcDir, TmpDir, Pn)
    print(msg)

    # --- Drawing figure ---
    # Draw the result of hydrograph for forecasting
    print('*** Now drawing... ***')
    Draw_f = HomeDir + DrawFlaFile
    file_data = open(Draw_f, "r")
    lines = file_data.readlines()
    DrawingFunc = lines[0]
    file_data.close()
    if DrawingFunc == '1':
        MsgBox = ""
        # if ConvQ2H == 3 or ConvQ2H == 33:
        #     if nFailure > 0: MsgBox = msg_Unst
        plt = Drawing.DrawHydro(PresentTimeTxt, ConvQ2H, HomeDir, Hydro_f_type, Hydro_f, Hydro_locNo, RsltDir, ObsData, \
                HQData, SecData, Pn, nStates, ls_StatesNo, BT_dy, FT_dy, ForecastType, OrgPn, \
                RRI_dt_min, PF_dt_min, MsgBox)
        OutDir = RsltDir + '/Drawing/'
        if os.path.isdir(OutDir) == False:os.mkdir(OutDir)
        if ForecastType == 'J':
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainJ_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        elif ForecastType == 'Y':
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainY_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        elif ForecastType == 'F':
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainF_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        elif ForecastType == 'AF-J':
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainAF-PredJ_Comb_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        elif ForecastType == 'AF-Y':
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainAF-PredY_comb_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        elif ForecastType == 'AF-F':
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainAF-PredF_comb_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        else:
            OutFigName = OutDir + PredictionPoint + '_' + PresentTimeTxt + '_hydro_RainXXX' + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.png'
        plt.savefig(OutFigName)
        plt.close()

    # === UPDATE TIME =========================================================================
    # Update Time
    Datetime_f = open(HomeDir + '/' + DatetimeFile, 'w')
    Datetime_f.write(NextTimeTxt)
    Datetime_f.close()
    
    # End of program
    msg = 'Data assimilation time: ' + str(PresentTime) + '>>> finished a PF cycle!!!'
    print(msg)
    with open(RsltDir_now + '/' + PresentTimeTxt + '_' + 'Finished.flg', 'w') as f:
        f.write(msg + "\n")
    sys.exit()
    
