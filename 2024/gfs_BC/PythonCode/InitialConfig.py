# -*- coding: utf-8 -*-

import os
import sys
import shutil
import glob
from multiprocessing import Pool
import numpy as np
import datetime
from distutils.dir_util import copy_tree
import ReadiniFile
import HQeq
import StateSpace
import ReadRRI_input
import UniformFlow
import SubFunc

DatetimeFile = 'datetime.txt'       # Current time (Format: YYYYMMDDhhmm)
RRI_InputTxt = 'RRI_Input.txt'
SedimentPotential = 0.10            # The potential of sediment [Unit: m/hr]
Resampling_f = 'Resampling.log'     # Common file name of posterior particle distribution
Likekihood_f = 'Likelihood.log'     # Common file name of likelihood
Sigma_f =      'Sigma.log'          # Standard deviation of the error from observation
InitHs =       'hs_init.out'        # hs initial condition for hot start
InitHr =       'hr_init.out'        # hr initial condition for hot start
InitGA =       'ga_init.out'        # ga initial condition for hot start
HQeq_f =       'HQeq.ini'           # HQ equation (coef. a & b, and more)
HQtab_f =      'HQtab.csv'          # HQ table
CrossSec_f =   'CrossSec.csv'       # River cross section
RivState_f =   'RivState_init.txt'  # River condition for an uniform flow (Gradient: X [I=1/X], Sediment depth: D [m])
CorrectedWL =  'CorrectedWL.txt'    # Error of observation and calculation


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
Rslt_Best_all, Rslt_Org_all, Rslt_WtMean_hs, Rslt_WtMean_hr, Rslt_WtMean_ga, Rslt_WtMean_qr, Rslt_OtherSt\
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
Tn = int((BT_dy + FT_dy) * 1440 / RRI_dt_min)   # Numbers of step: back-time + forecast-time
LikelihoodCondition = 1 # Fixed constant
Sigma_ErrH = SigmaErr_Const

# (1) Original particle is [1]  Fixed from 2024.12.31 by YN
print("--- [Original particle infomation] ----------------------------------")
OrgPn = OrgParticle
print(("Original particle >>> [No. " + str(OrgPn) + "] (All: " + str(Pn)) + " particles)")
print("\n\n\n")



# (2) Special (Fixed) particles is [1]  Fixed from 2024.12.31 by YN
print("--- [Special particles infomation (Fixed value particles)] ----------------------------------")
print(("The number of special particles (Fixed P.) >>> [ N = " + str(nFixP) + " / " + str(Pn)) + " ]")
for iStates in range(nStates):
    print('States' + str(iStates+1) + ' = [ ' + str(ls_StatesNo[iStates]) + ' ]')
    ls_v = []
    for iFixP in range(nFixP):
        ls_v.append(FixedValue[iStates][iFixP]) 
    print('Fixed value: \n' + str(ls_v))
print("\n\n\n")



# (3) Make & Check the directory construction
msg = SubFunc.CheckDir2024(HomeDir, Pn)



# 2. Time management ----------
Datetime_f = open(HomeDir + '/' + DatetimeFile, 'r')
PresentTimeTxt = str(Datetime_f.readline())
Datetime_f.close()
if PresentTimeTxt != PF_StartTime:
    print("--- [Current time infomation] ---------------------------------------")
    print("'PresentTimeTxt'(" + PresentTimeTxt + ")!= 'PF_StartTime'(" + str(PF_StartTime) + ")")
    print("   [NOTES] Change the current time in 'datetime.txt' >>> " + str(PF_StartTime))
    Datetime_f = open(HomeDir + '/' + DatetimeFile, 'w')
    Datetime_f.write(str(PF_StartTime))
    Datetime_f.close()
    PresentTimeTxt = str(PF_StartTime)
    print("\n\n\n")



print("--- [Time infomation for the first time] ----------------------------")
PresentTime = datetime.datetime.strptime(PresentTimeTxt, '%Y%m%d%H%M')
YYYY_Now = PresentTime.strftime("%Y")
MM_Now = PresentTime.strftime("%m")
DD_Now = PresentTime.strftime("%d")
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
print("\n\n\n")



print("--- [Make folders: 'Results' & 'Archives'] --------------------------")
# Make 'Results/YYYY/MM/DD' & 'Archives/YYYY/MM/DD'
DateDir_Now = YYYY_Now + '/' + MM_Now + '/' + DD_Now
# RsltDir_Now = RsltDir + '/' + DateDir_Now
# os.makedirs(RsltDir_Now, exist_ok=True)
ArchDir_RRI = ArchDir + '/' + DateDir_Now + '/RRI'
os.makedirs(ArchDir_RRI, exist_ok=True)
ArchDir_Riv = ArchDir + '/' + DateDir_Now + '/River'
os.makedirs(ArchDir_Riv, exist_ok=True)

DateDir_Prev = PreviousTime.strftime("%Y") + '/' + PreviousTime.strftime("%m") + '/' + PreviousTime.strftime("%d")
RsltDir_Prev = RsltDir + '/' + DateDir_Prev
os.makedirs(RsltDir_Prev, exist_ok=True)
print("   >>> Done!")
print("\n\n\n")



# 4. Read settings of RRI model & River model ----------
# (1) read 'RRI_Input.txt' ----------
Init_rri_input = InitDir + '/' + RRI_InputTxt
print("--- [Check a file: 'RRI_Input.txt' in 'InitialConditions'] ----------")
if os.path.exists(Init_rri_input) == True:
    print("   >>> OK!")
elif os.path.exists(Init_rri_input) == False:
    print("   >>> 'RRI_Input.txt' file does NOT exist in 'InitialConditions' folder [Required]!!")
    sys.exit()
with open(Init_rri_input) as f:
    org = f.readlines()
L3Path_rain, L4Path_dem, L5Path_acc, L6Path_dir, L14rain_xll, L15rain_yll, \
L18ns_riv, L19Num_LU, L38Val_RivThresh, L39Val_Cw, L40Val_Sw, L41Val_Cd, L42Val_Sd, \
L43Val_HeightPara, L44Val_Height, L46Flg_RivFile, L47Path_Width, L48Path_Depth, L49Path_Height, \
L51hs_init_flg, L51hr_init_flg, L51hg_init_flg, L51ga_init_flg, L66Path_LU, L100Path_Loc = ReadRRI_input.Read_RRI_input(org)
print("\n\n\n")



# (2) Check 'location.txt' ----------
print("--- [Confirm a 'location.txt'] --------------------------------------")
Location_f = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/' + L100Path_Loc 
if os.path.exists(Location_f) == True:
    Loc_Name, Loc_i, Loc_j, nOut = ReadRRI_input.Read_location(Location_f)
elif os.path.exists(Location_f) == False:
    print ("   >>> 'location.txt' file does NOT exist!!")
    sys.exit()
print("\n\n\n")



# (3) Read indexes in 'dem.txt' ----------
DemFile = CalcDir + '/Particle' + str(OrgPn).zfill(5) + '/RRI/' + L4Path_dem
ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value = ReadRRI_input.Read_index(DemFile)

# (4) Remove result (*.out) ----------
print("Removing results of the RRI Model...")
TmpDir = CalcDir + '/tmp'
msg = SubFunc.RemoveOutfile(CalcDir, TmpDir, Pn)
print(msg)
print("\n\n\n")



# (5) PF previous results in 'PreviousTime': ParticleDistribution#.log >>> Resampling.log >>> Likelihood.log >>> Sigma.log----------
print("+++ [Preparing the previous results for the particle filter] ++++++++++++++++++")
#  1) ParticleDistribution#.log <<< States/parameters
ParticleDistLog = ['ParticleDistribution1.log'] # list of states/parameters file in timeseries (1st row: μ[Mean_initDist, SD_initDist])
if nStates > 1:
    for iStates in range(nStates-1):
        ParticleDistLog.append('ParticleDistribution' + str(iStates + 2) + '.log')
LogFileName = []
for iStates in range(nStates):
    LogFileName.append(RsltDir_Prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + ParticleDistLog[iStates])
StatusList_zero = np.zeros(Pn)
StatusList = []
SedDepth = 0.0  # Initial sediment depth [Unit: meter]
for iStates in range(nStates):
    StatusList_tmp = StateSpace.InitialCondition(ls_StatesNo[iStates], ls_RRI_Param[iStates], Pn, nFixP, \
    FixedValue[iStates], Init_rri_input, ls_StatesOp[iStates], SedDepth, SedimentPotential, PF_dt_min)
    # output initial dictribution
    with open(LogFileName[iStates], 'w') as f:
        f.writelines(",".join(map(str, StatusList_zero)) + "\n")
        f.writelines(",".join(map(str, StatusList_tmp)) + "\n")
    StatusList.append(StatusList_tmp.tolist())
    print("(" + str(iStates+1) + ") StatusList_Initial #" + str(iStates+1) + ":")
    print(np.round(StatusList_tmp, 2))
    print("\n")

#  2) Resampling.log <<< Initial value: Serial number from 1 to Pn
Resampled_f = RsltDir_Prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + Resampling_f
k0 = np.asanyarray(list(range(Pn))) # Set serial number (1~Pn)
with open(Resampled_f, 'w') as f:
    f.writelines(",".join(map(str, k0 + 1)) + "\n")
print("Resampling.log:\n" + str(k0+1) + "\n")

#  3) Likelihood.log <<< Initial value: all zero
LikelihoodFile = RsltDir_Prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + Likekihood_f
Likelihood = np.zeros(Pn)
with open(LikelihoodFile, 'w') as f:
    f.writelines(",".join(map(str, Likelihood)) + "\n")
print("Likelihood.log:\n" + str(Likelihood) + "\n")

#  4) Sigma.log <<< Initial value: SigmaErr_Const
SigmaLogFile = RsltDir_Prev + '/' + PreviousTimeTxt + '_' + PredictionPoint + '_' + Sigma_f
with open(SigmaLogFile, 'w') as f:
    f.writelines(str(Sigma_ErrH) + "\n")
print("\n\n\n")



# (5) RRI initial condition files in 'PresentTime': hs >>> hr >>> ga ----------
# Move from 'InitialConditions' to 'Result' ***NOTE: just only move (Not change state-space)***
print("+++ [Preparing initial conditions for the RRI model] ++++++++++++++++++++++++++")
# (hs file) ---
Keyword_hs = InitDir + '/hs_*_' + BackTimeTxt + '.out'
ListHs = glob.glob(Keyword_hs)
if len(ListHs)==0:
    print('!!Caution!!! Not found a hs-file of initial conditions. \n >>> Check a file or your "datetime.txt".')
    sys.exit()
Init_hs = ListHs[0]
# Move 'tmp' >>> 'Result' (hs=3: using initial disribution)
if SelectConditions_hs == 0:
    hs_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + InitHs
    shutil.copy2(Init_hs, hs_seq_f)
elif SelectConditions_hs == 1:
    hs_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + InitHs
    shutil.copy2(Init_hs, hs_seq_f)
elif SelectConditions_hs == 2:
    hs_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + InitHs
    shutil.copy2(Init_hs, hs_seq_f)
elif SelectConditions_hs == 3:
    for iPn in range(Pn):
        if iPn <= 5:
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        elif iPn <= 8:
            print('       .')
        elif iPn == Pn-1:
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        hs_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + InitHs
        shutil.copy2(Init_hs, hs_seq_f)
        if (0 in ls_StatesNo) == True:
            # Change state space with system noise
            iLs = ls_StatesNo.index(0)
            StatusVal = StatusList[iLs][iPn]
            StatusVal = StateSpace.Change_StateSpace_hs_rate(1, StatusVal, hs_seq_f, ls_StatesOp[iLs])
            if ls_StatesOp[iLs] == 1:
                StatusList[iLs][iPn] = StatusVal
print("   [hs] >>> Done!")

# (hr file) ---
Keyword_hr = InitDir + '/hr_*_' + BackTimeTxt + '.out'
ListHr = glob.glob(Keyword_hr)
if len(ListHr)==0:
    print('!!Caution!!! Not found a hr-file of initial conditions. \n >>> Check a file or your "datetime.txt".')
    sys.exit()
Init_hr = ListHr[0]
# Move 'tmp' >>> 'Result' (hr is not change!)
if SelectConditions_hr == 0:
    hr_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + InitHr
    shutil.copy2(Init_hr, hr_seq_f)
elif SelectConditions_hr == 1:
    hr_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + InitHr
    shutil.copy2(Init_hr, hr_seq_f)
elif SelectConditions_hr == 2:
    hr_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + InitHr
    shutil.copy2(Init_hr, hr_seq_f)
elif SelectConditions_hr == 3:
    for iPn in range(Pn):
        if iPn <= 5:
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        elif iPn <= 8:
            print('       .')
        elif iPn == Pn-1:
            print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        hr_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + InitHr
        shutil.copy2(Init_hr, hr_seq_f)
        if (2 in ls_StatesNo) == True:
            # Change state space with system noise
            iLs = ls_StatesNo.index(2)
            StatusVal = StatusList[iLs][iPn]
            StatusVal = StateSpace.Change_StateSpace_hr_rate(2, StatusVal, hr_seq_f, ls_StatesOp[iLs])
            if ls_StatesOp[iLs] == 1: StatusList[iLs][iPn] = StatusVal
print("   [hr] >>> Done!")

# (ga file) ---
if L51ga_init_flg == 1:     # case of initial file flag = 1 in 'RRI_Input.txt'
    Keyword_ga = InitDir + '/ga_*_' + BackTimeTxt + '.out'
    ListGA = glob.glob(Keyword_ga)
    if len(ListGA)==0:
        print('!!Caution!!! Not found a ga-file of initial conditions. \n >>> Check a file or your "datetime.txt".')
        sys.exit()
    Init_ga = ListGA[0]
    # Move 'tmp' >>> 'Result' (ga also is not change!)
    if SelectConditions_ga == 0:
        ga_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_OrgP_' + InitGA
        shutil.copy2(Init_ga, ga_seq_f)
    elif SelectConditions_ga == 1:
        ga_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_BestP_' + InitGA
        shutil.copy2(Init_ga, ga_seq_f)
    elif SelectConditions_ga == 2:
        ga_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_WtMean_' + InitGA
        shutil.copy2(Init_ga, ga_seq_f)
    elif SelectConditions_ga == 3:
        for iPn in range(Pn):
            ga_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + InitGA
            shutil.copy2(Init_ga, ga_seq_f)
print("   [ga] >>> Done!")

# (RRI_Input.txt) ---   SelectConditions_para
Init_rri_input = InitDir + '/' + RRI_InputTxt
ListPara = glob.glob(Init_rri_input)
if len(ListPara)==0:
    print('!!!Caution!!! Not found a "RRI_Input.txt" \n >>> Check a file or your "datetime.txt".')
    sys.exit()
para_seq_f = ArchDir_RRI + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_' + RRI_InputTxt
shutil.copy2(Init_rri_input, para_seq_f)
print("   [RRI_Input.txt] >>> Done!\n\n\n")

# 6) [River_conditions] Move 'InitialConditions' >>> 'Result' (e.g., HQ eq., HQ table, Cross section, Sediment depth, and more)
print("+++ [Preparing river model] +++++++++++++++++++++++++++++++++++++++++++++++++++")
RivGrad = 0.0
nHQ = 0
SecCondIni = InitDir + '/' + RivState_f
if ConvQ2H == 1:    # ConvQ2H == 1 >>> Convert by a HQ equation
    nHQ, HQ_a, HQ_b, HQ_Hmax, HQ_Qmax = HQeq.importHQeq(HQData)     # HQData <<< 'HQeq.ini'
    CorrectB = 0
    New_HQ_b = HQ_b + CorrectB
    New_HQ_Hmax = HQ_Hmax + CorrectB
    #print nHQ, HQ_a, HQ_b, HQ_Hmax, HQ_Qmax
    CorrectedWL_f = ArchDir_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + CorrectedWL
    with open(CorrectedWL_f, 'w') as f:
        f.write(str(CorrectB) + "\n")
    HQeq_f = ArchDir_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + HQeq_f
    with open(HQeq_f, 'w') as f:
        f.write("[HQ_num]" + "\n")
        f.write("nHQ = " + str(nHQ) + "\n")
        f.write("[HQ_para]" + "\n")
        for iHQ in range(nHQ):
            f.write("a" + str(iHQ+1) + " = " + str(HQ_a[iHQ]) + "\n")
            f.write("b" + str(iHQ+1) + " = " + str(New_HQ_b[iHQ]) + "\n")
            f.write("Hmax" + str(iHQ+1) + " = " + str(New_HQ_Hmax[iHQ]) + "\n")
    print("   [H-Q equation] >>> Done!")
elif ConvQ2H == 4:    # ConvQ2H == 4 >>> Convert by a HQ table
    # HQData = '(Homedir)/ObsData/River/HQtab.csv'  by ReadIniFile.py
    HQ_table = np.loadtxt(HQData, delimiter=",", skiprows=0)
    HQtab_f = ArchDir_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + HQtab_f
    with open(HQtab_f, 'w') as f:
        np.savetxt(f, HQ_table, delimiter = ",", fmt = "%.5f")
    print("   [H-Q table] >>> Done!")
elif ConvQ2H == 2 or ConvQ2H == 21 or ConvQ2H == 31:
    if ((ConvQ2H == 21 or ConvQ2H == 31) and (7 in ls_StatesNo) == False):
        print ("Settings error!!! \n The estimated state-space must include 'SelectStates = 7'!!")
        sys.exit()
    if ConvQ2H == 2:
        print("   [Method] An uniform flow")
    elif ConvQ2H == 21:
        print("   [Method] Riverbed evolution depend on non-Gausian")
    elif ConvQ2H == 31:
        print("   [Method] Riverbed evolution depend on Gausian")
    # SecXY, ZeroElev, RivGrad, SedDepth = UniformFlow.Read_SecConditions(SecData, SecCondIni)
    SecXY, ZeroElev = UniformFlow.Read_CrossSection(SecData)
    RivGrad, SedDepth = UniformFlow.Read_RivConditions(SecCondIni)
    # [Archive of HQ table] ----------
    HQ_table, NewSecXY = UniformFlow.Sec2HQ_table(SecXY, ZeroElev, RivGrad, SedDepth)
    HQ_table = UniformFlow.ReviseHQ_table(HQ_table)
    HQ_table_f = ArchDir_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + HQtab_f
    with open(HQ_table_f, 'w') as oFile:
        np.savetxt(oFile, HQ_table, delimiter = ",", fmt = "%.5f")
    # [Archive of Cross section] ----------
    Sec_f = ArchDir_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + CrossSec_f
    with open(Sec_f, 'w') as f:
        f.write("X,Y,Zero/N\n")
        for elem in NewSecXY:
            f.write(str(elem[0]) + ',' + str(elem[1]-ZeroElev) + ',' + str(elem[2]) + "\n")
    # [River condition, L1: river gradient, L2:sediment depth]
    SecState_f = ArchDir_Riv + '/' + PresentTimeTxt + '_' + PredictionPoint + '_' + RivState_f
    with open(SecState_f, 'w') as f:
        f.write(str(int(1/RivGrad)) + "\n")
        f.write(str(SedDepth) + "\n")
    print("             >>> Done!")
print("\n\n\n")

print('+--------------------------------------------------------------------+\n')
print('     [Success]\n')
print('     *** Prepared all conditions for flood simulation!!! ***\n\n\n')
print('+--------------------------------------------------------------------+')
