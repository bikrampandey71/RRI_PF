# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import pandas as pd
import datetime
import shutil
import ReadiniFile
import SubFunc
import StateSpace
import glob

def datetime2elapstime(df_boundQr):
    nTime = len(df_boundQr)
    # nData = len(df_boundQr.columns)
    
    # Start time in the prediction time
    PredStartTimeTXT = df_boundQr.iat[0, 0]
    PredStartTime = datetime.datetime.strptime(PredStartTimeTXT, '%Y%m%d%H%M')
    # change elaps time [sec]
    for iTime in range(nTime):
        ElapsTimeTXT = df_boundQr.iat[iTime, 0]
        ElapsTime = datetime.datetime.strptime(ElapsTimeTXT, '%Y%m%d%H%M')
        delta = ElapsTime - PredStartTime
        df_boundQr.iat[iTime, 0] = delta.total_seconds()
    # print("df_boundQr: ",df_boundQr)
    return df_boundQr

def RRI_BoundQr(HomeDir, DateDir, RRI_con_Dir, CalcDir, BoundQr_ini, BoundaryQr, PresentTimeTxt, BackTime, ForecastTime, BT_day, FT_day, Pn, StatusList, ls_StatesNo, ls_StatesOp, EnsKF_bound):
    nBound_Qr, BoundQ_flag, BoundQ_DA, BoundQrFile, loc_i_Qr, loc_j_Qr = ReadiniFile.importBoundaryQCondition(BoundQr_ini)
    LimitV = 1.0
    AveV = 1.0
    StdV = 0.1
    for iBound_Qr in range(nBound_Qr):
        if BoundQ_flag[iBound_Qr] == 1 and BoundQrFile[iBound_Qr][0] == '.':    # if relative pass then direct pass
            BoundQrFile[iBound_Qr] = HomeDir + BoundQrFile[iBound_Qr][1:]
        elif BoundQ_flag[iBound_Qr] == 2:
            f_name = PresentTimeTxt+'_WtMeanQ_Pn'+str(Pn).zfill(3)+'_BT'+str(BT_dy).zfill(2)+'d_FT'+str(FT_dy).zfill(2)+'d.csv'
            BoundQrFile[iBound_Qr] = BoundQrFile[iBound_Qr] + '/Results/' + DateDir + '/' + f_name
    # print("  >>> BoundPoint = ",nBoundQ)
    BoundQ_Name = []
    BoundQ_f = []
    BoundQ_iloc = []
    BoundQ_jloc = []
    for iBound_Qr in range(nBound_Qr):
        if BoundQ_flag[iBound_Qr] >= 1:
            BoundQ_Name.append('BoundQ' + str(iBound_Qr + 1))
            BoundQ_f.append(BoundQrFile[iBound_Qr])
            BoundQ_iloc.append(loc_i_Qr[iBound_Qr])
            BoundQ_jloc.append(loc_j_Qr[iBound_Qr])
    # Make a qr_bound.txt
    dt_bound = 10 #[min]
    # df all datetime (back-forecst)
    dates = pd.date_range(start=BackTime, end=ForecastTime, freq=str(dt_bound)+'min', name='tmp')    # <<< append this df
    df_boundQr_0 = pd.DataFrame(dates)
    df_boundQr_0['Datetime'] = pd.DataFrame(df_boundQr_0['tmp'].dt.strftime('%Y%m%d%H%M'))
    df_boundQr_1 = pd.DataFrame(df_boundQr_0['Datetime'])
    # print(df_boundQr_1)   # df_boundQr_1: YYYYMMDDhhmm format
    
    ForecastTimeTxt = ForecastTime.strftime('%Y%m%d%H%M')
    BackTimeTxt = BackTime.strftime('%Y%m%d%H%M')
    for iBound_Qr in range(nBound_Qr):
        Name1 = BoundQ_Name[iBound_Qr]
        df_bound = pd.read_csv(BoundQ_f[iBound_Qr], header=0, names=('Datetime',Name1), dtype={'Datetime':'object',Name1:'float'})
        # Sort boundary condition back time to forecast time
        df_boundQr_sort = SubFunc.df_Sort4Bound(df_bound, ForecastTimeTxt, BackTimeTxt)
        # Merge df_datetime with df_SelectedQ
        df_boundQr_1 = pd.merge(df_boundQr_1, df_boundQr_sort, how='left', on='Datetime')
    # interporation of nan
    df_boundQr = df_boundQr_1.interpolate()     # does not work 'both', Why?
    df_boundQr = df_boundQr.fillna(method='bfill')
    # datetime >>> Elaps time [s]
    df_boundQr_sec = datetime2elapstime(df_boundQr)
    # print("df_boundQr_sec: ",df_boundQr_sec)

    # Write 'qr_bound_YYYYMMDDhhmmBTxxFTxx.txt' in 'Result' folder
    Bound_Org_f = RRI_con_Dir + '/' + PresentTimeTxt + '_' + BoundaryQr
    with open(Bound_Org_f, 'w') as f:
        f.write(str(nBound_Qr) + "\n")
        str_loc_i = 'loc_i     ' + '     '.join([str(n) for n in BoundQ_iloc])
        str_loc_j = 'loc_j     ' + '     '.join([str(n) for n in BoundQ_jloc])
        f.write(str_loc_i + "\n")
        f.write(str_loc_j + "\n")
        for iTime in range(len(df_boundQr_sec)):
            Time_Sec = str(int(df_boundQr_sec.iat[iTime, 0]))
            bound_data = ''
            for iBound_Qr in range(nBound_Qr):
                bound_data = bound_data + '     ' + str('{:0.3f}'.format(float(df_boundQr_sec.iat[iTime, 1 + iBound_Qr])))
            f.write(Time_Sec + '     ' + bound_data + "\n")

    # RRI-qr conditions (move file & change the state-space)
    for iPn in range(Pn):
        print(" - [ " + str(iPn + 1) + " / " + str(Pn) + " ] Particle;")
        bound_seq_p_f = CalcDir  + '/Particle' + str(iPn+1).zfill(5) + '/RRI/bound/' + BoundaryQr
        shutil.copy2(Bound_Org_f, bound_seq_p_f)
        if (8 in ls_StatesNo) == True:
            nBoundDA = sum(BoundQ_DA)
            # print("Number of the boundary point for DA: ",nBoundDA)
            BoundQ_idx = [i for i, x in enumerate(ls_StatesNo) if x == 8]
            BoundQ_ratio = []
            BoundQ_opt = []
            DA_cnt = 0
            for iBound_Qr in range(nBound_Qr):
                if BoundQ_DA[iBound_Qr] == 1:
                    if EnsKF_bound == 1:
                        BoundQ_ratio.append(rd.normal(1.0, 0.4))
                    elif EnsKF_bound == 0:
                        BoundQ_ratio.append(StatusList[BoundQ_idx[DA_cnt]][iPn])
                    BoundQ_opt.append(ls_StatesOp[BoundQ_idx[DA_cnt]])
                    # [Optional] ratio < 1.0 then normalization
                    if BoundQ_opt[iBound_Qr] == 1 and BoundQ_ratio[iBound_Qr] < LimitV:
                        BoundQ_ratio[iBound_Qr] = rd.normal(AveV, StdV)
                    # Update ratio
                    StatusList[BoundQ_idx[DA_cnt]][iPn] = BoundQ_ratio[iBound_Qr]
                    DA_cnt += 1
                elif BoundQ_DA[iBound_Qr] == 0:
                    BoundQ_ratio.append(1.0)  # (1.0) means time one
                    BoundQ_opt.append(0)  # (0) does not mean option
            # print("BoundQ_ratio:", BoundQ_ratio)
            # print("BoundQ_opt:", BoundQ_opt)
            StateSpace.Change_BoundQHTxt(8, bound_seq_p_f, BoundQ_DA, BoundQ_ratio, BoundQ_opt, str_loc_i, str_loc_j)  #_DA, _ratio, _opt: list type
            bound_Arch_f = RRI_con_Dir  + '/' + PresentTimeTxt + '_BT' + str(BT_dy).zfill(2) + '_P' + str(iPn+1).zfill(5) + '_' + BoundaryQr
            shutil.copy2(bound_seq_p_f, bound_Arch_f)
    return StatusList


