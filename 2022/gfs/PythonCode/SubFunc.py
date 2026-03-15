# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import pandas as pd
import datetime
import os
import shutil


def df_SortedObs(df_obs, PresentTimeTxt, BackTimeTxt):   # back time + 1step ~ the present time
    if BackTimeTxt == PresentTimeTxt:
        for i, v in df_obs.iterrows():
            if str(v['Datetime']) == PresentTimeTxt:
                Present_idx = i
                break
        df_obs_sort = pd.DataFrame(df_obs.iloc[[Present_idx],:])
    else:
        FindTimeTxt = PresentTimeTxt
        Limit_FW = 3 #[hour]
        SearchInterval = 10     # minimum: 10 [minutes] (now fix)
        if FindTimeTxt in df_obs.values:
            #print('Present time data could be find')
            Present_idx = (df_obs[df_obs['Datetime'].isin([FindTimeTxt])]).index[0]
        elif not FindTimeTxt in df_obs.values:
            #print('Present time data could be not find')
            nBack = int(Limit_FW * 60 / SearchInterval)     
            for x in range(nBack):
                FindTime = datetime.datetime.strptime(FindTimeTxt, '%Y%m%d%H%M')
                FindTime = FindTime - datetime.timedelta(minutes = SearchInterval)
                FindTimeTxt = '{:%Y%m%d%H%M}'.format(FindTime)
                if FindTimeTxt in df_obs.values:
                    #print('Near time data could be find')
                    Present_idx = (df_obs[df_obs['Datetime'].isin([FindTimeTxt])]).index[0]
                    break
        #print(Present_idx)
        FindTimeTxt = BackTimeTxt
        if FindTimeTxt in df_obs.values:
            #print('Back time data could be find')
            Back_idx = (df_obs[df_obs['Datetime'].isin([FindTimeTxt])]).index[0]
        elif not FindTimeTxt in df_obs.values:
            Limit_BK = 24 #[hour]
            #print('Back time data could be not find')
            nBack = int(Limit_BK * 60 / SearchInterval)
            for x in range(nBack):
                FindTime = datetime.datetime.strptime(FindTimeTxt, '%Y%m%d%H%M')
                FindTime = FindTime - datetime.timedelta(minutes = SearchInterval)
                FindTimeTxt = '{:%Y%m%d%H%M}'.format(FindTime)
                if FindTimeTxt in df_obs.values:
                    #print('Near time data could be find')
                    Back_idx = (df_obs[df_obs['Datetime'].isin([FindTimeTxt])]).index[0]
                    break
        #print(Back_idx)
        df_obs_sort = pd.DataFrame(df_obs.iloc[Back_idx+1:Present_idx+1,:])
        #print("df_obs_sort = ",df_obs_sort)
    return df_obs_sort


def df_Sort4Bound(df_bound, ForwardTimeTxt, BackTimeTxt):  # back time + 0step ~ the forward(=forecast) time
    if BackTimeTxt == ForwardTimeTxt:       # The case of non-backtime
        for i, v in df_bound.iterrows():    # 1 by 1 row
            if str(v['Datetime']) == ForwardTimeTxt:   # Case of match!
                Forward_idx = i
                break
        df_bound_sort = pd.DataFrame(df_bound.iloc[[Forward_idx],:])
    else:                               # The case of backtime
        FindTimeTxt = ForwardTimeTxt
        Limit_FW = 3 #[hour]
        SearchInterval = 10   # [minute]
        if FindTimeTxt in df_bound.values:
            #print('Present time data could be find')
            Forward_idx = (df_bound[df_bound['Datetime'].isin([FindTimeTxt])]).index[0]
        elif not FindTimeTxt in df_bound.values:
            #print('Present time data could be not find')
            nBack = int(Limit_FW * 60 / SearchInterval)
            for x in range(nBack):  # Previous version: nBack - 1
                FindTime = datetime.datetime.strptime(FindTimeTxt, '%Y%m%d%H%M')
                FindTime = FindTime - datetime.timedelta(minutes = SearchInterval)
                FindTimeTxt = '{:%Y%m%d%H%M}'.format(FindTime)
                if FindTimeTxt in df_bound.values:
                    #print('Near time data could be find')
                    Forward_idx = (df_bound[df_bound['Datetime'].isin([FindTimeTxt])]).index[0]
                    break
            # Change the datetime in the case of supplement.
            df_bound.iloc[Forward_idx, 0] = ForwardTimeTxt
        #print(Forward_idx)
        FindTimeTxt = BackTimeTxt
        if FindTimeTxt in df_bound.values:
            #print('Back time data could be find')
            Back_idx = (df_bound[df_bound['Datetime'].isin([FindTimeTxt])]).index[0]
        elif not FindTimeTxt in df_bound.values:
            Limit_BK = 24 #[hour]
            #print('Back time data could be not find')
            nBack = int(Limit_BK * 60 / SearchInterval) #[minute]
            for x in range(nBack):  # previous version: nBack - 1
                FindTime = datetime.datetime.strptime(FindTimeTxt, '%Y%m%d%H%M')
                FindTime = FindTime - datetime.timedelta(minutes = SearchInterval)
                FindTimeTxt = '{:%Y%m%d%H%M}'.format(FindTime)
                if FindTimeTxt in df_bound.values:
                    #print('Near time data could be find')
                    Back_idx = (df_bound[df_bound['Datetime'].isin([FindTimeTxt])]).index[0]
                    break
            if x < (nBack-1):
                # Change the datetime in the case of supplement.
                df_bound.iloc[Back_idx, 0] = BackTimeTxt
            elif x == (nBack-1):
                df_init = pd.DataFrame([BackTimeTxt, float(0.0)], index=df_bound.columns).T
                df_bound = pd.concat([df_init, df_bound], ignore_index=True, axis=0)
                # df_bound = df_bound.append(df_init)
                Back_idx = 0
                Forward_idx = Forward_idx + 1
        #print(Back_idx)
        df_bound_sort = pd.DataFrame(df_bound.iloc[Back_idx:Forward_idx+1,:])   # <<< diference obs wl
        #print(df_bound_sort)
    return df_bound_sort



def RemoveOutfile(CalcDir, TmpDir, Pn):
    # (1)remove 'tmp' folder in 'Particle'
    shutil.rmtree(TmpDir)
    os.mkdir(TmpDir)
    # (2)remove 'Particle#####' folder in 'Particle'
    for i in range(Pn):
        p = CalcDir + '/Particle' + str(i+1).zfill(5) + '/RRI/out/'
        shutil.rmtree(p)
        os.mkdir(p)
    msg = '   >>> Done!'
    return msg



def CheckDir2024(HomeDir, Pn):
    ErrorFlag = 0
    LogList = []
    CheckFolderLog = HomeDir + '/PythonCode/Calc.log'
    with open(CheckFolderLog, "w") as f:
        # [Now time]
        NowTime = datetime.datetime.now()
        NowText = '*********' + NowTime.strftime('%Y/%m/%d %H:%M:%S') + '*********'
        LogList.append(NowText)
        # ***[ObsData]**********************************************
        ObsFolder = HomeDir + '/ObsData'
        if os.path.exists(ObsFolder) == True:
            msgObs = 'OK! >>> [ObsData]'
        else:
            msgObs = 'NG! >>> [ObsData]'
            ErrorFlag = 1
        LogList.append(msgObs)
        # [ObsData > Discharge]
        DischargeFolder = ObsFolder + '/Discharge'
        if os.path.exists(DischargeFolder) == True:
            msgDischarge = 'OK! >>> [ObsData > Discharge]'
        else:
            msgDischarge = 'NG! >>> [ObsData > Discharge]'
            ErrorFlag = 1
        LogList.append(msgDischarge)
        # [ObsData > Rainfall]
        RainfallFolder = ObsFolder + '/Rainfall'
        if os.path.exists(RainfallFolder) == True:
            msgRainfall = 'OK! >>> [ObsData > Rainfall]'
        else:
            msgRainfall = 'NG! >>> [ObsData > Rainfall]'
            ErrorFlag = 1
        LogList.append(msgRainfall)
        # [ObsData > Rainfall > RainJ]
        RainJFolder = RainfallFolder + '/RainJ'
        if os.path.exists(RainJFolder) == True:
            msgRainJ = 'OK! >>> [ObsData > Rainfall > RainJ]'
        else:
            msgRainJ = 'NG! >>> [ObsData > Rainfall > RainJ] (e.g., Is "JmaJ" folder name old version??)!'
            ErrorFlag = 1
        LogList.append(msgRainJ)
        # [ObsData > Rainfall > JmaY]
        JmaYFolder = RainfallFolder + '/JmaY'
        if os.path.exists(JmaYFolder) == True:
            msgJmaY = 'OK! >>> [ObsData > Rainfall > JmaY]'
        else:
            msgJmaY = 'NG! >>> [ObsData > Rainfall > JmaY] *Not required depend on your condtions!'
            ErrorFlag = 1
        LogList.append(msgJmaY)
        # [ObsData > River]
        RiverFolder = ObsFolder + '/River'
        if os.path.exists(RiverFolder) == True:
            msgRiver = 'OK! >>> [ObsData > River]'
        else:
            msgRiver = 'NG! >>> [ObsData > River]'
            ErrorFlag = 1
        LogList.append(msgRiver)
        # [ObsData > WaterLevel]
        WaterLevelFolder = ObsFolder + '/WaterLevel'
        if os.path.exists(WaterLevelFolder) == True:
            msgWaterLevel = 'OK! >>> [ObsData > WaterLevel]'
        else:
            msgWaterLevel = 'NG! >>> [ObsData > WaterLevel]'
            ErrorFlag = 1
        LogList.append(msgWaterLevel)
        # ***[Particle]**********************************************
        ParticleFolder = HomeDir + '/Particle'
        if os.path.exists(ParticleFolder) == True:
            msgParticle = 'OK! >>> [Particle]'
        else:
            msgParticle = 'NG! >>> [Particle]'
            ErrorFlag = 1
        LogList.append(msgParticle)
        # [Particle > InitialConditions]
        InitialConditionsFolder = ParticleFolder + '/InitialConditions'
        if os.path.exists(InitialConditionsFolder) == True:
            msgInitialConditions = 'OK! >>> [Particle > InitialConditions]'
        else:
            msgInitialConditions = 'NG! >>> [Particle > InitialConditions]'
            ErrorFlag = 1
        LogList.append(msgInitialConditions)
        # [Particle > Particle#####]
        msgPn = 'Particle folder missing only...'
        for iPn in range(Pn):
            PnFolder = ParticleFolder + '/Particle' + str(iPn+1).zfill(5)
            if os.path.exists(PnFolder) == False:
                msgPn = 'NG! >>> [Particle > Particle##########]'
                ErrorFlag = 1
                LogList.append(msgPn)
        # ***[Results]**********************************************
        # If 'Results' is not exist, automatically made all directory.
        ResultsFolder = HomeDir + '/Results'
        if os.path.exists(ResultsFolder) == True:
            msgResults = '[Results] >>> OK!'
        else:
            msgResults = '[Results] >>> NG: Making Result directory...'
            ErrorFlag = 1
            os.mkdir(ResultsFolder)
        LogList.append(msgResults)
        # ***[Archives]**********************************************
        # If 'Results' is not exist, automatically made all directory.
        ArchivesFolder = HomeDir + '/Archives'
        if os.path.exists(ArchivesFolder) == True:
            msgArchives = '[Archives] >>> OK!'
        else:
            msgArchives = '[Results] >>> NG: Making Result directory...'
            ErrorFlag = 1
            os.mkdir(ArchivesFolder)
        LogList.append(msgArchives)

        # Output the result of confirm the folder construction to log-file
        f.write("\n".join(LogList))
    
    if ErrorFlag == 0:
        msg = '      >>> Done the folder construction'
    elif ErrorFlag == 1:
        msg = '      >>> [Notes] Check the "Calc.log"'
    return msg


# read tail line
def tail2list(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    Tail_line = lines[-1]
    Tail_List = [float(x.strip()) for x in Tail_line.split(',')]
    return Tail_List


# Non correct list
def NonCorrectlist(StatesNum, ls_StatesNo):
    ls_NonCorrect = []
    for iStatesNum in range(StatesNum):
        if ls_StatesNo[iStatesNum] == 0:
            ls_NonCorrect.append([1.0])
            continue
        elif ls_StatesNo[iStatesNum] == 1:
            ls_NonCorrect.append([1.0])
            continue
        elif ls_StatesNo[iStatesNum] == 2:
            ls_NonCorrect.append([0.0])
            continue
        elif ls_StatesNo[iStatesNum] == 3:
            ls_NonCorrect.append([1.0])
            continue
        elif ls_StatesNo[iStatesNum] == 4:
            ls_NonCorrect.append([0.0])
            continue
        elif ls_StatesNo[iStatesNum] == 5:
            ls_NonCorrect.append([1.0])
            continue
        elif ls_StatesNo[iStatesNum] == 6:
            ls_NonCorrect.append([1.0])
            continue
        elif ls_StatesNo[iStatesNum] == 7:
            ls_NonCorrect.append([0.0])
            continue
    return ls_NonCorrect


def ls_index(ls, value):
    return [i for i, x in enumerate(ls) if x == value]


def output_loc_point(Hydro_f_type, Hydro_f, nOut, Loc_Name, Hydro_locNo, Tn, Pn, CalcDir, RsltDir, PresentTimeTxt, BT_dy, FT_dy, RRI_dt_min, df_TimeSeries, Likelihood):
    # (2020) For output points; Archive result of RRI model ---------
    Name_p = []
    for iPn in range(Pn):
        Name_p.append('P' + str(iPn+1).zfill(5))    # e.g. ['P00001','P00002',...,'P0000N']
    # nOut = number of the location, Loc_Name = ['aaa','bbb','ccc',...,'nnn']
    for iOut in range(nOut):
        # --- 1) Summary of the calculated discharge for all particles
        hydro_sum = []
        if Hydro_f_type == 1:
            if (iOut+1) == Hydro_locNo:
                continue
            loc_col = 1 + iOut
            for iPn in range(Pn):
                hydro_path = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/' + Hydro_f
                hydro_p = []
                with open(hydro_path, 'r') as f:
                    for ln in f:
                        val = ln.split()
                        hydro_p.append(val[loc_col])
                hydro_sum.append(hydro_p)
        elif Hydro_f_type == 2:
            OutName = 'hydro_qr_' + Loc_Name[iOut] + '.txt'
            if OutName == Hydro_f:
                continue
            loc_col = 1
            for iPn in range(Pn):
                hydro_path = CalcDir + '/Particle' + str(iPn+1).zfill(5) + '/RRI/etc/calcHydro/' + OutName
                hydro_p = []
                with open(hydro_path, 'r') as f:
                    for ln in f:
                        val = ln.split()
                        hydro_p.append(val[loc_col])
                hydro_sum.append(hydro_p)
        qrO_sum = np.array(hydro_sum, dtype='float').T
        #print(qrO_sum)
        # --- 2) Stored the calculated discharge for all particles
        calcO_sum = np.zeros([Tn,Pn])
        calcO_sum = qrO_sum
        # --- 3) Add time series column, and Array to DataFrame
        df_calcO_sum = pd.DataFrame(calcO_sum, columns = Name_p)
        df_calcOTime_sum = pd.concat([df_TimeSeries, df_calcO_sum], axis=1)
        # --- 4) Archive to CSV for all particle
        AllParticleData_f = RsltDir + '/' + PresentTimeTxt + '_OtherSt_' + Loc_Name[iOut] + '_AllP_qr_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_calcOTime_sum.to_csv(path_or_buf = AllParticleData_f, header = True, index = False, encoding = "shift_jis")

        # (1) weight mean particle
        #All_hydroO = np.loadtxt(qrOsumName, dtype='float', delimiter=",", skiprows=1)
        All_hydroO = df_calcOTime_sum.to_numpy()
        f_T = All_hydroO.T
        TimeSeries = f_T[0]
        qrO_sum = f_T[1:]
        # Calculate in the weight mean at the output points
        WtMeanQ = np.dot(Likelihood, qrO_sum)
        df_WtMeanQ = pd.DataFrame(WtMeanQ, columns = ['WtMean'])
        df_WtMeanQTime = pd.concat([df_TimeSeries, df_WtMeanQ], axis=1)
        WtMean_f = RsltDir + '/' + PresentTimeTxt + '_OtherSt_' + Loc_Name[iOut] + '_WtMean_qr_Pn' + str(Pn).zfill(3) + '_BT' + str(BT_dy).zfill(2) + 'd_FT' + str(FT_dy).zfill(2) + 'd.csv'
        df_WtMeanQTime.to_csv(path_or_buf = WtMean_f, header = True, index = False, encoding = "shift_jis")


def rainBasin_extraction_position(loc_v, Ln, f_path):
    # loc_v = 134.567
    # Ln = 3
    # f_path = 'C:/RRI/RRI-PF/Minami_5s_No00_20230602Flood/Particles/Particle00002/RRI/etc/rainBasin_extraction/rainBasin_extraction.txt'
    #Ln3: xllcorner_rain / Ln4: yllcorner_rain
    Col = 0     # 1 column
    f = open(f_path, encoding='utf-8')
    org = f.readlines()
    f.close()
    Ln_txt = org[Ln - 1].split()
    Ln_txt[Col] = loc_v
    org[Ln - 1] = " ".join([str(n) for n in Ln_txt]) + '\n'
    # Write text file from data array
    f = open(f_path, 'w', encoding='utf-8')
    for x in org:
        f.write(str(x))
    f.close()
