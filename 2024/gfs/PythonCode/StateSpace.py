# -*- coding: utf-8 -*-

# --- Procedure of selected state space ---
import numpy as np
import numpy.random as rd
import pandas as pd
import StateSpace
import random

# Case of state space of hs, hr, parameter
def StateSpace_filtering(Pn, nFixP, FixedValue, Mean, SD):
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean    # Mean=1.0
    for iPn in range(1,Pn - nFixP):
        StatusList[iPn] = rd.normal(Mean, SD)
    if nFixP > 0:
        n = 0
        for iPn in range(Pn - nFixP, Pn):
            StatusList[iPn] = FixedValue[n]
            n += 1
    return StatusList

# Case of rate of hs/hr/rain
def StateSpace_rate_abs_normal(Pn, nFixP, FixedValue, Mean, SD):
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean    # Mean=1.0
    for iPn in range(1,Pn - nFixP):
        StatusList[iPn] = abs(rd.normal(Mean, SD))
    if nFixP > 0:
        n = 0
        for iPn in range(Pn - nFixP, Pn):
            StatusList[iPn] = FixedValue[n]
            n += 1
    return StatusList

# Case of add of hs/hr/rain
def StateSpace_rate_normal(Pn, nFixP, FixedValue, Mean, SD):
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean    # Mean=1.0
    for iPn in range(1,Pn - nFixP):
        StatusList[iPn] = rd.normal(Mean, SD)
    if nFixP > 0:
        n = 0
        for iPn in range(Pn - nFixP, Pn):
            StatusList[iPn] = FixedValue[n]
            n += 1
    return StatusList

def StateSpace_rate_abs_log(Pn, nFixP, FixedValue, Mean, SD):
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean    # Mean=1.0
    for iPn in range(Pn - nFixP):
        StatusList[iPn] = abs(random.lognormvariate(Mean, SD))
    if nFixP > 0:
        n = 0
        for iPn in range(Pn - nFixP, Pn):
            StatusList[iPn] = FixedValue[n]
            n += 1
    return StatusList

def StateSpace_rate_log(Pn, nFixP, FixedValue, Mean, SD):
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean    # Mean=1.0
    for iPn in range(1,Pn - nFixP):
        StatusList[iPn] = random.lognormvariate(Mean, SD)
    if nFixP > 0:
        n = 0
        for iPn in range(Pn - nFixP, Pn):
            StatusList[iPn] = FixedValue[n]
            n += 1
    return StatusList

# Case of position for rain distribution
def StateSpace_position_normal(Pn, nFixP, FixedValue, Mean, SD):
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean    # Mean=1.0
    for iPn in range(1,Pn - nFixP):
        StatusList[iPn] = rd.uniform(Mean-SD,Mean+SD)
    if nFixP > 0:
        n = 0
        for iPn in range(Pn - nFixP, Pn):
            StatusList[iPn] = FixedValue[n]
            n += 1
    return StatusList

def RRI_para(org, Line_rri_input, Mean_initDist, SD_initDist, Pn, nFixP, FixedValue):
    # Target columns >>> 3  [NOTES] This column means "Forest" in PRISM project.
    Target_clm = 3
    StatusList = np.zeros(Pn)
    StatusList[0] = Mean_initDist
    #Ln20-36: define land-use column
    Col = 0
    if 20 <= Line_rri_input <= 36:
        Col = Target_clm - 1     
    Ln_rri_input = org[Line_rri_input - 1].split()
    Ln_rri_input[Col] = Ln_rri_input[Col].replace('d', 'e')
    ParaVal = float(Ln_rri_input[Col])
    for iPn in range(1,Pn - nFixP):
        StatusList[iPn] = rd.normal(Mean_initDist, SD_initDist)
    # FixP is next step
    # if nFixP > 0:
    #     n = 0
    #     for iPn in range(Pn - nFixP, Pn):
    #         StatusList[iPn] = FixedValue[n]
    #         n += 1
    return ParaVal, StatusList

# Initial state space list
def InitialCondition(SelectStates, Line_rri_input, Pn, nFixP, \
    FixedValue, Init_rri_input, Option, SedDepth, SedPotential, PF_dt_min):
    if SelectStates == 0:   # add on 2024.11.14 by YN
        print("SelectStates = 0 >>> RRI-hs filtering")
        Mean_SysNoise = 1.0
        SD_SysNoise = 0.2
        StatusList = StateSpace_filtering(Pn, nFixP, FixedValue, Mean_SysNoise, SD_SysNoise)
        # e.g., StatusList=[1.0,1.0,...,(fixed value)] *finally add on system noise
    elif SelectStates == 2:   # add on 2024.11.14 by YN
        print("SelectStates = 2 >>> RRI-hr filtering")
        Mean_SysNoise = 1.0
        SD_SysNoise = 0.2
        StatusList = StateSpace_filtering(Pn, nFixP, FixedValue, Mean_SysNoise, SD_SysNoise)
        # e.g., StatusList=[1.0,1.0,...,(fixed value)] *finally add on system noise
    elif SelectStates == 1:     #1:hs_rate
        print("SelectStates = 1 >>> hs_rate")
        if Option == 1: print("      Option = 1 >>> activate")
        # For normal distribution parameters
        Mean_initDist = 1.0
        SD_initDist = 0.4
        StatusList = StateSpace_rate_abs_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
        # StatusList = StateSpace_rate_abs_log(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    elif SelectStates == 3:     #3:hr_rate
        print("SelectStates = 3 >>> hr_rate")
        Mean_initDist = 1.0
        SD_initDist = 0.4
        StatusList = StateSpace_rate_abs_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
        # StatusList = StateSpace_rate_abs_log(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    elif SelectStates == 5:     #5:RRI_Input.txt
        # Mean_SysNoise = 0.0
        # SD_SysNoise = 0.2
        # StatusList = StateSpace_filtering(Pn, nFixP, FixedValue, Mean_SysNoise, SD_SysNoise)
        # # e.g., StatusList=[1.0,1.0,...,(fixed value)] *finally add on system noise
        print("SelectStates = 5 >>> RRI-para filtering")
        f = open(Init_rri_input)
        org = f.readlines()
        f.close()
        if Line_rri_input == 21:        #L21:ns_slope
            Mean_initDist = 0.0
            SD_initDist = 0.05
            ParaVal, StatusList = StateSpace.RRI_para(org, Line_rri_input, Mean_initDist, SD_initDist, Pn, nFixP, FixedValue)
            StatusList = ParaVal * StatusList
        elif Line_rri_input == 28:      #L28:ka
            Mean_initDist = 1.0
            SD_initDist = 1.0
            ParaVal, StatusList = StateSpace.RRI_para(org, Line_rri_input, Mean_initDist, SD_initDist, Pn, nFixP, FixedValue)
            StatusList = ParaVal * StatusList
        elif Line_rri_input == 18:      #L18:ns_river
            Mean_initDist = 0.0
            SD_initDist = 0.01      # 0.025 ~ 0.045
            ParaVal, StatusList = StateSpace.RRI_para(org, Line_rri_input, Mean_initDist, SD_initDist, Pn, nFixP, FixedValue)
            StatusList = ParaVal + StatusList
        elif Line_rri_input == 23:      #L23:porosity
            Mean_initDist = 0.8
            SD_initDist = 0.1
            ParaVal, StatusList = StateSpace.RRI_para(org, Line_rri_input, Mean_initDist, SD_initDist, Pn, nFixP, FixedValue)
            StatusList = ParaVal * StatusList
        # # exception handling: Fixed value Para * Fixed value(1.0)
        if nFixP > 0:
            n = 0
            for i in range(Pn - nFixP, Pn):
                StatusList[i] = FixedValue[n]
                n += 1
        # check minus value >>> abs|value| 
        for iPn in range(Pn):
            if StatusList[iPn] < 0:
                StatusList[iPn] = abs(StatusList[iPn])
    elif SelectStates == 6:     #6:rain-ratio
        print("SelectStates = 6 >>> rainfall ratio")
        if Option == 1: print("      Option = 1 >>> activate")
        Mean_initDist = 1.0
        SD_initDist = 0.4
        StatusList = StateSpace_rate_abs_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    elif SelectStates == 61:     #61:rain-position_lon_x
        print("SelectStates = 61 >>> rainfall position (lon)")
        Mean_initDist = 0.0
        SD_initDist = 0.50
        StatusList = StateSpace_position_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    elif SelectStates == 62:     #62:rain-position_lat_y
        print("SelectStates = 62 >>> rainfall position (lat)")
        Mean_initDist = 0.0
        SD_initDist = 0.50
        StatusList = StateSpace_position_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    elif SelectStates == 7: #7:Sediment depth
        print("SelectStates = 7 >>> Sediment depth (delta)")
        if Option == 1: print("      Option = 1 >>> activate")
        Mean_initDist = SedDepth
        SD_initDist = SedPotential / (60 / PF_dt_min)
        StatusList = StateSpace_rate_abs_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    elif SelectStates == 8: #8:boundary Q
        print("SelectStates = 8 >>> boundary discharge (ratio)")
        if Option == 1: print("      Option = 1 >>> activate")
        Mean_initDist = 1.0
        SD_initDist = 0.4
        StatusList = StateSpace_rate_abs_normal(Pn, nFixP, FixedValue, Mean_initDist, SD_initDist)
    return StatusList



# Change states for hs file
def Change_StateSpace_hs_rate(SelectStates, StatusVal, hs_f, Option):     # divide on 2024.10.18
    if SelectStates == 0 or SelectStates == 1:   #0, 1: hs
        print("      SelectStates = ",SelectStates,">>> RRI-hs is changing...")
        if Option == 1: print("         └≫ Option: Active")
        LimitV = 1.0    # <<< threshold ratio
        AveV = 1.0      # <<< Mean: basically around 1.0
        StdV = 0.1      # <<< Standard deviation
        StatusVal = StateSpace.Rate(StatusVal, hs_f, Option, LimitV, AveV, StdV)
        return StatusVal

def Change_StateSpace_hr_rate(SelectStates, StatusVal, hr_f, Option):     # divide on 2024.10.18
    if SelectStates == 2 or SelectStates == 3:   #2, 3: hr
        print("      SelectStates = ",SelectStates,">>> RRI-hr is changing...")
        if Option == 1: print("         └≫ Option: Active")
        LimitV = 1.0    # <<< threshold ratio
        AveV = 1.0      # <<< Mean: basically around 1.0
        StdV = 0.1      # <<< Standard deviation
        StatusVal = StateSpace.Rate(StatusVal, hr_f, Option, LimitV, AveV, StdV)
        return StatusVal

def Change_RRI_Input(SelectStates, StatusVal, Ln_rri_input, rri_input_path): # divide on 2024.10.18
    if SelectStates == 5 or SelectStates == 61 or SelectStates == 62: #5: RRI parameters, #61: rain_lon, #62: rain_lat
        print("      SelectStates = ",SelectStates,">>> RRI-para (Ln = ",Ln_rri_input,") is changing...")
        StateSpace.Para(StatusVal, Ln_rri_input, rri_input_path)

def Change_RainTxt(SelectStates, StatusVal, rain_f, BT_dy, Option):       # divide on 2024.10.18
    if SelectStates == 6: #6: rainfall distribution
        print("      SelectStates = 6 >>> rainfall distribution (ratio)")
        LimitV = 1.0
        AveV = 1.0
        StdV = 0.1
        if Option == 1: print("         └≫ Option: Active")
        StatusVal = StateSpace.Rain(StatusVal, rain_f, BT_dy, Option, LimitV, AveV, StdV)
        return StatusVal

def Change_BoundQHTxt(SelectStates, f_path, ls_DA, ls_Ratio, ls_Option, str_loc_i, str_loc_j):       # add on 2024.11.05
    if SelectStates == 8: #8: boundary Q/H in RRI
        print("      SelectStates = 8 >>> boundary Q/H (ratio)")
        nBound_Qr = len(ls_Ratio)
        data_org = pd.read_csv(f_path, header=None, skiprows=3, delim_whitespace=True)
        # print("data_v(org):", data_org)
        data_rev = StateSpace.Bound(ls_Ratio, data_org, ls_Option, ls_DA)
        # print("data_v(rev):", data_rev)
        # write the boundary file
        with open(f_path, 'w') as f:
            f.write(str(nBound_Qr) + "\n")
            f.write(str_loc_i + "\n")
            f.write(str_loc_j + "\n")
            for iTime in range(len(data_rev)):
                Time_Sec = str(int(data_rev.iat[iTime, 0]))
                bound_data = ''
                for iBound_Qr in range(nBound_Qr):
                    bound_data = bound_data + '     ' + str('{:0.3f}'.format(float(data_rev.iat[iTime, 1 + iBound_Qr])))
                f.write(Time_Sec + '     ' + bound_data + "\n")

def Rate(Ratio, f_path, Option, LimitV, AveV, StdV):
    Ratio_Before = format(Ratio, '.4f')
    org = np.loadtxt(f_path)
    if Option == 0:
        print("             State value:",Ratio_Before)
    elif Option == 1:
        if Ratio < LimitV:
            Ratio = abs(rd.normal(AveV, StdV))  # N(Mean, SD^2)=N(1, 0.5^2) 0~2
            print("             State value:",Ratio_Before," >>> ",format(Ratio, '.4f'))
        else:
            print("             State value:",Ratio_Before,"*NOT change")
    elif Option == 2:
        Ratio = abs(rd.normal(AveV, StdV))  # N(Mean, SD^2)=N(1, 0.5^2) 0~2
    new = np.where( org < 0, org, org * Ratio )
    # Write text file from data array
    f = open(f_path, 'w')
    np.savetxt(f, new, delimiter = "   ", fmt = "%.5f")
    f.close()
    return Ratio

def Add(Delta, f_path, Option, LimitV, AveV, StdV):
    Delta_Before = format(Delta, '.4f')
    org = np.loadtxt(f_path)
    if Option == 1:
        if Delta < LimitV:
            Delta = rd.normal(AveV, StdV)  # N(Mean, SD^2)=N(1, 0.5^2) 0~2
            print("             State value:",Delta_Before," >>> ",format(Delta, '.4f'))
    new = np.where(org < 0, org, org + Delta)
    new2 = np.where(new < 0, 0.0, new)
    # Write text file from data array
    f = open(f_path, 'w')
    np.savetxt(f, new2, delimiter = "   ", fmt = "%.5f")
    f.close()

def Para(ParaVal, Ln, f_path):
    # Target columns >>> 3  [NOTES] This column means "Forest" in PRISM project.
    Target_clm = 3
    #Ln20-36: define land-use column
    Col = 0
    if 20 <= Ln <= 36:
        Col = Target_clm - 1
    f = open(f_path)
    org = f.readlines()
    f.close()
    Ln_rri_input = org[Ln - 1].split()
    print("         [L" + str(Ln) + "] " + Ln_rri_input[-1] + ': ' + Ln_rri_input[Col] + "   >>>   " + str(format(ParaVal, '.4f')))
    Ln_rri_input[Col] = ParaVal
    org[Ln - 1] = " ".join([str(n) for n in Ln_rri_input]) + '\n'
    # Write text file from data array
    f = open(f_path, 'w')
    for x in org:
        f.write(str(x))
    f.close()

def Bound(ls_Ratio, df_data_v, ls_Option, ls_DA):       # add on 2024.11.06
    nBound_Qr = len(ls_Ratio)
    for iBound_Qr in range(nBound_Qr):
        Option = ls_Option[iBound_Qr]
        DA_flg = ls_DA[iBound_Qr]
        Ratio = ls_Ratio[iBound_Qr]
        if DA_flg == 1:
            print("            + [St.",iBound_Qr+1,"] DA func.: [ON]  >>> Ratio =", format(Ratio, '.3f'))
        elif DA_flg == 0:
            print("            + [St.",iBound_Qr+1,"] DA func.: [OFF] >>> Ratio =", format(Ratio, '.3f'))
        if Option == 1: print("                   └≫ Option: Active")
        df_data_v[1+iBound_Qr] = df_data_v[1+iBound_Qr] * Ratio
    return df_data_v

def Rain(Ratio, f_path, BT_dy, Option, LimitV, AveV, StdV):
    Ratio_Before = format(Ratio, '.3f')
    f = open(f_path)
    RainsFile = f.readlines()
    f.close()
    #End line & 1st line
    Ln_End = len(RainsFile)
    Ln_1st = RainsFile[0].split()
    time = int(Ln_1st[0])       #time[sec]
    Num_loc_i = int(Ln_1st[2])  #latitude direction
    Num_loc_j = int(Ln_1st[1])  #longitude direction
    nTimes = int(Ln_End / (Num_loc_i + 1))
    # Write text file from data array
    f = open(f_path, 'w')
    if Option == 0:
        print("             State value:",Ratio_Before)
    elif Option == 1:
        if Ratio < LimitV:
            Ratio = abs(rd.normal(AveV, StdV))  # N(Mean, SD^2)=N(1, 0.5^2) 0~2
            print("             State value:",Ratio_Before," >>> ",format(Ratio, '.3f'))
        else:
            print("             State value:",Ratio_Before,"*NOT change")
    Ratio_Upd = Ratio
    for iTimes in range(nTimes):
        NowLn = RainsFile[iTimes*(Num_loc_i + 1)].split()
        NowTime = int(NowLn[0])
        NowLn_list = [[NowTime, Num_loc_j, Num_loc_i]]
        #np.savetxt(f, NowTime + Num_loc_j + Num_loc_i, fmt='% 10d')
        np.savetxt(f, NowLn_list, fmt='% 10d')
        #f.write(str(NowLn))
        org = np.zeros((Num_loc_i, Num_loc_j), dtype = float)
        for i in range(Num_loc_i):      # Bug fixed on 12th Nov, 2024 by YN
            Ln_i = RainsFile[iTimes*(Num_loc_i + 1)+i+1].split()
            for j in range(Num_loc_j):  # Bug fixed on 12th Nov, 2024 by YN
                org[i, j] = float(Ln_i[j])
                #org = 0
        # Default case: the correction including the prediction time (for 6 hours )
        # If you don't correct for prediction rainfall, activate under the two rows.
        # if NowTime > BT_dy * 3600:     # Bug fixed on 18th Dec, 2024 by YN
        #     Ratio = 1.0
        new = org * Ratio
        np.savetxt(f, new, delimiter = "   ", fmt = "%.5f")
    f.close()
    return Ratio_Upd
