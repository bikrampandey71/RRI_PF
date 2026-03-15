# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import pandas as pd
import UniformFlow
import ReadiniFile
import math
import copy

def CalcQcQi(HomeDir):

    ini_Config4Sediment = 'SedimentConfig.ini'
    Config4Sediment_f = HomeDir + '/' + ini_Config4Sediment
    Section1Dir, B1, B2, RivGrad_T, Dist, SedDepth, Rn, TauSm, S, Dm, Di, Qc \
    = ReadiniFile.ReadSedimentCondition(Config4Sediment_f)

    RivGrad = 1.0/RivGrad_T
    h1 = []
    Fr1 = []
    Zb2_Elev = []
    #Qc = 0.0   #136.8

    ### (1) Calculated the critical tractive depth and its discharge ##################
    #TauSm = 0.05
    #S = 1.65 
    #Dm = 0.01
    hc = (S*Dm)/RivGrad*TauSm
    #print('hc = ' + str(hc) + ' [m]')
    Qc = (hc**(5.0/3.0))*(RivGrad**(1.0/2.0))*B2/Rn
    #print('Qc = ' + str(Qc) + ' [m3/s]')
    ### (1) Calculated the critical tractive depth and its discharge ##################

    ### (1') The condition of stoping riverbed evolution ##################
    Qi = 99999.9
    if Di > 0:
        #TauSm = 0.05
        #S = 1.65 
        #Di = 0.05 (i=90)
        TauSi = TauSm * ((math.log(19))**2) / ((math.log(19*Di/Dm))**2)
        hi = (S*Di)/RivGrad*TauSi
        #print('hc = ' + str(hc) + ' [m]')
        Qi = (hi**(5.0/3.0))*(RivGrad**(1.0/2.0))*B2/Rn
        #print('Qc = ' + str(Qc) + ' [m3/s]')
    ### (1') Calculated the critical tractive depth and its discharge ##################
    return Qc, Qi


def Q2RivDepth_Prediction(HomeDir, Q1, Qc, Qi, FlagQi):

    ini_Config4Sediment = 'SedimentConfig.ini'
    Config4Sediment_f = HomeDir + '/' + ini_Config4Sediment
    Section1Dir, B1, B2, RivGrad_T, Dist, SedDepth, Rn, TauSm, S, Dm, Di, Qc \
    = ReadiniFile.ReadSedimentCondition(Config4Sediment_f)
    
    #for a online
    Sec1_path = HomeDir + '/' + Section1Dir

    RivGrad = 1.0/RivGrad_T
    h1 = []
    Fr1 = []
    Zb2_Elev = []

    ### (1) Discharge table considering the critical discharge ##################
    # Read section: Ary(X, Y, Zero/N)
    Sec1_Elev, Zero1_Elev = UniformFlow.Read_CrossSection(Sec1_path)
    Zb1_Elev = np.min(Sec1_Elev[:,1])
    # Read discharge
    #print(Q1)
    Q1_sort = Q1
    #Q1_sort = np.where(Q1_sort>Qi, Qi, Q1_sort)
    DeclineWL_func = 1
    # DeclineWL_func = 0: riverbed can be lowered accoroding critical discharge
    # DeclineWL_func = 1: riverbed cannot be lowered
    # DeclineWL_func = 2: riverbed can be conditionally lowered (mixed particle size) 
    # DeclineWL_func = 3: Non-prediction for sediment 
    #print(DeltaZb)
    if DeclineWL_func == 0:
        Q1_sort = np.where(Q1_sort<Qc, Qc, Q1_sort)
    elif DeclineWL_func == 1:
        MaxQ = Q1_sort[0]
        for t in range(len(Q1_sort)):
            if Q1_sort[t] >= MaxQ:
                MaxQ = Q1_sort[t]
            elif Q1_sort[t] < MaxQ:
                Q1_sort[t] = MaxQ
    elif DeclineWL_func == 2:
        MaxQ = Q1_sort[0]
        for t in range(len(Q1_sort)): #When t=0, DeltaZb[0] is fixed.
            if FlagQi == 0:
                if Q1_sort[t] < Qc:
                    Q1_sort[t] = 0.01
                elif Q1_sort[t] > Qi:
                    FlagQi = 1
            elif FlagQi == 1:
                if Q1_sort[t] < Qi:
                    Q1_sort[t] = Qi
    #print(Q1_sort)
    #print(DeltaZb)
    ### (1) Discharge table considering the critical discharge ##################

    ### (2) Calculated the water depth and the Fluid Number at upper cross section ##################
    # Section + Discharge >>> HQ table
    HQ_table1 = UniformFlow.Sec2HQ_table(Sec1_Elev, Zero1_Elev, RivGrad, SedDepth)
    # HQ table >>> Revise HQ table
    HQ_table1 = UniformFlow.ReviseHQ_table(HQ_table1)

    h1 = UniformFlow.CalcQ2H_EachP(Q1_sort, HQ_table1)
    Fr1 = Q1_sort/B1/h1/((9.8*h1)**0.5)
    ### (2) Calculated the water depth and the Fluid Number at upper cross section ##################

    ### (3) Calculated the riverbed evolution ##################
    #DeltaZb = h1 * (1-(B1/B2)**(24.0/35.0)) + Fr1**2*(1-(B1/B2)**(32.0/35.0))/2.0
    DeltaZb2 = h1 * ((1-(B1/B2)**(24.0/35.0)) + 1.0/2.0*(Fr1**2)*(1-(B1/B2)**(32.0/35.0))) - RivGrad*Dist

    DeltaZb2_Elev = DeltaZb2 + Zb1_Elev

    return DeltaZb2_Elev

