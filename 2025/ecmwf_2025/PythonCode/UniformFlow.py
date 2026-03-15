# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import pandas as pd
import copy
#import Sediment


def INTERSEC(x1,y1,x2,y2,el):
    if x1 != x2:
        a = (y2 - y1)/(x2 - x1)
        b = y1
        dx = ( el - b ) / a # distance from x1
    else:
        dx = 0
    xx = x1 + dx
    return xx

def XY2SlopeIntercept(x1,y1,x2,y2):
    if x1 != x2:
        a = (y2 - y1)/(x2 - x1)
        b = y1 - a * x1
    return a, b
    
def SecRevise_flat(OrgXYN, SD):
    # Average-roughness is calculated by simple mean.
    NewXYN = np.zeros((1, 3), dtype = float)
    #print NewXYN
    NewXYN[0, 0] = OrgXYN[0, 0]
    NewXYN[0, 1] = OrgXYN[0, 1]
    SkipCnt = 0
    SumN = 0
    for i in range(OrgXYN.shape[0]-1):
        x1 = OrgXYN[i,0]
        y1 = OrgXYN[i,1]
        #n1 = OrgXYN[i,2]
        x2 = OrgXYN[i+1,0]
        y2 = OrgXYN[i+1,1]
        n2 = OrgXYN[i+1,2]
        if y1 > SD and y2 < SD:
            xx = INTERSEC(x1,y1,x2,y2,SD)
            yy = SD
            NN = (SumN + n2) / (SkipCnt + 1)
            NewXYN = np.append(NewXYN, np.array([[xx, yy, NN]]), axis=0)
            SumN = n2
            SkipCnt = 1
        elif y1 < SD and y2 > SD:
            xx = INTERSEC(x1,y1,x2,y2,SD)
            yy = SD
            NN = (SumN + n2) / (SkipCnt + 1)
            NewXYN = np.append(NewXYN, np.array([[xx, yy, NN]]), axis=0)
            xx = x2
            yy = y2
            NN = n2
            NewXYN = np.append(NewXYN, np.array([[xx, yy, NN]]), axis=0)
            SumN = 0
            SkipCnt = 0
        elif y1 >= SD and y2 >= SD:
            xx = x2
            yy = y2
            NN = (SumN + n2) / (SkipCnt + 1)
            NewXYN = np.append(NewXYN, np.array([[xx, yy, NN]]), axis=0)
            SumN = 0
            SkipCnt = 0
        elif y1 <= SD and y2 <= SD:
            SumN += n2
            SkipCnt += 1
    #print NewXYN
    return NewXYN

def SecRevise_rate(OrgXYN, SD):
    # Average-roughness is calculated by simple mean.
    #NewXYN = OrgXYN[:,:]
    NewXYN = copy.deepcopy(OrgXYN)
    OrgDepthMax = OrgXYN[:,1].max()
    OrgDepthMin = OrgXYN[:,1].min()
    for i in range(NewXYN.shape[0]-1):
        NewXYN[i,1] = OrgXYN[i,1] + (OrgDepthMax - OrgXYN[i,1])/(OrgDepthMax - OrgDepthMin) * SD
    #print OrgXYN
    #print NewXYN
    return NewXYN

def Read_CrossSection(SecDir):
    #SecDir  = 'C:/RRI/RRI-PF/RivSeri_PF/ObsData/River/SectionXYN.csv'
    # --- [Read the "section X-Y data"] ---
    SecElev = np.loadtxt(SecDir, delimiter=",", skiprows=1)
    nPoint = SecElev.shape[0]
    nCol = SecElev.shape[1]
    # Get the zero datumn elevation
    ZeroElev = SecElev[0, 2]
    #print 'Zero elevation = ',ZeroElev
    return SecElev, ZeroElev


def Read_RivConditions(RivState_init):
    #RivState_init = 'C:/RRI/RRI-PF/RivSeri_PF/Particle/InitialConditions/RivState_init.txt'
    # --- [Read the "river bed slope"] ---
    RivState = np.loadtxt(RivState_init)
    RivGrad = 1.0 / RivState[0]
    SedDepth = RivState[1]
    return RivGrad, SedDepth


def Sec2HQ_table(SecElev, ZeroElev, RivGrad, SedDepth):
    # River bed is converted elevation to zero height
    SecZeroH = copy.deepcopy(SecElev)               #preparation
    SecZeroH[:,1] = SecZeroH[:,1] - ZeroElev    #elev >>> zeroH
    dH = SecZeroH[:,1].min()                    #delta height
    #print(SecZeroH)

    # River bed is converted elevation to river depth
    SecDepth = copy.deepcopy(SecElev)               #preparation
    MaxElev = SecElev[:,1].max()                #elev
    MinElev = SecElev[:,1].min()                #elev
    SecDepth[:,1] = SecElev[:,1] - MinElev      #elev >>> depth
    MaxDepth = SecDepth[:,1].max()              #depth

    # Revise considering sediment deposition depth
    #SedDepth=1     #test
    if SedDepth != 0:
        #print ("**************************** \n  Updating section data!!  \n****************************")
        #SecDepth = SecRevise_flat(SecDepth, SedDepth)
        SecDepth = SecRevise_rate(SecDepth, SedDepth)
    SecZeroH_New = copy.deepcopy(SecElev)               #preparation
    SecZeroH_New[:,1] = SecDepth[:,1] + dH + ZeroElev  #depth >>> elev
    SecZeroH_New[0,2] = SedDepth
    #print(SecZeroH_New)

    # Make H-Q table ---------------------------------------------------------------------------------
    HQ = np.zeros((1, 2), dtype = float)
    # Calculate uniform-flow
    MaxDepth = SecDepth[:,1].max()
    MinDepth = SecDepth[:,1].min()
    #HQ[0, 0] = SedDepth + 0.0 #case of depth
    HQ[0, 0] = SedDepth + dH    #case of zero height
    #HQ[0, 0] = SedDepth + dH + ZeroElev   #case of elevation
    HQ[0, 1] = 0.0

    nDevide = 25.0
    dWL = (MaxDepth - MinDepth) / nDevide
    WL = MinDepth + dWL
    #WL = MinElev + 0.1

    while WL < MaxDepth:
        S = 0.0
        A = 0.0
        N = 0.0
        # 1) Charactaristic of the section data, S and A
        for i in range(SecDepth.shape[0]-1):
            dS = 0.0
            dA = 0.0
            dL = 0.0
            x1 = SecDepth[i,0]
            y1 = SecDepth[i,1]
            n1 = SecDepth[i,2]
            x2 = SecDepth[i+1,0]
            y2 = SecDepth[i+1,1]
            n2 = SecDepth[i+1,2]
            if y1 > WL and y2 < WL:
                x1 = INTERSEC(x1,y1,x2,y2,WL)
                y1 = WL
                dS = np.sqrt(np.square(y2-y1)+np.square(x2-x1))
                dA = 0.5 * ( ( WL - y1 ) + ( WL- y2 ) ) * ( x2 - x1 )
            elif y1 < WL and y2 > WL:
                x2 = INTERSEC(x1,y1,x2,y2,WL)
                y2 = WL
                dS = np.sqrt(np.square(y2-y1)+np.square(x2-x1))
                dA = 0.5 * ( ( WL - y1 ) + ( WL- y2 ) ) * ( x2 - x1 )
            elif y1 >= WL and y2 >= WL:
                dS = 0
                dA = 0
            elif y1 <= WL and y2 <= WL:
                dS = np.sqrt(np.square(y2-y1)+np.square(x2-x1))
                dA = 0.5 * ( ( WL - y1 ) + ( WL- y2 ) ) * ( x2 - x1 )
            dN = n2
            dN = dS * dN
            #print dS, dA, dN
            S += dS
            A += dA
            N += dN
        # 2) Calculation of the average-roughness
        #print S, A, N
        if S > 0.0:
            R = A / S
            Roughness = N / S
            v = ( R ** ( 2.0 / 3.0 ) ) * ( RivGrad ** ( 1.0 / 2.0 ) ) / Roughness
            Q = A * v
        else:
            R = 0.0
            v = 0.0
            Q = 0.0
        #HQ = np.append(HQ, np.array([[WL, Q]]), axis=0) #Depth
        HQ = np.append(HQ, np.array([[WL + dH, Q]]), axis=0) #Zero height
        #HQ = np.append(HQ, np.array([[WL + dH + ZeroElev, Q]]), axis=0) #Elevation
        WL += dWL   # Calclate water level every delt water level
    #Check the HQ table
    #print HQ
    return HQ, SecZeroH_New

#Revised HQ relation
def ReviseHQ_table(table):
    n = table.shape[0]
    table_rev = copy.deepcopy(table)
    for i in range(n-1):
        q1 = table[i,1]
        q2 = table[i+1,1]
        if q2 > q1:
            continue
        elif q2 <= q1:
            table_rev = np.delete(table_rev, i+1, 0)
            #print 'delete this Row'
    max_v = table[n-1,0] + 0.6
    table_rev = np.append(table_rev, np.array([[max_v,99999]]), axis=0)
    #print table_rev
    return table_rev


#Convert qr array to hr array batch processing for the number of HQ table
#calc_sum = UniformFlow.CalcQ2H(qr_sum, HQ_table, Tn, Pn)
def ConvQ2H_all_HQtab(ary_q, table):
    n_t = ary_q.shape[0]
    n_p = ary_q.shape[1]
    n_hq = table.shape[0]
    conv_sum = np.zeros([n_t,n_p])
    ary_a = np.zeros(n_hq, dtype = float)
    ary_b = np.zeros(n_hq, dtype = float)
    for i in range(n_hq-1):
        tmp_sum = ary_q
        y1 = table[i, 0]
        y2 = table[i+1, 0]
        x1 = table[i, 1]
        x2 = table[i+1, 1]
        ary_a[i] = (y2 - y1)/(x2 - x1)
        ary_b[i] = y1 - ary_a[i] * x1
        Q_threshold_l = table[i, 1]
        Q_threshold_u = table[i+1, 1]
        tmp_sum = np.where((tmp_sum >= Q_threshold_l) & (tmp_sum <= Q_threshold_u) , ary_a[i] * tmp_sum + ary_b[i], 0.0)
        conv_sum = conv_sum + tmp_sum
    return conv_sum


#Convert qr array to hr array for the each particle
#calc_sum = UniformFlow.CalcQ2H(qr_sum, HQ_table, Tn, Pn)
def CalcQ2H_EachP(ary_q, table):
    n_hq = table.shape[0]
    n_t = len(ary_q)
    conv_sum = np.zeros([n_t])
    ary_a = np.zeros(n_hq, dtype = float)
    ary_b = np.zeros(n_hq, dtype = float)
    for i in range(n_hq-1):
        tmp_sum = ary_q
        y1 = table[i, 0]
        y2 = table[i+1, 0]
        x1 = table[i, 1]
        x2 = table[i+1, 1]
        ary_a[i] = (y2 - y1)/(x2 - x1)
        ary_b[i] = y1 - ary_a[i] * x1
        Q_threshold_l = table[i, 1]
        Q_threshold_u = table[i+1, 1]
        tmp_sum = np.where((tmp_sum >= Q_threshold_l) & (tmp_sum <= Q_threshold_u) , ary_a[i] * tmp_sum + ary_b[i], 0.0)
        conv_sum = conv_sum + tmp_sum
    return conv_sum


#Convert qr_t to hr_t (1 by 1)
def ConvQ2H_1x1_HQtab(ary_q, table):
    n_hq = table.shape[0]
    n_t = len(ary_q)
    q2h = np.zeros([n_t])
    ary_a = np.zeros(n_hq, dtype = float)
    ary_b = np.zeros(n_hq, dtype = float)
    for i in range(n_hq-1):
        tmp_q = ary_q
        y1 = table[i, 0]
        y2 = table[i+1, 0]
        x1 = table[i, 1]
        x2 = table[i+1, 1]
        ary_a[i] = (y2 - y1)/(x2 - x1)
        ary_b[i] = y1 - ary_a[i] * x1
        Q_threshold_l = table[i, 1]
        Q_threshold_u = table[i+1, 1]
        tmp_q = np.where((tmp_q >= Q_threshold_l) & (tmp_q <= Q_threshold_u), \
                          ary_a[i] * tmp_q + ary_b[i], 0.0)
        q2h = q2h + tmp_q
    return q2h

def RevisedSec(SecElev, ZeroElev, SedDepth):
    # River bed is converted elevation to zero height
    SecZeroH = copy.deepcopy(SecElev)               #preparation
    SecZeroH[:,1] = SecZeroH[:,1] - ZeroElev    #elev >>> zeroH
    dH = SecZeroH[:,1].min()                    #delta height

    # River bed is converted elevation to river depth
    SecDepth = copy.deepcopy(SecElev)               #preparation
    MaxElev = SecElev[:,1].max()                #elev
    MinElev = SecElev[:,1].min()                #elev
    SecDepth[:,1] = SecElev[:,1] - MinElev      #elev >>> depth
    MaxDepth = SecDepth[:,1].max()              #depth
    
    # Revise considering sediment deposition depth
    if SedDepth != 0:
        #print ("**************************** \n  Updating section data!!  \n****************************")
        #SecDepth = SecRevise_flat(SecDepth, SedDepth)
        SecDepth = SecRevise_rate(SecDepth, SedDepth)
    
    return SecDepth
    

#Convert discharge to water level for a forecast including a riverbed evolution
#conv_qr2hr = UniformFlow.CalcQ2H_RiverbedEvo(ElementQR, SecXY, ZeroElev, RivGrad, SediDepth_BTFT)
def CalcQ2H_RiverbedEvo_UniformFlow(ary_q, SecElev, ZeroElev, RivGrad, SedDepth):
    n_t = len(ary_q)
    ary_h = np.zeros([n_t])
    #n_d = len(SedimentDepth)
    qt = np.zeros(n_t, dtype = float)
    for t in xrange(n_t):
        #(0) Sediment depth
        SedDepth_t = SedDepth[t]
        #(1) Make HQ by SecRevise_rate
        HQ = Sec2HQ_table(SecElev, ZeroElev, RivGrad, SedDepth_t)
        HQ_rev = ReviseHQ_table(HQ)
        #(2) Get discharge
        #(3) Calculate the water level by uniform Flow
        qt_t = ary_q[t]
        q2h = CalcQ2H_1to1(qt_t, HQ_rev)
        #print('Q = ' + str(qt_t))
        #print('H = ' + str(q2h))
        qt[t] = q2h
        ary_h[t] = q2h
    #print(ary_h)
    return ary_h
