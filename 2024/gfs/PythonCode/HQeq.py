# -*- coding: utf-8 -*-

# --- HQ式を取込み、H->QやQ->Hの変換を担う ---
import numpy as np
import configparser

# HQ.iniからparameterを呼び出しDataFrameで返す。
def importHQeq(f):
    iniFile = configparser.ConfigParser()
    iniFile.read(f)
# --- [HQ_num] ---
    nHQ = int(iniFile.get('HQ_num', 'nHQ'))
# --- [HQ_para] ---
    # Q = a*(H+b)^2の"a"
    HQ_a = 0
    if nHQ > 0:
        HQ_a = np.zeros(nHQ)
        for i in range(nHQ):
            name = 'a' + str(i + 1)
            HQ_a[i] = float(iniFile.get('HQ_para', name))
    # Q = a*(H+b)^2の"b"
    HQ_b = 0
    if nHQ > 0:
        HQ_b = np.zeros(nHQ)
        for i in range(nHQ):
            name = 'b' + str(i + 1)
            HQ_b[i] = float(iniFile.get('HQ_para', name))
    # Q = a*(H+b)^2の適用上限水位 <- Hmax
    HQ_Hmax = 0
    if nHQ > 0:
        HQ_Hmax = np.zeros(nHQ)
        for i in range(nHQ):
            name = 'Hmax' + str(i + 1)
            HQ_Hmax[i] = float(iniFile.get('HQ_para', name))
    # Q = a*(H+b)^2の適用上限流量を計算 <- Qmax
    HQ_Qmax = 0
    if nHQ > 0:
        HQ_Qmax = np.zeros(nHQ)
        for i in range(nHQ):
            name = 'Qmax' + str(i + 1)
            HQ_Qmax[i] = HQ_a[i] * ( HQ_Hmax[i] + HQ_b[i]) ** 2
    
    #print HQ_a
    #print HQ_b
    #print HQ_Hmax
    #print HQ_Qmax
    return nHQ, HQ_a, HQ_b, HQ_Hmax, HQ_Qmax



# Convert all Q to WL using a HQ eq.
# ary_q is multipule dimension
def ConvQ2H_all_HQeq(ary_q, n_hq, ary_a, ary_b, ary_qmax):
    n_t = ary_q.shape[0]
    n_p = ary_q.shape[1]
    conv_sum = np.zeros([n_t,n_p])
    for i in range(n_hq):
        tmp_sum = ary_q
        if i == 0:
            Q_threshold_l = 0
            Q_threshold_u = ary_qmax[i]
        else:
            Q_threshold_l = ary_qmax[i-1]
            Q_threshold_u = ary_qmax[i]
        tmp_sum = np.where((tmp_sum >= Q_threshold_l) & (tmp_sum <= Q_threshold_u) , np.sqrt(tmp_sum / ary_a[i]) - ary_b[i], 0)
        conv_sum = conv_sum + tmp_sum
    return conv_sum

# ary_q is one dimension
def ConvQ2H_1x1_HQeq(ary_q, n_hq, ary_a, ary_b, ary_qmax):
    n_t = len(ary_q)
    conv_sum = np.zeros([n_t])
    for i in range(n_hq):
        tmp_sum = ary_q
        if i == 0:
            Q_threshold_l = 0
            Q_threshold_u = ary_qmax[i]
        else:
            Q_threshold_l = ary_qmax[i-1]
            Q_threshold_u = ary_qmax[i]
        tmp_sum = np.where((tmp_sum >= Q_threshold_l) & (tmp_sum <= Q_threshold_u) , np.sqrt(tmp_sum / ary_a[i]) - ary_b[i], 0)
        conv_sum = conv_sum + tmp_sum
    return conv_sum

