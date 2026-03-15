# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import pandas as pd
import ReadRRI_input
import re

# --- Read RRI_input.txt ---
def Read_RRI_input(org):
    rain_f = org[3-1].strip()
    if left(rain_f, 1) == '.':
        rain_f = rain_f[1:]
    dem = org[4-1].strip()
    if left(dem, 1) == '.':
        dem = dem[1:]
    acc = org[5-1].strip()
    if left(acc, 1) == '.':
        acc = acc[1:]
    dir = org[6-1].strip()
    if left(dir, 1) == '.':
        dir = dir[1:]
    rain_xll = org[14-1].split()
    rain_xll = float(rain_xll[0].replace('d', 'e'))
    rain_yll = org[15-1].split()
    rain_yll = float(rain_yll[0].replace('d', 'e'))
    ns_riv = org[18-1].split()
    ns_riv = float(ns_riv[0].replace('d', 'e'))
    nLU = org[19-1].split()
    nLU = int(nLU[0])
    RivThresh = org[38-1].split()
    RivThresh = float(RivThresh[0].replace('d', 'e'))
    Cw = org[39-1].split()
    Cw = float(Cw[0].replace('d', 'e'))
    Sw = org[40-1].split()
    Sw = float(Sw[0].replace('d', 'e'))
    Cd = org[41-1].split()
    Cd = float(Cd[0].replace('d', 'e'))
    Sd = org[42-1].split()
    Sd = float(Sd[0].replace('d', 'e'))
    HeightPara = org[43-1].split()
    HeightPara = float(HeightPara[0].replace('d', 'e'))
    Height_lmt = org[44-1].split()
    Height_lmt = float(Height_lmt[0].replace('d', 'e'))
    RivFlg = org[46-1].split()
    RivFlg = int(RivFlg[0])
    Width = org[47-1].strip()
    if left(Width, 1) == '.':
        Width = Width[1:]
    Depth = org[48-1].strip()
    if left(Depth, 1) == '.':
        Depth = Depth[1:]
    Height = org[49-1].strip()
    if left(Height, 1) == '.':
        Height = Height[1:]
    InitLine = org[51-1].strip()
    InitFlg = InitLine.split()
    hs_init_flg = 0
    hr_init_flg = 0
    hg_init_flg = 0
    ga_init_flg = 0
    if InitFlg[0] == '1': hs_init_flg = 1
    if InitFlg[1] == '1': hr_init_flg = 1
    if InitFlg[2] == '1': hg_init_flg = 1
    if InitFlg[3] == '1': ga_init_flg = 1
    LU = org[66-1].strip()
    if left(LU, 1) == '.':
        LU = LU[1:]
    Loc = org[100-1].strip()
    if left(Loc, 1) == '.':
        Loc = Loc[1:]
    return rain_f, dem, acc, dir, rain_xll, rain_yll, \
           ns_riv, nLU, RivThresh, Cw, Sw, Cd, Sd, \
           HeightPara, Height_lmt, RivFlg, Width, Depth, Height, \
           hs_init_flg, hr_init_flg, hg_init_flg, ga_init_flg, LU, Loc

def Read_map(file, loc_i, loc_j):
    f = open(file, 'r')
    Map_org = f.readlines()
    f.close()
    i = 0
    for Line_org in Map_org:
        i += 1  # loc_i direction
        Line_org = Line_org.strip()
        Value_org = re.split(r'\s+', Line_org)
        #print Value_org
        nClm = len(Value_org)
        if i > 6:
            for j in range(nClm - 1): # loc_j direction
                loc_Map[i-7, j] = float(Value_org[j])
        elif i == 1:
            ncols = int(Value_org[1])
        elif i == 2:
            nrows = int(Value_org[1])
            loc_Map = np.zeros([nrows, ncols], dtype = float)
    loc_Val = loc_Map[loc_i-1, loc_j-1]
    #print loc_Val
    return loc_Val

def Read_index(file):
    f = open(file, 'r')
    Map_org = f.readlines()
    f.close()
    i = 0
    for Line_org in Map_org:
        i += 1  # loc_i
        Line_org = Line_org.strip()
        Value_org = re.split(r'\s+', Line_org)
        nClm = len(Value_org)  # loc_j
        if i == 1:
            ncols = int(Value_org[1])
        elif i == 2:
            nrows = int(Value_org[1])
            loc_Map = np.zeros([nrows, ncols], dtype = float)
        elif i == 3:
            xllcorner = float(Value_org[1])
        elif i == 4:
            yllcorner = float(Value_org[1])
        elif i == 5:
            cellsize = float(Value_org[1])
        elif i == 6:
            NODATA_value = int(Value_org[1])
    return ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value

def left(text, n):
    return text[:n]

def right(text, n):
    return text[-n:]

def mid(text, n, m):
    return text[n-1:n+m-1]

# (2020) extract output point for RRI ---------
def Read_location(file):
    f = open(file, 'r')
    Loc_org = f.readlines()
    f.close()
    Loc_Name = []
    Loc_i = []
    Loc_j = []
    i = 0
    for Loc_iLine in Loc_org:
        i += 1
        Loc_iLine = Loc_iLine.strip()
        Loc_Value = re.split(r'\s+', Loc_iLine)
        #print Loc_Value
        Loc_Name.append(Loc_Value[0])
        Loc_i.append(int(Loc_Value[1]))
        Loc_j.append(int(Loc_Value[2]))
    print('   >>> Output point is ' + str(len(Loc_org)) + ': ' + str(Loc_Name))
    #print Loc_i
    #print Loc_j
    return Loc_Name, Loc_i, Loc_j, i
