# -*- coding: utf-8 -*

import multiprocessing as multi
from multiprocessing import Pool
#import shutil
#from distutils.dir_util import copy_tree        #上書きコピーできるようにする
import numpy as np
import numpy.random as rd
import os

# --- 並列処理用のラッピング ---
def wrapper(args):
    return CalcParticle(*args)
# ----------------------------

# --- wrapping and prediction ---
def multi_process(EnsList):
    nEns = len(EnsList)
    nCPU = multi.cpu_count()
    if nEns < nCPU:
        po = Pool(nEns)
    else:
        po = Pool(nCPU)
    output = po.map(wrapper, EnsList)
    po.close()  
    return output
# ----------------------------

# --- Function for the calculation by the RRI model ---
def CalcParticle(iPn, CalcDir, Hydro_f_type):
    CdDir = CalcDir + '/' + 'Particle' + str(iPn+1).zfill(5) + '/RRI/'
    #calcRRI ----------
    CalcEXE = '0_rri_1_4_2_6_NonOMP_dt600s.exe'     #Case of simgle particle: 0_rri_1_4_2_6_detail_dt600.exe
    os.chdir(CdDir)
    os.system(CalcEXE)
    # calcHydro ----------
    if Hydro_f_type == 2:
        CalcHydro = 'calcHydro.exe'
        CdDirHydro = CalcDir + '/' + 'Particle' + str(iPn+1).zfill(5) + '/RRI/etc/calcHydro'
        os.chdir(CdDirHydro)
        os.system(CalcHydro)
    #[default] rainBasin ----------
    # CalcHyeto = 'rainBasin.exe'
    # CdDirHyeto = CalcDir + '/' + 'Particle' + str(iPn+1).zfill(5) + '/RRI/etc/rainBasin'
    # os.chdir(CdDirHyeto)
    # os.system(CalcHyeto)
    #[advance]rainBasin_extraction for MCC ----------
    CalcHyeto = 'rainBasin_extraction.exe'
    CdDirHyeto = CalcDir + '/' + 'Particle' + str(iPn+1).zfill(5) + '/RRI/etc/rainBasin_extraction'
    os.chdir(CdDirHyeto)
    os.system(CalcHyeto)
    #Result = os.system('0_rri_1_4_2.exe')
    #os.system('RRI_ALL.bat')
# ----------------------
