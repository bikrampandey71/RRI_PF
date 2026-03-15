# -*- coding: utf-8 -*-
import configparser

def ReadiniFile(ini_f):
    iniFile = configparser.ConfigParser()
    # iniFilePath = './../RRI-PFconfig.ini'
    iniFile.read(ini_f)
    # 190214, add by YN: convert the discharge to the water level
    ConvQ2H = int(iniFile.get('Function', 'ConvQ2H'))
    # --- [File] ---
    # 190214, revised by YN: Home directoryを統一
    HomeDir = iniFile.get('File', 'HomeDir')
    # HydroFile  = iniFile.get('File', 'HydroFile')
    PredictionPoint = iniFile.get('File', 'PredictionPoint')
    Hydro_f_type = int(iniFile.get('File', 'Hydro_f_type'))    # 241217 add by YN
    Hydro_f = iniFile.get('File', 'Hydro_f')
    Hydro_locNo = -9999
    if Hydro_f_type == 1:
        Hydro_locNo = int(iniFile.get('File', 'Hydro_locNo'))
    elif Hydro_f_type == 2:
        Hydro_f = Hydro_f.replace('(PredictionPoint)', PredictionPoint)
    CalcDir = iniFile.get('File', 'CalcDir')
    CalcDir = HomeDir + "/" + CalcDir
    InitDir = iniFile.get('File', 'InitDir')
    InitDir = HomeDir + "/" + InitDir
    RsltDir = iniFile.get('File', 'RsltDir')
    RsltDir = HomeDir + "/" + RsltDir
    ArchDir = iniFile.get('File', 'ArchDir')
    ArchDir = HomeDir + "/" + ArchDir
    ObsData = iniFile.get('File', 'ObsData')
    ObsData = HomeDir + "/" + ObsData
    SecData = iniFile.get('File', 'SecData')
    SecData = HomeDir + "/" + SecData
    RainDir = iniFile.get('File', 'RainDir')
    RainDir = HomeDir + "/" + RainDir
    Rain_extraction_No = int(iniFile.get('File', 'Rain_extraction_No'))   # add 2024.12.17 by YN
    BoundHr_func = int(iniFile.get('File', 'BoundHr_func'))
    BoundHr_ini = iniFile.get('File', 'BoundHr_ini')
    BoundHr_ini = HomeDir + "/" + BoundHr_ini
    BoundQr_func = int(iniFile.get('File', 'BoundQr_func'))
    BoundQr_ini = iniFile.get('File', 'BoundQr_ini')
    BoundQr_ini = HomeDir + "/" + BoundQr_ini
    HQData  = iniFile.get('File', 'HQData')
    HQData  = HomeDir + "/" + HQData
    # --- [PF] ---
    ParticleNum = int(iniFile.get('PF', 'ParticleNum'))             # Get Particle Number from iniFile as integer
    ResamplingMethod = int(iniFile.get('PF', 'ResamplingMethod'))
    LikelihoodMethod = int(iniFile.get('PF', 'LikelihoodMethod'))   # LikelihoodRate : past vs now
    StatesNum = int(iniFile.get('PF', 'StatesNum'))                 # Number of the states
    # --- read States ------
    StatesNo = []
    tmp = iniFile.get('PF', 'SelectStates')
    tmp2 = [int(x.strip()) for x in tmp.split(',')]
    StatesNo = [tmp2[iStates] for iStates in range(StatesNum)]
    # --- read States option ------
    StatesOption = []
    tmp = iniFile.get('PF', 'StatesOption')
    tmp2 = [int(x.strip()) for x in tmp.split(',')]
    StatesOption = [tmp2[iStates] for iStates in range(StatesNum)]
    # --- read original particle ------
    OrgParticle = int(iniFile.get('PF', 'OrgParticle'))
    OrgValue = []
    tmp = iniFile.get('PF', 'OrgValue')
    tmp2 = [float(x.strip()) for x in tmp.split(',')]
    for j in range(StatesNum):
        OrgValue.append(tmp2[j])
    # --- read Fixed value ------
    FixedParticleNum = int(iniFile.get('PF', 'FixedParticleNum'))
    FixedValue = []
    for j in range(StatesNum):
        FixedValue_state = []
        for i in range(FixedParticleNum):
            name3 = 'FixedValue' + str(i + 1)
            tmp = iniFile.get('PF', name3)
            tmp2 = [float(x.strip()) for x in tmp.split(',')]
            FixedValue_state.append(tmp2[j])
        FixedValue.append(FixedValue_state)
    # --- read mean value in system noise ------
    Mean_SysNoise = []
    tmp = iniFile.get('PF', 'Mean_SysNoise')
    tmp2 = [float(x.strip()) for x in tmp.split(',')]
    Mean_SysNoise = [tmp2[iStates] for iStates in range(StatesNum)]
    # --- read Standard deviation in system noise ------
    SD_SysNoise = []
    tmp = iniFile.get('PF', 'SD_SysNoise')
    tmp2 = [float(x.strip()) for x in tmp.split(',')]
    SD_SysNoise = [tmp2[iStates] for iStates in range(StatesNum)]
    # --- if SelectStates = 5, read lines of RRI_Input.txt ------
    Line_rri_input = []
    tmp = iniFile.get('PF', 'Line_rri_input')
    tmp2 = [x.strip() for x in tmp.split(',')]
    for iStates in range(StatesNum):
        if tmp2[iStates] == '':
            Line_rri_input.append(-1)
        else:
            Line_rri_input.append(int(tmp2[iStates]))
    #print(Line_rri_input)
    SequentialConditions_RRI  = int(iniFile.get('PF', 'SequentialConditions_RRI'))  # common settings (hs/hr/ga)
    # SelectConditions_hs  = int(iniFile.get('PF', 'SelectConditions_hs'))  # Select the slope conditions in RRI model: hs
    # SelectConditions_hr  = int(iniFile.get('PF', 'SelectConditions_hr'))  # Select the river conditions in RRI model: hr
    # SelectConditions_ga  = int(iniFile.get('PF', 'SelectConditions_ga'))  # Select the river conditions in RRI model: ga
    # SelectConditions_sed = int(iniFile.get('PF', 'SelectConditions_sed'))  # Select the riverbed conditions in Egashira model: sediment
    # LikelihoodCondition  = int(iniFile.get('PF', 'LikelihoodCondition'))  # Fixed to "constant"
    # Min_ErrH = float(iniFile.get('PF', 'Min_ErrH'))
    # Ratio_ErrH = float(iniFile.get('PF', 'Ratio_ErrH'))
    SigmaErr_Const = float(iniFile.get('PF', 'SigmaErr_Const'))
    # Sigma_Corr = float(iniFile.get('PF', 'Sigma_Corr'))
    # --- [Time] ---
    BT_dy = int(iniFile.get('Time', 'BackDays'))
    FT_dy = int(iniFile.get('Time','ForecastDays'))
    ForecastType = iniFile.get('Time', 'ForecastType')
    # CombinedFastAR_Minutes = int(iniFile.get('Time', 'CombinedFastAR_Minutes'))     # Add, 2020/11/17, For FastAR in PRISM2020
    # DelayOption = int(iniFile.get('Time', 'DelayOption'))                           # Add, 2020/11/17, For FastAR in PRISM2020
    # FastMinFlag = int(iniFile.get('Time', 'FastMinFlag'))                           # Add, 2020/12/02, For FastAR in PRISM2020
    RRI_dt_min = int(iniFile.get('Time', 'RRI_dt_min'))
    PF_dt_min = int(iniFile.get('Time', 'PF_dt_min'))
    # --- [Output] ---
    Rslt_Best_all = int(iniFile.get('Output', 'Rslt_Best_all'))
    Rslt_Org_all = int(iniFile.get('Output', 'Rslt_Org_all'))
    Rslt_WtMean_hs = int(iniFile.get('Output', 'Rslt_WtMean_hs'))
    Rslt_WtMean_hr = int(iniFile.get('Output', 'Rslt_WtMean_hr'))
    Rslt_WtMean_ga = int(iniFile.get('Output', 'Rslt_WtMean_ga'))
    Rslt_WtMean_qr = int(iniFile.get('Output', 'Rslt_WtMean_qr'))
    Rslt_OtherSt = int(iniFile.get('Output', 'Rslt_OtherSt'))
    # --- [SimulationDA] ---    
    PF_StartTime = int(iniFile.get('SimulationDA', 'PF_StartTime'))
    PF_EndTime   = int(iniFile.get('SimulationDA', 'PF_EndTime'))
    return ConvQ2H, HomeDir, PredictionPoint, Hydro_f_type, Hydro_f, Hydro_locNo, CalcDir, InitDir, RsltDir, ArchDir, \
           ObsData, HQData, SecData, RainDir, Rain_extraction_No, BoundHr_func, BoundHr_ini, BoundQr_func, BoundQr_ini, \
           ParticleNum, ResamplingMethod, LikelihoodMethod, FixedParticleNum, StatesNum, StatesNo, StatesOption, \
           Line_rri_input, FixedValue, OrgParticle, OrgValue, SequentialConditions_RRI, \
           BT_dy, FT_dy, ForecastType, RRI_dt_min, PF_dt_min, \
           PF_StartTime, PF_EndTime, Mean_SysNoise, SD_SysNoise, SigmaErr_Const, \
           Rslt_Best_all, Rslt_Org_all, Rslt_WtMean_hs, Rslt_WtMean_hr, Rslt_WtMean_ga, Rslt_WtMean_qr, Rslt_OtherSt
    # SelectConditions_hs, SelectConditions_hr, SelectConditions_ga <<< integrated as RRI on 2024.11.13 by YN 

# BoundaryH_condition.iniからNum, Bound_file, i_loc, j_locを呼び出しデータフレームとして返す。
def importBoundaryHCondition(f):
    iniFile = configparser.ConfigParser()
    iniFile.read(f)
    # --- [Point_num] ---
    nBound = int(iniFile.get('Bound_num', 'nBound'))
    BoundFlagH = []
    ZeroTPm = []
    BoundFile = []
    loc_i = []
    loc_j = []
    for iBound in range(nBound):
        # --- [Bound_flag] ---
        name = 'BoundFlag' + str(iBound + 1)
        BoundFlagH.append(int(iniFile.get('Bound_flag', name)))
        # --- [Bound_file] ---
        name = 'BoundFile' + str(iBound + 1)
        BoundFile.append(str(iniFile.get('Bound_file', name)))
        # --- [DatumH] ---
        name = 'ZeroTPm' + str(iBound + 1)
        ZeroTPm.append(float(iniFile.get('DatumH', name)))
        # --- [Point_loc] ---
        name = 'loc_i' + str(iBound + 1)
        loc_i.append(int(iniFile.get('Bound_loc', name)))
        name = 'loc_j' + str(iBound + 1)
        loc_j.append(int(iniFile.get('Bound_loc', name)))
    return nBound, BoundFlagH, BoundFile, ZeroTPm, loc_i, loc_j

# BoundaryQ_condition.iniからNum, Bound_file, i_loc, j_locを呼び出しデータフレームとして返す。
def importBoundaryQCondition(f):
    iniFile = configparser.ConfigParser()
    iniFile.read(f)
    # --- [Point_num] ---
    nBound = int(iniFile.get('Bound_num', 'nBound'))
    BoundFlagQ = []
    BoundDA = []
    BoundFile = []
    loc_i = []
    loc_j = []
    for iBound in range(nBound):
        # --- [Bound_flag] ---
        name = 'BoundFlag' + str(iBound + 1)
        BoundFlagQ.append(int(iniFile.get('Bound_flag', name)))
        # --- [Bound_DA] ---
        name = 'BoundDA' + str(iBound + 1)
        BoundDA.append(int(iniFile.get('Bound_DA', name)))
        # --- [Bound_file] ---
        name = 'BoundFile' + str(iBound + 1)
        BoundFile.append(str(iniFile.get('Bound_file', name)))
        # --- [Point_loc] ---
        name = 'loc_i' + str(iBound + 1)
        loc_i.append(int(iniFile.get('Bound_loc', name)))
        name = 'loc_j' + str(iBound + 1)
        loc_j.append(int(iniFile.get('Bound_loc', name)))
    return nBound, BoundFlagQ, BoundDA, BoundFile, loc_i, loc_j


# from 'SedimentConfig.ini'
def ReadSedimentCondition(f):
    iniFile = configparser.ConfigParser()
    iniFile.read(f)
    # --- [File] ---
    Sec1_file  = iniFile.get('File', 'Sec1_file')
    # --- [RiverConditions] ---
    B1 = float(iniFile.get('RiverConditions', 'B1'))      # B1: width in the upper cross section
    B2 = float(iniFile.get('RiverConditions', 'B2'))      # B2: width in the target cross section
    RivGrad_T = float(iniFile.get('RiverConditions', 'RivGrad_T'))   # Grad in the river
    Dist = float(iniFile.get('RiverConditions', 'Dist'))  # Dist: river length betweet 1 and 2
    SedDepth = float(iniFile.get('RiverConditions', 'SedDepth'))  # Initial depth by the riverbed evolution
    Rn = float(iniFile.get('RiverConditions', 'Rn'))  # Roughness in the river at upper cross section
    # --- [CriticalWaterDepth] ---
    TauSm = float(iniFile.get('CriticalWaterDepth', 'TauSm'))  # TauSm
    S = float(iniFile.get('CriticalWaterDepth', 'S'))  # S
    Dm = float(iniFile.get('CriticalWaterDepth', 'Dm'))  # m=60
    Di = float(iniFile.get('CriticalWaterDepth', 'Di'))  # i=90
    Qc = float(iniFile.get('CriticalWaterDepth', 'Qc'))  # Qc: compulsion critical discharge (default Qc = 0.0)
    return Sec1_file, B1, B2, RivGrad_T, Dist, SedDepth, Rn, TauSm, S, Dm, Di, Qc
