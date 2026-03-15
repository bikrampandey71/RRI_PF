# -*- coding: utf-8 -*-
import configparser
import datetime
import sys
import subprocess
import main

def get_time(config_f):
    iniFile = configparser.ConfigParser()
    iniFile.read(config_f)
    # --- [File] ---
    HomeDir = iniFile.get('File', 'HomeDir')
    # --- [Time] ---
    BT_dy = int(iniFile.get('Time', 'BackDays'))
    FT_dy = int(iniFile.get('Time','ForecastDays'))
    RRI_dt_min = int(iniFile.get('Time', 'RRI_dt_min'))
    PF_dt_min = int(iniFile.get('Time', 'PF_dt_min'))
    # --- [SimulationDA] ---    
    PF_StartTime = int(iniFile.get('SimulationDA', 'PF_StartTime'))
    PF_EndTime   = int(iniFile.get('SimulationDA', 'PF_EndTime'))
    return HomeDir, BT_dy, FT_dy, RRI_dt_min, PF_dt_min, PF_StartTime, PF_EndTime


if __name__ == '__main__':
    # Read settings ----------
    Config_PF_f = './../RRI-PFconfig.ini'
    HomeDir, BT_dy, FT_dy, RRI_dt_min, PF_dt_min, PF_StartTime, PF_EndTime \
    = get_time(Config_PF_f)

    # End of condition ----------
    EndTime = datetime.datetime.strptime(str(PF_EndTime), '%Y%m%d%H%M')

    # Call calc weight mean ----------???
    # command = ["python","???.py"]
    # proc = subprocess.Popen(command)
    # proc.communicate()

    # Call main.py ----------
    while True:
        with open(HomeDir + '/datetime.txt', 'r') as f:
            PresentTimeTxt = str(f.readline())
        PresentTime = datetime.datetime.strptime(PresentTimeTxt, '%Y%m%d%H%M')
        print('Computation for PF-RRI... >>> Current time: ' + str(PresentTime))

        command = ["python","main.py"]
        proc = subprocess.Popen(command)
        proc.communicate()
        NextTime = PresentTime + datetime.timedelta(minutes = PF_dt_min)
        print('\n\nProcessing...\n      >>> Continue to the next step: ' + str(NextTime))
        if PresentTime > EndTime:
            print('+--------------------------------------------------+\n\n\n')
            print('     *** Process Completed Successfully!!! ***\n\n\n')
            print('+--------------------------------------------------+')
            sys.exit()


    
