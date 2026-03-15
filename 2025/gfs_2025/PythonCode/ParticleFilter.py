# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import pandas as pd

class ParticleFilter(object):
    def __init__(self, obs_v, obs_t, calc_v, Pn, Sigma_ErrH, ResamplingMethod, LikelihoodMethod, nFixP, OrgPn, DA_flag):
        self.obs_v = obs_v
        self.obs_t = obs_t
        self.calc_v = calc_v
        self.Pn = Pn
        self.Sigma_ErrH = Sigma_ErrH
        self.ResamplingMethod = ResamplingMethod
        self.LikelihoodMethod = LikelihoodMethod
        self.nFixP = nFixP
        self.OrgPn = OrgPn
        self.DA_flag = DA_flag
        
    def norm_likelihood(self, y, x, Sig):
        return (np.sqrt(2*np.pi))**(-1) * np.exp(-(((y-x)/Sig)**2)/2)  # Evaluate likelihood by present error
    
    # def norm_likelihood2(self, Co, Sig):
    #     return (np.sqrt(2*np.pi))**(-1) * np.exp(-(((1-Co)/Sig)**2)/2)   # Evaluate likelihood by historical error (referd by NILIM)

    def norm_likelihood3(self, RMSE, Sig):
        return (np.sqrt(2*np.pi))**(-1) * np.exp(-((RMSE/Sig)**2)/2)   # Evaluate likelihood by RMSE
        #return (np.sqrt(2*np.pi))**(-1) * np.exp(-((math.log(RMSE)/Sig)**2)/2)


    def resampling_2025(self, weights):
        """
        D'Hondt method
        [Notes] considering Fixed Particle & Org Particle
        args:
          - self: obs_v, obs_t, calc_v, Pn, Sigma_ErrH, ResamplingMethod, LikelihoodMethod, nFixP, OrgPn
          - weights: normalized weight (= w_normed)
        return:
          - k: resampled index of particles
        ［アルゴリズム］固定粒子やオリジナル粒子は尤度(wt)に依らず常に選ばれ、リサンプリング前の尤度(wt)をドント方式の手続きに従い事前に減らしている。
        add by YN on 01.29, 2025
        """
        k = []                  # index of particles
        wt = weights.copy()     # weights: org wt, wt: updated wt
        idx = np.asanyarray(range(self.Pn))
        # Add the count +1.0 >>> update wt following the D'Hondt method (wt/count)
        ParticleSize = np.zeros(self.Pn)
        ParticleSize[self.OrgPn-1] += 1.0
        wt[self.OrgPn-1] = weights[self.OrgPn-1] / (ParticleSize[self.OrgPn-1] + 1.0)
        for iFix in range(self.nFixP):
            ParticleSize[self.Pn - 1 - iFix] += 1.0
            wt[self.Pn - 1 - iFix] = weights[self.Pn - 1 - iFix] / (ParticleSize[self.Pn - 1 - iFix] + 1.0)

        ParticleSize_sum = int(np.sum(ParticleSize))
        for _ in range(self.Pn):
            max_idx = np.argmax(wt)
            if max_idx is None:
                return None
            ParticleSize[max_idx] += 1
            ParticleSize_sum += 1
            if ParticleSize_sum == self.Pn:
                break
            wt[max_idx] = weights[max_idx]/(ParticleSize[max_idx] + 1.0)
        for iPn in range(self.Pn):
            for _ in range(int(ParticleSize[iPn])):
                k.append(idx[iPn])
        # DA switch
        if self.DA_flag == 0:
            print("\n\n\n")
            print("*** (Caution!!!) *****************************")
            print("Data Assimilation: Not activate caused by WL","\n")
            print("Resampling before:\n",np.round(k, 3))
            print("            ↓↓↓")
            k = range((self.Pn))
            print("Resampling after :\n",np.round(k, 3))
            print("***************************** (Caution!!!) ***")
        return k

    def get_filtered_value(self, w_normed, x):
        return np.dot(w_normed, x)
    
    def simulate(self):
        w1          = np.zeros(self.Pn)
        # w2          = np.zeros(self.Pn)
        w3          = np.zeros(self.Pn)
        w_normed    = np.zeros(self.Pn)
        # Correlation = np.zeros(self.Pn)
        RMSE        = np.zeros(self.Pn)

        for iPn in range(self.Pn):
            # obs_rnd = self.obs_v + np.random.rand(len(self.obs_v))/100          # For the calc. of correlation
            # Correlation[iPn] = np.corrcoef(self.calc_v[:,iPn], obs_rnd)[0,1]        # Calculation for correlation
            RMSE[iPn] = np.sqrt(np.mean((self.obs_v - self.calc_v[:,iPn])**2))      # Calculation for RMSE
            w1[iPn] = self.norm_likelihood(self.obs_v[len(self.obs_v)-1], self.calc_v[len(self.obs_v)-1, iPn], self.Sigma_ErrH) # Evaluation by a present error
            # w2[iPn] = self.norm_likelihood2(Correlation[iPn], self.Sigma_Corr)      # Evaluation for a historical error by the correlation
            w3[iPn] = self.norm_likelihood3(RMSE[iPn], self.Sigma_ErrH)             # Evaluation for a historical error by the RMSE

        if self.LikelihoodMethod == 0:                # NowError
            print("Likelihood Method = 0 >>> Only present time")
            w_normed = w1/np.sum(w1)
        elif self.LikelihoodMethod == 3:              # RMSE
            print("Likelihood Method = 3 >>> RMSE")
            w_normed = w3/np.sum(w3)
        elif self.LikelihoodMethod == 4:              # RMSE + NowError
            print("Likelihood Method = 4 >>> RMSE + Present error")
            w_normed = (w3 + w1) / (np.sum(w3 + w1))
        # elif self.LikelihoodMethod == 1:              # Correlation
        #     print("Likelihood Method = 1 >>> Only historical error")
        #     w_normed = w2/np.sum(w2)
        # elif self.LikelihoodMethod == 2:              # NowError + Correlation
        #     print("Likelihood Method = 2 >>> present time and historical error")
        #     w_normed = (w1 * w2) / (np.sum(w1 * w2))     
        #print w_normed

        df = pd.DataFrame(w_normed, columns=(['data']))
        k_No1_v = df['data'].max()
        k_No1 = df[df['data'] == k_No1_v].index[0]
        BestPn = k_No1+1

        # --- Resampling ---
        # --- The D'Hondt method (ResamplingMethod=3) is recommended since the impact of this setting is small.
        if self.ResamplingMethod == 3:    #D'Hondt method
            print("ReasamplingMethod = 3 >>> D'Hondt method")
            k = self.resampling_2025(w_normed) 
        return k, w_normed, BestPn
            # k: index of particle (Not Particle No#), k_No1: index of best particle same as k
            # w_normed: Weights of particle (=normalized likelihood)
