from typing import Tuple, Union, List, Dict, Any

import numpy as np
import torch
import os
import csv
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy

from utils import generate_dir_name

from multiprocessing import Pool

class GridScorer(object):
    def __init__(self, res:int=50):
        self.res = res
        self.disc = self._get_disc(xlims=(-(res-1),res-1),ylims=(-(res-1),res-1),res=2*res-1)
        self.angl = self._get_angl(xlims=(-(res-1),res-1),ylims=(-(res-1),res-1),res=2*res-1)

    def _get_disc(self,
                  xlims:Tuple[float,float],
                  ylims:Tuple[float,float],
                  res:int,
                  center:Tuple[float,float]=[0.0,0.0]):
        """Calculates the distance from the center"""
        X, Y = np.meshgrid(np.linspace(xlims[0],xlims[1],res), np.linspace(ylims[0],ylims[1],res))
        X = X - center[0]
        Y = Y - center[1]
        return np.sqrt(X**2 + Y**2)
    
    def _get_angl(self,
                  xlims:Tuple[float,float],
                  ylims:Tuple[float,float],
                  res:int,
                  center:Tuple[float,float]=[0.0,0.0]):
        """Calculates the angle in degrees"""
        X, Y = np.meshgrid(np.linspace(xlims[0],xlims[1],res), np.linspace(ylims[0],ylims[1],res))
        X = X - center[0]
        Y = Y - center[1]
        return np.angle(X + Y * 1j, deg=True)
    
    def _get_sac_norm(self,
                      width:int,
                      height:int):
        """Calculates the denominator for sac"""
        X1, Y1 = np.meshgrid(np.arange(width)+1, np.arange(height)+1)
        X2, Y2 = np.meshgrid(np.arange(width)[::-1]+1, np.arange(height)[::-1]+1)
        m11 = X1*Y1
        m21 = X2*Y1
        m12 = X1*Y2
        m22 = X2*Y2
        mat = np.array([m11,m21,m12,m22])
        mat = np.min(mat,axis=0)
        return mat

    def calc_sac(self, x:np.ndarray):
        assert len(x.shape)==2 and x.shape[0]==x.shape[1]
        res = x.shape[0]
        sac = scipy.signal.fftconvolve(x,x[::-1,::-1])
        #sac = scipy.signal.correlate(x,x)
        norm = (x**2).sum()/np.ones_like(x).sum()
        sac_norm = self._get_sac_norm(2*res-1,2*res-1) * norm
        sac = np.divide(sac,sac_norm)
        return sac
    
    def calc_crad(self, sac:np.ndarray):
        assert len(sac.shape)==2 and sac.shape[0]==sac.shape[1]
        assert (sac.shape[0]+1)%2==0
        res = (sac.shape[0]+1)//2

        rbins, step = np.linspace(0,res*0.9*(199/200),200,retstep=True)
        crad = np.zeros(rbins.shape)
        nan_loc = []
        for i, r in enumerate(rbins):
            mask = (self.disc>=r)&(self.disc<r+step)
            if mask.sum()==0:
                crad[i] = np.nan
                nan_loc.append(True)
            else:
                crad[i] = sac[mask].sum()/mask.sum()
                nan_loc.append(False)
        nan_loc = np.array(nan_loc)
        crad = np.interp(x=rbins,xp=rbins[~nan_loc],fp=crad[~nan_loc]) # interpolate nan values
        rbins_new, step_new = np.linspace(0,res*0.9*(1999/2000),2000,retstep=True) # increase resolution by 10 times
        crad = np.interp(rbins_new,rbins,crad)
        crad = scipy.ndimage.gaussian_filter(crad,sigma=5*step/step_new) # smooth by sigma=5 times the original binsize

        r0, r1, r2, message = self._get_peaks(crad)
        r0 = r0 * step_new
        r1 = r1 * step_new
        r2 = r2 * step_new
        return crad, r0, r1, r2, message

    def _get_peaks(self, crad:np.ndarray):
        """Finds the peaks in the crad"""
        peaks = np.argwhere(np.diff((np.diff(crad) > 0).astype(int))==-1)
        trghs = np.argwhere(np.diff((np.diff(crad) > 0).astype(int))==1)
        message = '--'
        if len(peaks)==0:
            r0 = None
            message += 'no peaks found--'
        else:
            r0 = peaks[0][0]
        if len(trghs)<2:
            if len(trghs)==1:
                message += 'one trough found--'
                r1, r2 = None, None
            else:
                message += 'zero troughs found--'
                r1, r2 = None, None
        else:
            r1, r2 = trghs[0][0], trghs[1][0]
        return r0, r1, r2, message

    def calc_cpol(self, sac:np.ndarray, r1:float, r2:float):

        dbins, step = np.linspace(-180,175,72,retstep=True)
        cpol = np.zeros(dbins.shape)
        nan_loc = []
        for i, d in enumerate(dbins):
            mask = (self.disc>=r1)&(self.disc<r2)&(self.angl>=d)&(self.angl<d+step)
            if mask.sum()==0:
                cpol[i] = np.nan
                nan_loc.append(True)
            else:
                cpol[i] = sac[mask].sum()/mask.sum()
                nan_loc.append(False)
        nan_loc = np.array(nan_loc)
        cpol = np.interp(x=dbins,xp=dbins[~nan_loc],fp=cpol[~nan_loc])
        return cpol

    def calc_score(self, x:np.ndarray):
        sac = self.calc_sac(x)
        crad, r0, r1, r2, message = self.calc_crad(sac)
        if message!='--':
            return r0, r1, r2, message, 0, 0, 0, 0
        else:
            cpol = self.calc_cpol(sac,r1,r2)
            ftcpol = np.fft.fft(cpol)
            max_freq = np.argmax(abs(ftcpol)[1:36])+1 # Find frequency with maximum power
            max_phase = np.angle(ftcpol[max_freq],deg=True) # Find the corresponding phase

            score_norm = np.sum(cpol**2) - (cpol.sum()**2)/len(cpol) # Calculate the denominator for the grid score
            score_60 = (2 * (abs(ftcpol[6])**2) / len(cpol))/score_norm
            score_90 = (2 * (abs(ftcpol[4])**2) / len(cpol))/score_norm
            
            return r0, r1, r2, message, max_freq, max_phase, score_60, score_90

    def run(self, options:object, activations:np.ndarray, perturbation:Union[Tuple[float,float],None]=None):
        arg = [(act,options) for act in activations]
        with Pool(processes=64) as p:
            results = p.starmap(self.calc_score,arg)

        r0, r1, r2, message, max_freq, max_phase, score_60, score_90 = zip(*results)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,perturbation)}/grid60.npy',score_60)
        np.save(f'data/{options.run_ID}/{generate_dir_name(options,perturbation)}/grid90.npy',score_90)

        with open(f'data/{options.run_ID}/{generate_dir_name(options,perturbation)}/grid_stats.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['neuron_id','r0','r1','r2','message','max_freq','max_phase','score_60','score_90'])
            for nid, line in enumerate(results):
                writer.writerow([str(nid)] + [str(item) for item in line])