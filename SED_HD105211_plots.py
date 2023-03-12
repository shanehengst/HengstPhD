#!/usr/bin/env python3
#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
    Created on Thur Jan 9 2021
    Latest Version: Mon Apr 21 2021
    
    @author: Shane
    
    1-Dimensional model of debris discs
    Grains affected by radiation pressure (using beta = Frad/Fgrav)
    Optical Constants for grain: miepython
    """

#libraries
import miepython as mpy
import re
import pandas as pd
import astropy
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, FixedFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import random
import numpy as np
import math
import subprocess
import os
from os import path
import time
from skimage.transform import rescale
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import CustomKernel
from astropy.convolution import convolve
from astropy.modeling.models import BlackBody as BBody
from astropy import units as u
from PyAstronomy.pyasl import planck
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from scipy.integrate import simps, quad, trapz
from photutils.centroids import fit_2dgaussian
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from scipy import interpolate
from scipy.integrate import quad, simps, trapz
import scipy.stats as st
from scipy.special import logit, expit
from scipy.interpolate import CubicSpline
from csaps import csaps
#import fast_histrogram
from fast_histogram import histogram1d, histogram2d
import emcee
import corner
#import numba
#from numba import njit, jit, prange

#Constants (kg-m-s)
#Universal Contants
G = 6.673705*10**-11        #Gravitational constant
c = 299792458           #Speed of Light
h = 6.62607004*10**-34      #Planck's constant
kb = 1.38064852*10**-23     #boltzman constant

#Sun values (Note in Solar units L=M=R=1)
L_s = 3.845*10**26           #luminosity
M_s = 1.9885*10**30          #mass
R_s = 6.957*10**8             #radius
T_s = 5770                #surface temperature (K)

#Solar System Units
au = 149597870700       #astronomical unit defined by IAU (https://cneos.jpl.nasa.gov/glossary/au.html)
Me = 5.972*10**24           #Earth Mass [kg]
pc = 3.086*10**16           #parsec [m]

#Mathematical constants
pi = 3.1415926535       #pi (10 d.p.)


#initial time
t0 = time.time()

#Format tick labels
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))


#-----------------------------------------------------------------#
#Function: Bolometric Black Body function as a function of wavelength
def Blam(T,w):
  
    #Inputs:
    #T: Temperature (float) [Kelvin]
    #w: Wavelength range (array float) [micron] - to be converted to metres
    L = (2*h*c**2/((w*1e-6)**5))*(np.exp(h*c/((w*1e-6)*kb*T))-1)**-1
    
    #Ouput
    #Luminosity as function of wavelength [SI Units]
    
    return L

#Object name
object = 'HD105211'

Ts = 7244       #star temperature [K]
wr = np.geomspace(0.1,3000,3000) # Wavelength space in microns

f_spek = np.loadtxt('HD105211_spek.txt', dtype=[('wave', float), ('col',float),('col2',float),('fspek', float)])

wave = f_spek["wave"]
fspek = f_spek["fspek"]

fun_spek = interpolate.interp1d(np.log10(wave),np.log10(fspek),kind = 'linear', fill_value = 'extrapolate')
f_10spek = fun_spek(np.log10(wr))
flux_sa = 10**(f_10spek)

#directories
direc = os.getcwd()
direcmain = '/SEDs_'+object
main_direc = direc + direcmain
subprocess.run(['mkdir',main_direc])

#-----------------------------------------------------------------#
#-----------------------------------------------------------------#
##emcee functions###
##Inlike() function for emcee
#def lnlike(phot_wav,phot_flux,phot_unc,func_wav,func_flux):
def lnlike(theta,phot_wav,phot_flux,phot_unc):
    Amp,Temp = theta
    func_flux = Amp*Blam(Temp,wr)*((wr*10**-6)**2)/(c)
    func_wf = interpolate.interp1d(wr,func_flux)
    flam = func_wf(phot_wav)
#    flam = Amp*Blam(Temp,phot_wav)
#    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2
    chi2 = -0.5*c2
    print(chi2)
    return chi2

def lnprior(theta):
    
    Amp,Temp = theta
    if 1e10 < Amp < 1e15 and 40 < Temp < 200: #noting blowout for dirty ice is 2.69 micron for HD 105211 / astrosilicate: blowout 1.601 0.044 -> 0.711
        return 0.0
 
    else:
        return -np.inf


def lnprob(theta,wav,flx,unc):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,wav,flx,unc)

#-----------------------------------------------------------------#
#-----------------------------------------------------------------#

##HD 105211 sorted into wavelength regimes
##For plotting puprposes
onr_lam = [0.349,0.411,0.440,0.466,0.546,0.55,0.64,0.79,1.26,1.6,2.2200]
onr_flx = [24.02,62.4,69.91,72.7,69.91,79.99,92.2,88.4,59.16,45.85,29.78]
onr_unc = [0.24,0.62,6.45,0.77,6.45,7.38,0.46,0.4,7.65,5.93,1.92]

mir_lam = [3.4, 8.28, 9,12.0,13.0,18.0,22.0,24.0,31.0]
mir_flx = [14.84,2.87,2.27,1.42,1.105,0.69,0.434,0.368,0.228]
mir_unc = [1.34,0.118,0.07,0.18,0.039,0.03,0.007,0.015,0.011]

mir_slam = [27,33,35]
mir_sflx = [0.296,0.222,0.214]
mir_sunc = [0.015,0.022,0.038]

fir_lam = [70.0,100.0,160.0,1338]
fir_flx = [0.733,0.728,0.564,2.447e-3]
fir_unc = [0.063,0.096,0.0955,0.1512e-3]

#spectra
#HD 105211
irs_spec = ascii.read('CASHD10511.txt', delimiter = ',')
irs_spec = np.loadtxt('CASHD10511.txt', dtype=[('wavelength', float), ('flux', float), ('error', float)])

#Make sure we are in the right directory
os.chdir(main_direc)


###Plot beta + gaussian + power: SEDs
plt.clf()
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Flux Density [Jy]')
ax.set_xlim([0.3, 3000])
ax.set_ylim([10**-3, 200])
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.errorbar(onr_lam,onr_flx,yerr=onr_unc,fmt='o',mec='green',mfc='green',ecolor='black',capsize=4.,capthick=1, label = 'Optical/Near-IR ')

figname = 'SED_'+object+'_onr.eps'
plt.savefig(figname)

ax.plot(wr,flux_sa,'k', label = 'Photosphere')
figname = 'SED_'+object+'_onr+SS.eps'
plt.savefig(figname)

ax.errorbar(irs_spec["wavelength"].data,irs_spec["flux"].data,yerr=irs_spec["error"].data,fmt='.',color='blue',ecolor='blue', label = 'IRS Spectrum')

figname = 'SED_'+object+'_onr+SS+irs.eps'
plt.savefig(figname)

ax.errorbar(mir_lam,mir_flx,yerr=mir_unc,fmt='o',mec='skyblue',mfc='skyblue',ecolor='black',capsize=4.,capthick=1, label = 'Mid-IR')
ax.errorbar(mir_slam,mir_sflx,yerr=mir_sunc,fmt='o',mec='cyan',mfc='cyan',ecolor='black',capsize=4.,capthick=1, label = 'Synth-Phot')

figname = 'SED_'+object+'_onr+SS+irs+mir.eps'
plt.savefig(figname)
ax.errorbar(fir_lam,fir_flx,yerr=fir_unc,fmt='o',mec='red',mfc='red',ecolor='black',capsize=4.,capthick=1, label = 'Far-IR/Sub-mm')

figname = 'SED_'+object+'_onr+SS+irs+mir+fir.eps'
plt.savefig(figname)


ax.legend(loc = 'upper right')
figname = 'SED_'+object+'_final-withlegend.eps'
plt.savefig(figname)

#stellar subtraction
lam = [70.0,100.0,160.0]
flx = [0.733,0.728,0.564]
unc = [0.063,0.096,0.0955]

#unc = 3*unc

f_ss = fun_spek(np.log10(lam))
flux_s = 10**(f_ss)
flux_ss = flx - flux_s

ax.errorbar(lam,flux_ss,yerr=unc,fmt='o',mec='red',mfc='white',ecolor='black',capsize=4.,capthick=1, label = 'Stellar Subtraction')
figname = 'SED_'+object+'_SS.eps'
plt.savefig(figname)

##Emcee Inputs##
nwalkers = 30
niter = 200
initial = np.array([1e13,100]) #Variables to be tested: A,T
ndim = len(initial)
p0 = [np.array(initial) + np.array([random.uniform(-1e3,1e3),random.uniform(-10,10)],) for i in range(nwalkers)]

print('Walkers for simulation:')
print(p0)
#print(zed)
unc = np.multiply(unc,3)
data = lam,flux_ss,unc


def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 1000)
    burnin = sampler.get_chain()
    
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state, burnin
    
sampler, pos, prob, state, burnin = main(p0,nwalkers,niter,ndim,lnprob,data)

samples = sampler.flatchain

#sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc = np.median(samples, axis=0)
Amp_mcmc, Temp_mcmc = np.median(samples, axis=0)
Amp_mcmc, Temp_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))

print("Amp_min: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) [-]]".format(Amp_mcmc[0],Amp_mcmc[1], Amp_mcmc[2]))
print("Temp: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) K".format(Temp_mcmc[0],Temp_mcmc[1], Temp_mcmc[2]))

#plot best fit
#plt.clf()
DiscBB = Amp_mcmc[0]*Blam(Temp_mcmc[0],wr)*((wr*10**-6)**2)/(c)
#DiscBB = 1*Blam(60,wr)
#wr = np.geomspace(10,3000)
#DiscBB = 1e9*Blam(90,wr*1e-6)
#plt.plot(wr,DiscBB)
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
#print(zed)

#print(np.max(DiscBB))
#print(zed)
ax.plot(wr,DiscBB,'k:', label = 'Disc BB')
figname = 'SED_'+object+'_DiscFitBB .eps'
plt.savefig(figname)

#Plot Corner
labels = ['Amplitude','Temp [K]']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], title_fmt ='.3f')
cornerfig = object+'_Emcee_corner_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.eps'
fig.savefig(cornerfig)


#Plot chains

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
#s2 = burnin.append(samples)
s2 = np.concatenate((burnin, samples), axis=0)

for i in range(ndim):
    ax = axes[i]
    ax.plot(s2[:, :, i], "k", alpha=0.3)
    ax.plot(burnin[:, :, i], "r", alpha=0.3)  #plot burn-in independently
    ax.set_xlim(0, len(burnin) + len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

#b_time = np.ones(len(burnin))
#ax.axvline('k-.')#, label = '')

chains = object+'_Emcee_chains_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.pdf'
fig.savefig(chains)



t1 = time.time()
t = round((t1 - t0)/60,4)
print(f'Total Time: {t} minutes')



