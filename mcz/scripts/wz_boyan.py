#!/usr/bin/env python
import numpy as np
import pickle as pkl
import mcz
import os
import jax
import jax.numpy as jnp
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys
import argparse
from importlib.resources import files

pkg = files("mcz")
bossFile = pkg / "data" / "Metadetect_BOSS_WZ_28august.pickle"
rmFile = pkg / "data" / "Metadetect_RM_WZ_28august.pickle"
wdmFile = pkg / "data" / "ccl_wdm.npz"

def run(startk, nk,
        boyanFile = 'combined_nz_samples_y6_RU_ZPU_LHC_fullZ_1e4_sum1_stdRUmethod_Aug26.h5'):

    # Read Boyan's files
    pz = h5py.File(boyanFile)
    pzsamp = jnp.stack( [jnp.array(pz['bin{:d}'.format(i)]) for i in range(4)], axis = 0)
    zzz = np.array(pz['zbins'])
    # Make triangular kernel set
    pzK = mcz.Tz(zzz[1]-zzz[0], len(zzz)-1, startz=zzz[0])
    # Free memory
    del pz
    pzsamp = pzsamp[:,startk*1000:(startk+nk)*1000,:]  # Just for now

    # Read fiducial cosmology integrals

    tmp = np.load(wdmFile.open('rb'))

    wdm = interp1d(tmp['z'], tmp['wdm'], kind='cubic',bounds_error=False, fill_value=(tmp['wdm'][0],tmp['wdm'][-1]))
    c_over_H = interp1d(tmp['z'], tmp['c_over_H'], kind='cubic',bounds_error=False, 
                            fill_value=(tmp['c_over_H'][0],tmp['c_over_H'][-1]))
    x = np.concatenate( (np.array([0.,]),tmp['z']))
    y = np.concatenate( (np.array([0.,]),tmp['cda']))
    cda = interp1d(x,y, kind='cubic',bounds_error=False, fill_value=(y[0],y[-1]))


    #### Read William's WZ data and other parameters

    # Data using BOSS as reference 
    bossdata = pkl.load(bossFile.open('rb'))
    boss = {}

    zr = bossdata['z Boss WZ']
    zr0 = zr[0]
    dzr = zr[1]-zr[0]
    Nr = len(zr)
    boss['z_r'] = np.array(zr)  # Save z_r values for plotting purposes

    # Correlation data:
    boss['w'] = jnp.array(bossdata['w_ur Boss bin1--4'])
    Nu = boss['w'].shape[0]
    print('BOSS Nr:',Nr, 'w shape:',boss['w'].shape)
    cov_w = np.array(bossdata['Full-Cov noisy w_ur BOSS(with Hartlap)'])
    # Reshape the dense covariance matrix
    boss['Sw'] = mcz.prep_cov_w_dense(cov_w,Nu,Nr)
    print('Sw shape:',boss['Sw'].shape)
      
    # Assign a nominal bias to each u
    b_u = jnp.ones( Nu, dtype=float) * 1.1

    # And unkowns' magnification priors
    alpha0, sig = bossdata['alpha_meta [mu,sigma]']
    boss['alpha_u0'] = jnp.ones(Nu, dtype=float) * alpha0
    boss['sigma_alpha_u'] = jnp.ones(Nu, dtype=float) * sig  

    # Collect reference info
    boss['b_r']= jnp.array(bossdata['br Boss'])

    # Read the sigmas on the Sys coefficients. 
    sig = np.array(bossdata['Sys [s_uk]_k'])
    # !!! Set the constant terms' sigma to smaller value since
    # largely degenerate with bias
    sig[:,0] = 0.1
    # Also set sigmas to a positive minimum to avoid possible ugliness
    # with sigma=0
    sig = np.maximum(sig,0.001)
    boss['sigma_s_uk'] = jnp.array(sig)
    Nk = boss['sigma_s_uk'].shape[1]

    boss['Sys_kr'] = mcz.sys_basis(zr0,dzr,Nr,Nk)

    # Make alpha_r a single value
    boss['alpha_r_basis'] = jnp.ones((Nr,1), dtype=float)
    alpha0, sig = bossdata['alpha_boss [mu,sigma]']
    boss['ar0'] = jnp.array([alpha0,])
    boss['sigma_ar'] = jnp.array([sig])

    print('Nk:',Nk,'sys priors:',boss['sigma_s_uk'])

    ## Do the cosmological integrals for its bins
    A, Mu, Mr = mcz.prep_wz_integrals(pzK,
                                      zr0, dzr, Nr,
                                      cda, c_over_H, wdm)
    boss['A_mr'] = A
    boss['Mu_mr'] = Mu
    boss['Mr_mr'] = Mr

    # Now read RM data
    rmdata = pkl.load(rmFile.open('rb'))

    rm = {}
    # Redmagic differs because each reference bin is a sum over other bins, with this 
    # matrix:
    rmnz = jnp.array(rmdata['RM n(z) from BOSS WZ'])
    # summing over these "original" bins:
    rmz = rmdata['z array for RM n(z)']
    rmnz = rmnz / np.sum(rmnz, axis=1)[:,np.newaxis]
    zr0 = rmz[0]
    dzr = rmz[1]-rmz[0]
    Nr = len(rmz)

    rm['z_r'] = np.array(rmz)  # Save z_r values for plotting purposes

    # Do the cosmological integrals in the original bins
    A, Mu, Mr = mcz.prep_wz_integrals(pzK,
                                          zr0, dzr, Nr,
                                          cda, c_over_H, wdm)
    # Then average them over the n(z)'s of the RM bins
    rm['A_mr'] = jnp.einsum('mr,sr->ms',A, rmnz)
    rm['Mu_mr'] = jnp.einsum('mr,sr->ms',Mu, rmnz)
    rm['Mr_mr'] = jnp.einsum('mr,sr->ms',Mr, rmnz)

    # Now assign nominal z's to the RM bins:
    zr = rmdata['z RM WZ']
    zr0 = zr[0]
    dzr = zr[1]-zr[0]
    Nr = len(zr)

    # Correlation data:
    rm['w'] = jnp.array(rmdata['w_ur RM bin1--4'])

    cov_w = np.array(rmdata['Full-Cov w_ur RM(with Hartlap)'])
    rm['Sw'] = mcz.prep_cov_w_dense(cov_w,Nu,Nr)

    # And unkowns' magnification priors
    alpha0, sig = rmdata['alpha_meta [mu,sigma]']
    rm['alpha_u0'] = jnp.ones(Nu, dtype=float) * alpha0
    rm['sigma_alpha_u'] = jnp.ones(Nu, dtype=float) * sig  

    # Collect reference info
    rm['b_r']= jnp.array(rmdata['br '])

    # Read the sigmas on the Sys coefficients. 
    sig = np.array(rmdata['Sys [s_uk]_k'])
    # Set sigmas to a positive minimum to avoid possible ugliness
    # with sigma=0
    sig = np.maximum(sig,0.001)
    rm['sigma_s_uk'] = jnp.array(sig)
    Nk = rm['sigma_s_uk'].shape[1]

    rm['Sys_kr'] = mcz.sys_basis(zr0,dzr,Nr,Nk)

    # Make alpha_r a single value
    rm['alpha_r_basis'] = jnp.ones((Nr,1), dtype=float)
    alpha0, sig = rmdata['alpha_red [mu,sigma]']
    rm['ar0'] = jnp.array([alpha0,])
    rm['sigma_ar'] = jnp.array([sig])

    # Concatenate two surveys into one set of arrays
    allSpec = mcz.concatenate_surveys(boss,rm)

    # Probability maximization over b_u
    logpwz = jax.jit(mcz.logpwz_dense, static_argnames=['return_qmap','return_derivs'])

    def optimize_bu(f_um, survey, b_u=np.ones(Nu) * 1.1):
        '''Find values of b_u that maximize the probability
        of the WZ data under the n(z) given by f_um.
        Arguments:
        `f_um`: Array of shape (Nu,Nm) specifying n(z)
        `survey`: Dictionary of WZ data/priors arguments
        needed by `mcz.logpwz` function
        `b_u`:  Starting values of bin biases.
        Returns:
        `logp`: Log of maximized probability
        `b_u`:  Maximizing bias values '''
    
        def func(b_u):
            '''Function to be minimized - return -log p and
            its gradient w.r.t. b_u'''
            logp, derivs = logpwz(f_um, jnp.array(b_u),
                                      **allSpec,
                                      return_derivs=True)
            return -logp, -derivs[1]

        out = minimize(func, b_u, jac=True, method='BFGS', options={'xrtol':0.01})
        return -out.fun, out.x

    logp = []
    bu = []
    for i in range(10000):
        out = optimize_bu(pzsamp[:,i,:], allSpec)
        logp.append(out[0])
        bu.append(out[1])
        if i%100==0:
            print('done',i)
    return np.array(logp), np.array(bu)

def go():
    # Collect arguments for function from command line

    parser = argparse.ArgumentParser(description='''Assign b_u-optimized WZ probabilities to 3sDir samples''')
    parser.add_argument('startk', help='First sample to use (in thousands)', type=int, default=0)
    parser.add_argument('nk', help='Number of samples to process (in thousands)', type=int, default=10)
    args = parser.parse_args()

    logp, bu = run(args.startk, args.nk)
    # Save data to a file
    np.savez('boyan_{:03d}_{:03d}'.format(args.startk, args.nk), logp=logp, bu=bu)

    sys.exit(0)

if __name__=='__main__':
    go()
