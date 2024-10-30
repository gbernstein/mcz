#!/usr/bin/env python
import numpy as np
import mcz
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import h5py
from scipy.interpolate import interp1d
import os
import sys
import argparse

useMPI = True
try:
    from mpi4py import MPI
except:
    useMPI = False

from importlib.resources import files

pkg = files("mcz")
bossFile = pkg / "data" / "Metadetect_BOSS_WZ_15oct.pickle"
rmFile = pkg / "data" / "Metadetect_RM_WZ_15oct.pickle"
qsoFile = pkg / "data" / "Metadetect_QSO_WZ_15oct.pickle"
wdmFile = pkg / "data" / "ccl_wdm.npz"

def integrals(kernels, wzdata, wdmFile=wdmFile):
    '''Compute the necessary cosmological factors for WZ
    using the fiducial distance/power estimates in wdmFile.
    The code will detect whether (like RedMagic) the WZ reference
    bins are actually distributed over a set of kernel elements.
    Arguments:
    `kernels` is an array of kernels for the unknowns' n(z).
    `wzdata`  is either a single dict-like of WZ data specifications,
              or a list of several to be done consecutively.
    `wdmFile` holds the precomputed cosmological functions of z.
    Returns:
    None - each dictionary is updated with `A_mr, Mu_mr, Mr_mr` entries.'''
    
    # Read fiducial cosmology integrals                                                                                        
    tmp = np.load(wdmFile.open('rb'))

    wdm = interp1d(tmp['z'], tmp['wdm'], kind='cubic',bounds_error=False, fill_value=(tmp['wdm'][0],tmp['wdm'][-1]))
    c_over_H = interp1d(tmp['z'], tmp['c_over_H'], kind='cubic',bounds_error=False,
                            fill_value=(tmp['c_over_H'][0],tmp['c_over_H'][-1]))
    x = np.concatenate( (np.array([0.,]),tmp['z']))
    y = np.concatenate( (np.array([0.,]),tmp['cda']))
    cda = interp1d(x,y, kind='cubic',bounds_error=False, fill_value=(y[0],y[-1]))
    
    if type(wzdata) is not list:
        wzdata = [wzdata]

    # Now do calculations for each wz dict:
    for wz in wzdata:
        ## Do the cosmological integrals for bins                                                                              
        if 'dndz' in wz:
            # This set of reference bins is mapped into rectangular bins
            # Now assign nominal z's to the RM bins:  
            dndz = wz['dndz']
            zr = wz['z_dndz']
            zr0 = zr[0]
            dzr = zr[1]-zr[0]
            Nr = len(zr)    # This is number of the rectangular bins
            A, Mu, Mr = mcz.prep_wz_integrals(kernels,
                                              zr0, dzr, Nr,
                                              cda, c_over_H, wdm)
            # Then average each ref bin over its n(z) of rectangular bins
            wz['A_mr'] = jnp.einsum('mr,sr->ms',A, dndz)
            wz['Mu_mr'] = jnp.einsum('mr,sr->ms',Mu, dndz)
            wz['Mr_mr'] = jnp.einsum('mr,sr->ms',Mr, dndz)
            
        else:
            # Reference bins are in discrete redshift bins
            zr = wz['z_r']
            zr0 = zr[0]
            dzr = zr[1]-zr[0]
            Nr = len(zr)
            A, Mu, Mr = mcz.prep_wz_integrals(kernels,
                                              zr0, dzr, Nr,
                                              cda, c_over_H, wdm)
            wz['A_mr'] = A
            wz['Mu_mr'] = Mu
            wz['Mr_mr'] = Mr
    return

def _opt_blockR(f_um, wzdata, feedback=0.8, iterations=10,
                b_u = jnp.array([0.73,1.06,1.17,0.76])):
    '''Do fixed number of iterations of very dumb Newton iteration
    on the b_u values, then return logp and b_u values.
    Version for block-R covW matrix.'''

    db = 0.02  # Increment for numerical Hessian
    db_max = 0.3 # Biggest step allowed for any iteration of b_u
    n = b_u.shape[0]

    for i in range(iterations):
        logp, dbu = mcz.logpwz_blockR(f_um, b_u, return_dbu=True, **wzdata)
        # Build a Hessian crudely:
        H = jnp.zeros( (n,n), dtype=float)
        for j in range(n):
            b = b_u.at[j].add(db)
            dp = mcz.logpwz_blockR(f_um, b,return_dbu=True, **wzdata)[1]
            b = b_u.at[j].add(db)
            H = H.at[j].set((dp - dbu)/db)
        evals, evecs = jnp.linalg.eigh(H)
        # Only shift in eigen-directions that are concave in logp
        inv_evals = jnp.where(evals<0, 1/evals, 0)
        shift = -jnp.einsum('ij,j,kj,k->i',evecs,inv_evals,evecs,dbu)
        factor = jnp.max(jnp.abs(shift/db_max))
        factor = jnp.minimum(1/factor, feedback)
        b_u = b_u + factor*shift
    logp_final = mcz.logpwz_blockR(f_um, b_u,**wzdata)[0]
    return logp_final, logp_final-logp, b_u


def _opt_dense(f_um, wzdata, feedback=0.8, iterations=10,
          b_u = jnp.array([0.73,1.06,1.17,0.76])):
    '''Do fixed number of iterations of very dumb Newton iteration
    on the b_u values, then return logp and b_u values.
    Version for dense covW matrix.'''
    db = 0.02  # Increment for numerical Hessian
    db_max = 0.3 # Biggest step allowed for any iteration of b_u
    n = b_u.shape[0]

    for i in range(iterations):
        logp, dbu = mcz.logpwz_dense(f_um, b_u, return_dbu=True, **wzdata)
        # Build a Hessian crudely:
        H = jnp.zeros( (n,n), dtype=float)
        for j in range(n):
            b = b_u.at[j].add(db)
            dp = mcz.logpwz_dense(f_um, b,return_dbu=True, **wzdata)[1]
            b = b_u.at[j].add(db)
            H = H.at[j].set((dp - dbu)/db)
        evals, evecs = jnp.linalg.eigh(H)
        # Only shift in eigen-directions that are concave in logp
        inv_evals = jnp.where(evals<0, 1/evals, 0)
        shift = -jnp.einsum('ij,j,kj,k->i',evecs,inv_evals,evecs,dbu)
        factor = jnp.max(jnp.abs(shift/db_max))
        factor = jnp.minimum(1/factor, feedback)
        b_u = b_u + factor*shift
        logp_final = mcz.logpwz_dense(f_um, b_u,**wzdata)[0]
    return logp_final, logp_final-logp, b_u

def run(startk, nk,
        boyanFile = 'boyan_100M_Oct20.h5',
        useRM=True,
        chunk=1000,
        outFile = None):
    '''Calculate log(p) of WZ measurements for each of samples in
    Boyan's 3sDir sample file.  Rows of Boyan's table to use are
    specified in `startk,nk`.  `useRM` determines whether to add
    RedMagic data to BOSS+QSO data.
    Either writes to an output file given by `outFile`, or
    returns arrays of logp per sample, dlogp of last step, and b_u that optimize logp.'''
    
    # Split up the job if we have multiple tasks running under MPI
    rank = 0
    if useMPI:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        endk = startk + nk
        nk = (nk-1)//size + 1
        startk = startk + rank*nk
        nk = min(nk, endk-startk)
        print('Rank',rank,'starting at',startk,'doing',nk)

    if useRM:
        opt_bu = jax.jit(jax.vmap(_opt_dense, in_axes=(0,None), out_axes=0))
    else:
        opt_bu = jax.jit(jax.vmap(_opt_blockR, in_axes=(0,None), out_axes=0))
    # Read Boyan's files                                                                                                       
    pz = h5py.File(boyanFile)
    pzsamp = np.stack( [pz['bin{:d}'.format(i)][startk*1000:(startk+nk)*1000,:] for i in range(4)], axis = 1)
    print('pzsamp shape', pzsamp.shape)

    # Make triangular kernel set                                                                                               
    zzz = np.array(pz['zbins'])
    pzK = mcz.Tz(zzz[1]-zzz[0], len(zzz)-1, startz=zzz[0])
    # Free memory and close HDF5 file
    del pz

    # Open the WZ data files for BOSS and QSO
    b = {k:jnp.array(v) for k,v in np.load('boss_18sep.npz').items()}
    q = {k:jnp.array(v) for k,v in np.load('qso_18sep.npz').items()}
    integrals(pzK, [b,q])

    # Combine info from all spectro
    wz = mcz.concatenate_surveys(b,q)
    # Add RedMagic if desired:
    if useRM:
        r = {k:jnp.array(v) for k,v in np.load('rm_18sep.npz').items()}
        # Compute cosmological integrals
        integrals(pzK, r)
        wz = mcz.concatenate_surveys(wz,r)

    out = []
    for start in range(0,pzsamp.shape[0],chunk):
        print('Start',start)
        out.append(opt_bu(pzsamp[start:start+chunk], wz))
    logp = np.concatenate([o[0] for o in out])
    dlogp = np.concatenate([o[1] for o in out], axis=0)
    b_u = np.concatenate([o[2] for o in out], axis=0)
    if outFile is None:
        # Return results
        return logp, dlogp, b_u
    else:
        # Save data to a file
        np.savez(outFile + '_{:03d}_{:03d}'.format(startk, nk), logp=logp, b_u=b_u, dlogp=dlogp)

def go():
    # Collect arguments for function from command line

    parser = argparse.ArgumentParser(description='''Assign b_u-optimized WZ probabilities to 3sDir samples''')
    parser.add_argument('startk', help='First sample to use (in thousands)', type=int, default=0)
    parser.add_argument('nk', help='Number of samples to process (in thousands)', type=int, default=10)
    parser.add_argument('--useRM', help='Include RedMagic WZ data or just BOSS+QSO?', action='store_true')
    parser.add_argument('-c','--chunk', help='Samples per dispatch to GPU', type=int,default=1000)
    parser.add_argument('-o','--out', help='Output npz file prefix', type=str, default='boyan_wz')
    args = parser.parse_args()
    print(args)

    print('Doing',args.startk, args.nk)
    run(args.startk, args.nk, useRM=args.useRM, chunk=args.chunk, outFile=args.out)

    sys.exit(0)

if __name__=='__main__':
    go()
