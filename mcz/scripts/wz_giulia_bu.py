#!/usr/bin/env python
# Calculate WZ likelihood for a single Maglim bin
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
                b_u = jnp.array([1.])):
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
          b_u = jnp.array([1.])):
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

def run(startk, nk, bin,
        giuliaFile = 'giulia_3M_Nov15.h5',
        chunk=1000,
        outFile = None):
    '''Calculate log(p) of WZ measurements for each of samples in one bin of
    Giulia's 3sDir sample file.  Rows of Giulia's table to use are
    specified in `startk,nk`. 
    Uses just BOSS data.
    Either writes to an output file given by `outFile`, or
    returns arrays of logp per sample, dlogp of last step, and b_u that optimize logp.'''
    
    bossFile = 'maglim_bin{:d}_21nov.npz'.format(bin)

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

    opt_bu = jax.jit(jax.vmap(_opt_blockR, in_axes=(0,None), out_axes=0))
    # Read the desired bin from Giulia's file
    pz = h5py.File(giuliaFile)
    pzsamp = np.array( [pz['nz_bin{:d}'.format(bin)][startk*1000:(startk+nk)*1000,:]])
    pzsamp = np.swapaxes(pzsamp,0,1)  # Put in order (sample, u, z)

    print('pzsamp shape', pzsamp.shape)

    # Make triangular kernel set                                                                                               
    zzz = np.array(pz['zbins'])
    dz = zzz[1]-zzz[0]
    pzK = mcz.Tz(dz, len(zzz)-1, z0=zzz[0]+dz/2)
    # Free memory and close HDF5 file
    del pz

    # Open the WZ data files for BOSS
    wz = {k:jnp.array(v) for k,v in np.load(bossFile).items()}
    integrals(pzK, wz)

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
        np.savez(outFile + '_{:03d}_{:03d}_bin{:d}'.format(startk, nk,bin), logp=logp, b_u=b_u, dlogp=dlogp)

def go():
    # Collect arguments for function from command line

    parser = argparse.ArgumentParser(description='''Assign b_u-optimized WZ probabilities to 3sDir samples''')
    parser.add_argument('startk', help='First sample to use (in thousands)', type=int, default=0)
    parser.add_argument('nk', help='Number of samples to process (in thousands)', type=int, default=10)
    parser.add_argument('bin', help='Which Maglim bin to do (0-5)', type=int)
    parser.add_argument('-c','--chunk', help='Samples per dispatch to GPU', type=int,default=500)
    parser.add_argument('-o','--out', help='Output npz file prefix', type=str, default='giulia_wz')
    args = parser.parse_args()

    print('Doing',args.startk, args.nk, args.bin)
    run(args.startk, args.nk, args.bin, chunk=args.chunk, outFile=args.out)

    sys.exit(0)

if __name__=='__main__':
    go()
