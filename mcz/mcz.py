# Some implementations in JAX for the PZWZ likelihood
import numpy as np
import jax
import jax.numpy as jnp


class Qlna:
    def __init__(self, dlna, nz):
        '''Class representing a series of kernels K_i for dn/dz that
        are stepwise quadratic functions Q(lna) for lna = log(1+z) and have continuous first
        derivatives and integrate to unity over dlna.
        Since dlna = dz / (1+z) it will be true that if
        dn/dlna = Q   then   dn/dz = (dlna/dz) dn/dlna = (1+z) Q(log(1+z)).
        Arguments:
        `dlna` is the step between nodes in lna space
        `nz`   is the number of kernels.'''
        self.dlna = dlna
        self.nz = nz

    def __call__(self,k,z):
        '''Evaluate dn/dz for the kernel with index k at an array of
        z values.'''
        # Doing duplicative calculations to make this compatible
        # with JAX arithmetic.
        lna = jnp.log(1+z)
        index = jnp.floor(lna/self.dlna).astype(int)
        phase = lna/self.dlna - index
        kvals = jnp.stack( (0.5*phase*phase,
                            0.75 - (phase-0.5)*(phase-0.5),
                            0.5*(1-phase)*(1-phase)), axis=-1)
        mask = jnp.stack( (index==k, index==k+1, index==k+2), axis=-1)
        return jnp.sum( kvals * mask, axis=-1) / (1+z)
        

    def kvals(self, z):
        '''Evaluate the kernels' dn/dz at an array of z values.
        Returns two JAX arrays (on default device):
        * The first is an integer array giving the index of the
        kernel function for which this z is in its lower-z piece.
        * The second has shape (3, nz) with each row giving the
        evaluation of dn/dz contributed by the kernel in which
        this is the first, second, and third part of the piecewise
        function.'''
        lna = jnp.log(1+z)
        index = jnp.floor(lna/self.dlna).astype(int)
        phase = lna/self.dlna - index
        kvals = jnp.stack( (0.5*phase*phase,
                            0.75 - (phase-0.5)*(phase-0.5),
                            0.5*(1-phase)*(1-phase)), axis=-1)
        # zero any coefficients that will reach beyond nz kernels
        # And add the factor of (1+z) that makes it a dn/dz 
        mask1 = jnp.stack( (index>=0, index-1>=0, index-2>=0), axis=-1 )
        mask2 = jnp.stack( (index<self.nz, index-1<self.nz, index-2<self.nz), axis=-1)
        kvals = kvals * mask1 * mask2 / (1+z)[:,jnp.newaxis]

        return index, kvals

    def dot(self, a, index, kval, zaxis=0):
        '''Evaluate the function implied by coefficient array a where
        `index` and `kval` have been evaluated at the desired z values.
        The input `a` array has dimension `self.nz` on the specifiec
        `zaxis` and can have any other dimensions.  This axis will be replaced
        with an axis of length len(index).'''
        a = jnp.moveaxis(a, zaxis, -1)
        # Make a lookup of which element of a is needed for each poly piece
        ii = jnp.stack( [index, index-1, index-2], axis=-1)
        ii = jnp.clip(ii, 0, self.nz-1)
        # kval will brodcast over any extra dimensions of a:
        out = jnp.sum(a[...,ii] * kval, axis=-1)
        out = jnp.moveaxis(out, -1, zaxis)
        return out
    
def logpdeep(M_c, a, Delta_d, A_d):
    '''Return the log-likelihood of the deep counts M_c per cell'''
    nc = jnp.einsum('zcB,z->c',a,(1+Delta_d))
    out = -A_d*jnp.sum(nc) + jnp.einsum('c,c->',M_c, jnp.log(nc))
    return out

def logpspec(psz, psB, cs, a, Delta_s):
    '''Return log-likelihood of the spectroscopic data, where
    `psz` is array of shape (nspec, nz) giving n(z) of each spectro
          galaxy as sum over the nz redshift kernels.
          galaxy being in z bin
    `psB` is array of shape (nspec, nB) giving prob of each spectro
          galaxy being observed in bin B (including compost bin)
    `cs`  are the deep cells for each spec galaxy
    `a`   are teh a_zcB coefficients
    `Delta_s` are the spectro-field overdensities'''
    ps = jnp.einsum('sz,zB,z,sb->s',psz, a[:,cs,:],1+Delta_s,pzB) 
    ps = ps / jnp.einsum('zB,z,sb->s',a[:,cs,:],1+Delta_s,pzB)
    return jnp.sum(jnp.log(ps))

def wzhat(a, W, sys, alpha_r, alpha_Bz, b_r, b_Bz, D_zr, D_rz):
    '''Return predicted w_Br values for pairs of source bin and reference bin.
    `a`   are the a_zcB coefficients
    `W`   are the reference clustering terms at z,r
    `sys` are the systematic functions of z,B
    `alpha` are the magnification values of reference and sources
    `b`   are teh bias values of ref and source pops
    `D_zr, D_rz` are requisite magnification distance factors. They are
          transposes in the sense that the are the same function evaluated
          with redshifts of z bins and reference bins swapped, but since
          these are sampled at different z's, the matrices are not transposes
          of each other.
    Returns an array of shape (nB, nr)'''
    out = jnp.einsum('zcB,zr,zB->Br',a,W,sys) \
          + jnp.einsum('zcB,r,Bz,zr->Br',a,b_r, alpha_Bz, D_zr) \
          + jnp.einsum('zcB,Bz,r,rz->Br',a,b_Bz, alpha_r, D_rz)
    return out / jnp.sum(a, axis=1)  # Sum over c axis

def logpw(wdata, what, invcovw):
    '''Return log of (Gaussian) likelihood of observed w given theory and
    its covariance'''
    dw = wdata - what
    return -0.5 * jnp.einsum('Br,BrBr,Br->',dw,invcovw,dw)



                               
        
