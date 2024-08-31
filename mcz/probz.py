# Some implementations in JAX for the PZWZ likelihood
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre

##################################################################
### Redshift distribution kernels
##################################################################

class Rz:
    def __init__(self, dz, nz):
        '''Class representing rectangular n(z) kernels (bins) in z.
        The n(z) will return 0 if evaluated at lower bound, 1/dz at the upper
        bound, so be careful about integrations.
        Arguments:
        `dz`: the step between bin edges (starting at zero)
        `nz`: the number of kernels.'''
        self.dz = dz
        self.nz = nz

    def __call__(self,k,z):
        '''Evaluate dn/dz for the kernel with index k at an array of
        z values.'''
        # Doing duplicative calculations to make this compatible
        # with JAX arithmetic.
        inbin = jnp.logical_and( z>k*self.dz, z<=(k+1)*self.dz)
        return jnp.where(inbin, 1/self.dz, 0.)
        
    def zbounds(self):
        '''Return lower, upper bounds in z of all the bins in (nz,2) array'''
        zmin = np.arange(self.nz)*self.dz
        zmax = zmin + self.dz
        return np.stack( (zmin, zmax), axis=1)

class Tz:
    def __init__(self, dz, nz, startz=0.):
        '''Class representing triangular n(z) kernels (bins) in z.
        First kernel is centered at startz+dz so it's linear at z=startz.
        Arguments:
        `dz`: the step between bins (starting at zero)
        `nz`: the number of kernels.
        `startz`: Lower z limit of first bin'''
        self.dz = dz
        self.nz = nz
        self.startz = startz

    def __call__(self,k,z):
        '''Evaluate dn/dz for the kernel with index k at an array of
        z values.'''
        # Doing duplicative calculations to make this compatible
        # with JAX arithmetic.
        return jnp.maximum(0., 1 - np.abs((z-self.startz)/self.dz - k - 1)) / self.dz
        
    def zbounds(self):
        '''Return lower, upper bounds in z of all the bins in (nz,2) array'''
        zmin = np.arange(self.nz)*self.dz + self.startz
        zmax = zmin + 2*self.dz 
        return np.stack( (zmin, zmax), axis=1)
    
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
        return jnp.sum( kvals * mask, axis=-1) / (1+z) / self.dlna
        
    def zbounds(self):
        '''Return lower, upper bounds in z of all the bins in (nz,2) array'''
        zmin = np.exp(np.arange(self.nz)*self.dlna) - 1
        zmax = np.exp(np.arange(3,self.nz+3)*self.dlna) - 1
        return np.stack( (zmin, zmax), axis=1)

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
        kvals = kvals * mask1 * mask2 / (1+z)[:,jnp.newaxis] / self.dlna

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
    
##################################################################
### Probabilities for deep, spec counts
##################################################################

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


##################################################################
### Routines for evaluating the w(z) probability.
### Going to aim on doing all the repeated calculations with JAX.
##################################################################

def sys_basis(zr0, dzr, Nr,
              Nk):
    '''Create a matrix of shape (Nk, Nr) giving the value of the Sys(z_r) basis functions
    at each reference redshift r.  
    Arguments:
    `zr0, dzr`:  Central redshift of first reference bin and increment between them
    `Nr`:        Number of reference bins
    `Nk`:        Number of terms (=polynomial order +1) of the systematic error function S_k(r)
    Returns:
    `Sys_kr`:    JAX device array of systematic-adjustment terms.'''

    zr = zr0 + np.arange(Nr)*dzr
    # Rescale redshifts to central 85% of the range [-1,1]
    zmean = 0.5*(zr[-1]+zr[0])
    zspan = Nr*dzr
    u = 0.85 * ( 2*(zr-zmean) / zspan)

    # Scaling coefficient for Legendre polynomials:
    ak = np.sqrt(2*np.arange(Nk)+1) / 0.85

    # Build coeffs from Legendre - skip the constant
    out = np.array( [ak[k] * eval_legendre(k, u) for k in range(Nk)] )

    return jnp.array(out)

def prep_wz_integrals(kernels,
                     zr0, dzr, Nzr,
                     cda, c_over_H, wdm,
                     oversample_r=6,
                     Omega_m=0.26):
    r'''Calculate integrals of dark-matter angular clustering and
    magnification over reference n(z)'s and unknowns' z kernels.
    Reference bins are assumed to be rectangular in z.

    Arguments:
    `kernels`: a class that will evaluate dn/dz of all the redshift
               kernels K(z), having members `nz, zbounds()` and can be called
               with arguments (k,z) to give dn/dz of kernel k at redshift(s) z.
               Such as `Qlna` instance.
    `zr0, dzr, Nzr`:  Center of first reference z bin, width, and number of bins,
               respectively.
    `cda, c_over_H`: Functions of z returning the comoving angular diameter to z, and
               the value c/H(z), in matching units.
    `wdm`:     Function returning the w_dm quantity as a function of an
               array of values of z.
               w_dm = H / (c Da^2) \int d\theta W(\theta) 
                  \sum_\ell [(2\ell+1)/4\pi] P_\delta((ell+0.5)/Da) P_\ell(\theta)
               where Da = comoving angular diameter distance to z, P_\delta is matter
               power spectrum in 3d, and P_\ell are Legendre polynomials.
    `oversample_r`:  Oversampling factor for integrating over the reference bins.
    `Omega_m`: total matter density parameter

    Returns: Three JAX device arrays, each of shape (Nk, Nr), where Nk is number 
    of unknown's kernels and Nr is number of reference bins.
    `A_mr`     is \int dz K_k(z) n_r(z) w_dm(z)
    `Mu_mr`    is coefficient for magnification of unknowns by references,
               3 \Omega_m  H0/c \int dz_r n_r(z_r) w_dm(z_r) x (H0/H(z_r)) * (1+z_r)
                  \int_{z_u>z_r} dz_u K_k(z_u)  cda(z_r) * (1-cda(z_r)) / cda(z_u)
    `Mr_mr`    is coefficient for magnification of references by unknowns.  Swap u,r in integral above.'''

    # Points at which to sample r functions
    zr = np.linspace( zr0-dzr/2, zr0+(Nzr-0.5)*dzr, Nzr*oversample_r+1)

    def collapse_r(array, axis=0):
        '''Take oversampled function of zr and integrate back into Nzr bins'''
        # Move the zr axis to the front
        x = np.moveaxis(array,axis,0)
        # Start with endpoints for trapezoid integration
        out = 0.5*(x[:-1:oversample_r] + x[oversample_r::oversample_r])
        # Add in middle points
        for i in range(1,oversample_r):
            out += x[i::oversample_r]
        # Include integration factor which yields unit integral over each r bin.
        out *= 1. / oversample_r
        # Move axis back
        if axis!=0:
            out = np.moveaxis(out,0,axis)
        return out

    dzk = 0.005   # dz for integrating over unknowns' kernels


    Nk = kernels.nz
    zbounds = kernels.zbounds()

    # Create empty arrays for results
    A = np.zeros( (Nk,Nzr), dtype=float)
    Mu = np.zeros( (Nk,Nzr), dtype=float)
    Mr = np.zeros( (Nk,Nzr), dtype=float)

    # Calculate needed quantities at reference z's
    cda_r = cda(zr)
    H0H_r = c_over_H(zr) / c_over_H(0.)  # H(z) / H0
    wdm_r = wdm(zr)
    chi_r = cda(zr)

    # Loop over kernels
    for k in range(Nk):
        # First the clustering, evaluated at common set of z's
        A[k] = collapse_r(wdm_r * kernels(k,zr))
        # Now the lensing
        zmin = zbounds[k,0]
        zmax = zbounds[k,1]
        n = int(np.floor( (zmax-zmin)/dzk)) + 1  # Number of z intervals to squeeze into kernel
        zk = np.linspace(zmin,zmax,n+1)
        dz = zk[1]-zk[0] 
        kk = kernels(k,zk)  # The kernel values at the endpoint will be zero

        chi_u = cda(zk)
        # Calculate when the unknowns are the lensing sources
        # Mean lensing factor over the kernel:
        lens_u = dz * np.sum(np.maximum(0., 1 - chi_r[:,np.newaxis]/chi_u[np.newaxis,:])*kk, axis=1)
        Mu[k] = collapse_r( lens_u * wdm_r * chi_r * (1+zr) * H0H_r)


        # When the references are the lensing sources:
        lens_r = np.maximum(0., 1 - chi_u[np.newaxis,:]/chi_r[:,np.newaxis]) # r is 1st index
        # Integrate over zk
        mm = dz * np.sum(kk*lens_r * wdm(zk) * chi_u * (1+zk) * (c_over_H(zk)/c_over_H(0.)), axis=1)
        # Integrate over zr:
        Mr[k] = collapse_r(mm)
    
    Mu *= 3 * Omega_m / c_over_H(0.)
    Mr *= 3 * Omega_m / c_over_H(0.)

    return jnp.array(A), jnp.array(Mu), jnp.array(Mr)

def prep_cov_w(cov_w):
    '''Get the inverse square root of w covariance for each u, which is the form
    we need for logpwz (D_q^-1 * U^T_w in notes).
    This version assumes the cov matrix has no cross-bin correlations so is
    provided (and Sw returned) with shape (Nu,Nr,Nr)'''
    
    # Take eigenvals/vecs for each u
    s,U = jnp.linalg.eigh(cov_w)
    # Sw is a "square root" of inverse of cov_w, so cov_w^{-1} = Sw.T @ Sw
    Sw = jnp.einsum('ur,usr->urs',s**(-0.5), U)  # indexed by (u,r,r') now.
    return Sw

def prep_cov_w_dense(cov_w,Nu,Nr):
    '''Get the inverse square root of w covariance for each u, which is the form
    we need for logpwz (D_q^-1 * U^T_w in notes).
    This version assumes the cov matrix is fully populated and is
    provided as (Nu*Nr, Nu*Nr),
    and Sw will be returned with shape (Nu,Nu,Nr,Nr)'''
    
    # Take eigenvals/vecs 
    s,U = jnp.linalg.eigh(cov_w)
    # Sw is a "square root" of inverse of cov_w, so cov_w^{-1} = Sw.T @ Sw
    Sw = jnp.einsum('r,sr->rs',s**(-0.5), U)  # indexed by (ur,u'r') now.
    Sw = jnp.swapaxes(Sw.reshape(Nu,Nr,Nu,Nr),2,1) # Now indexed by (u,u',r,r')
    return Sw

def prep_cov_w_diagr(cov_w):
    '''Get the inverse square root of w covariance for each u, which is the form
    we need for logpwz (D_q^-1 * U^T_w in notes).
    This version assumes the cov matrix is block-diagonal on r and is
    provided (and Sw returned) with shape (Nu,Nu,Nr)'''
    
    # Reshape matrix to be dense
    Nu = cov_w.shape[0]
    Nr = cov_w.shape[2]
    c = jnp.moveaxes(cov_w, 2, 0)  # Move r axis to front
    # Take eigenvals/vecs of blocks over u,u'
    s,U = jnp.linalg.eigh(cov_w)
    # Sw is a "square root" of inverse of cov_w, so cov_w^{-1} = Sw.T @ Sw
    Sw = jnp.einsum('ru,ruv->uvr',s**(-0.5), U)  # indexed by (u, u',r) now.
    return Sw

def w_model(f_um, b_u,  
            s_uk, alpha_u,  ar, b_r,
            alpha_r_basis, Sys_kr,
            A_mr, Mu_mr, Mr_mr, **kwargs):
    '''Calculate model values for w_ur.
    `f_um`:       Fraction of galaxies from unknown set u that lie in redshift kernel k,
                  shape (Nu,Nk)
    `b_u`:        Bias values for galaxies in u, shape (Nu)
    `b_r`:        Bias values for reference galaxy subset r, shape (Nr)
    `alpha_u`:    Magnification coefficients for each unknown bin
    `ar`:         Coefficients defining alpha_r
    `s_uk`:       Clustering Sys function coefficients
    `alpha_r_basis`:  Matrix of shape (Nr, Nar) defining basis functions over zr
                  for alpha_r, such that alpha_r = alpha_r_basis @ ar.
    `Sys_kr`:     Value of Sys basis function k at redshift of bin r, shape (Nk,Nr).
    `A_mr, Mu_mr, Mr_mr`: Theory matrices integrating over DM clustering for the
                  clustering and magnification terms of model for w, as calculated
                  in routines above.  Each has shape (Nm, Nr)
    Returns:
    `w_ur`:       Array of cross-correlations between unknown and reference bins,
                  shape (Nu,Nr).'''

    sys_ur = 1 + s_uk @ Sys_kr
    w_ur = jnp.einsum('um,u,r,mr,ur->ur', f_um, b_u, b_r, A_mr, sys_ur) \
         + jnp.einsum('um, r, u, mr->ur', f_um, b_r, alpha_u, Mu_mr) \
         + jnp.einsum('um, ra, a, u, mr->ur', f_um, alpha_r_basis, ar, b_u, Mr_mr)
    return w_ur
        
def concatenate_surveys(s1,s2):
    '''Concatenate quantities from two different redshift reference surveys
    into one.  Each input is a dictionary containing arrays for
    `b_r, alpha_r_basis, ar0, sigma_ar, Sys_kr, sigma_s_uk, w, Sw,`
    `A_mr, Mu_mr, Mr_mr, z_r.`  The output is another dictionary for
    which the `r,` `a`, and `k` axes have been concatenated.
    Any `alpha_u0,sigma_alpha_u` arguments are forwarded to output too.'''
    out = {}
    for k in ('b_r', 'ar0', 'sigma_ar', 'z_r'):
        # Concatenate on the first axis
        out[k] = jnp.concatenate( (s1[k], s2[k]), axis=0)

    # Concatenate on 2nd axis
    for k in ('sigma_s_uk', 'w', 'A_mr', 'Mu_mr', 'Mr_mr'):
        # Concatenate on the first axis
        out[k] = jnp.concatenate( (s1[k], s2[k]), axis=1)

    # Concatenate both axis - block matrix output
    for k in ('alpha_r_basis', 'Sys_kr'):
        nx,ny = s1[k].shape
        t = np.zeros( (nx+s2[k].shape[0], ny+s2[k].shape[1]), dtype=float)
        t[:nx, :ny] = s1[k]
        t[nx:, ny:] = s2[k]
        out[k] = jnp.array(t)

    for k in ('alpha_u0','sigma_alpha_u'):
        if k in s1:
            if k in s2:
                if not np.all(np.isclose(s1[k],s2[k])):
                    raise ValueError('s1 and s2 have differing {:s}:'.format(k) \
                                         + str(s1[k]) + ' vs ' + str(s2[k]))
            out[k] = s1[k]
        elif k in s2:
            out[k] = s2[k]
                                         

    # Fill blocks for a merged Sw, assuming no cross-survey covariance.
    # Depends on what kind of covariances we have...
    k = 'Sw'
    Nu = s1[k].shape[0]
    if s1[k].ndim==4:
        # Dense covariance, indices (u,u',r,r')
        n1 = s1[k].shape[2]
        Nr = s1[k].shape[2] + s2[k].shape[2]
        t = np.zeros( (Nu,Nu,Nr,Nr), dtype=float)
        t[:,:,:n1,:n1] = s1[k]
        t[:,:,n1:,n1:] = s2[k]
        out[k] = jnp.array(t)

    elif s1[k].shape[1]==Nu:
        # Block diagonal covariances on r, no cross-z cov
        # indices (u,u',r)
        out[k] = jnp.concatenate( (s1[k],s2[k]), axis=2)

    else:
        # Block diagonal on u, no cross-bin cov
        # indices (u,r,r')
        n1 = s1[k].shape[1]
        Nr = s1[k].shape[1] + s2[k].shape[1]
        t = np.zeros( (Nu,Nr,Nr), dtype=float)
        t[:,:n1,:n1] = s1[k]
        t[:,n1:,n1:] = s2[k]
        out[k] = jnp.array(t)
        

    return out

def logpwz(f_um, b_u,
           alpha_u0, sigma_alpha_u, 
           b_r, alpha_r_basis, ar0, sigma_ar,
           Sys_kr, sigma_s_uk, 
           w, Sw,
           A_mr, Mu_mr, Mr_mr, z_r,
           return_qmap=False,
           return_derivs=False):
    '''Calculate the log of the probability of observing wz data, and derivatives.
    The f_mu and b_u values are considered free parameters, and b_r as fixed.
    The values of the sys coefficients s_uk, and the magnification parameters
    alpha_u and ar (specifying alpha_r), are nuisance parameters with Gaussian priors that will
    be marginalized on the fly within this routine.

    The reference bins r can concatenate measurements from different spectroscopic surveys'
    bins. 

    It is assumed that there is no covariance between w_ur values with distinct u's, i.e.
    Cov(w_{ur}, w_{u'r'}) = 0 if u!=u'.

    Parameters:
    `f_um`:       Fraction of galaxies from unknown set u that lie in redshift kernel k,
                  shape (Nu,Nk)
    `b_u`:        Bias values for galaxies in u, shape (Nu)
    `b_r`:        Bias values for reference galaxy subset r, shape (Nr)
    `alpha_u0, sigma_alpha_u`: Mean and sigma of Gaussian priors on magnification
                  coefficients of unknown sources bin u, shape (Nu) each.
    `alpha_r_basis`:  Matrix of shape (Nr, Nar) defining basis functions over zr
                  for alpha_r, such that alpha_r = alpha_r_basis @ (ar0 + ar).
    `ar0, sigma_ar`:  Means and std dev's of Gaussian priors on the basis coefficients
                  of the alpha_r values.
    `Sys_kr`:     Value of Sys function k at redshift of bin r, shape (Nk,Nr).
    `sigma_s_uk`: Prior uncertainties on sys-error coefficients s_uk.  The means
                  of priors are assumed to be zero, shape (Nu,Nk).
    `w`:          Array of cross-correlations between unknown and reference bins,
                  shape (Nu,Nr).
    `Sw`:         "Square root" of the covariance matrix for w, produced in
                  routine above, shape (Nu, Nr, Nr).
    `A_mr, Mu_mr, Mr_mr`: Theory matrices integrating over DM clustering for the
                  clustering and magnification terms of model for w, as calculated
                  in routines above.  Each has shape (Nm, Nr)
    `return_qmap`: If True, second return value is list [s_uk, alpha_u, ar] of
                  the maximizing values of marginalized parameters, with
                  shapes (Nu,Nk), (Nu), (Nar) respectively.
    `return_derivs`: If True, last return value is list [ dlogp_df, dlogp_dbu]
                  with shapes (Nu,Nm), (Nu).

    Returns:
    `logp`:       Scalar value of log p(w | f_um, b_u, b_r)
    qmap list:    Max posterior values of q, if `return_qmap==True`
    derivs:       Derivatives of logp list, if `return_derivs==True`'''
    
    
    # The explicit parameter vector x will be concatenation of f_um and b_u.  We
    # won't actually need that vector
    # Total of Nu x Nm + Nu.
    Nu, Nm = f_um.shape
    Nr = b_r.shape[0]
    Nk = Sys_kr.shape[0]
    Nw = Nu*Nr
    Nar = alpha_r_basis.shape[1]  # Number of controlling params for alpha_r

    # We'll want derivs w.r.t. non-marginalized parameters f_um and b_u (could do b_r later),
    # so I will construct in two parts: first, _df is with respect to
    # f_um and will be indexed by [u,m].
    # Second part noted as _dbu will be indexed by [u].

    # w0 is the model for w evaluated at the mean values of the marginalized parameters q
    # w0_df is its derivs w.r.t. f_um:
    w0_df = jnp.einsum('u,r,mr->urm',b_u,  b_r,A_mr) + \
            jnp.einsum('u, r, mr->urm', alpha_u0, b_r, Mu_mr) + \
            jnp.einsum('u, ra, a, mr->urm', b_u, alpha_r_basis, ar0, Mr_mr)    # indexed by [u,r,m]

    # Derivs w.r.t. b_u
    w0_dbu = jnp.einsum('um,r,mr->ur', f_um,  b_r, A_mr) + \
             jnp.einsum('um, ra, a, mr->ur', f_um, alpha_r_basis, ar0, Mr_mr)  # indexed by [u,r]

    # Value:
    w0 = jnp.einsum('um,urm->ur',f_um, w0_df)  # indexed by [u,r]

    # The Delta vector holds the values of sqrt(cov_w)^{-1}*(w - w0)
    Delta = jnp.einsum('urs,us->ur',Sw,w-w0)  # Indexed by [u,r]

    # And its derivatives w.r.t. parameters:
    Delta_df = jnp.einsum('urs,usm->urm',Sw,-w0_df)  # Indexed by [u,r,m] where (u,m) index f
    Delta_dbu = jnp.einsum('urs,us->ur',Sw,-w0_dbu)  # Indexed by [u,r], where (u) indexes b_u

    # The W_q matrix gives derivatives of w_model w.r.t. marginalized parameters q,
    # times the standard deviation of each q.
    # The implicit (marginalized) parameters will concatenate:
    # * q0 = s_uk (Nu * Nk entries)
    # * q1 = alpha_u (Nu entries)
    # * q2 = ar (Nar entries, determining alpha_r)
    #
    N1 = Nu*Nk   # Index of the first element of q1
    N2 = N1 + Nu # Index of first element of q2
    Nq = N2 + Nar # Total size of marginalization
    
    # We will construct the combination B = Sw * w_q * D_q
    # I will keep these components of q in distinct arrays because they are
    # diagonal on different combinations of indices
    # 
    # First the s_uk values.  Note duplication (diagonal) of u index in w_ur and s_uk
    B0_df = jnp.einsum('usr,u,r,mr,kr,uk->uskm',Sw,b_u, b_r, A_mr, Sys_kr,sigma_s_uk) # indexed by [u,r,k,m]
    B0 = jnp.einsum('urkm,um->urk',B0_df, f_um) # indexed by [u,r,k]
    B0_dbu = jnp.einsum('usr,um,r,mr,kr,uk->usk',Sw,f_um, b_r,A_mr,Sys_kr,sigma_s_uk) # Indexed by [u,r,k]

    # Next the alpha_u terms
    B1_df = jnp.einsum('usr,r,mr,u->usm',Sw,b_r, Mu_mr, sigma_alpha_u) # indexed by [u,r,m]
    B1 = jnp.einsum('urm,um->ur',B1_df, f_um) # indexed by [u,r]
    # B1_dbu = 0.
    
    # And the alpha_r terms
    B2_df = jnp.einsum('usr,u,mr,ra, a->usam',Sw,b_u, Mr_mr, alpha_r_basis, sigma_ar) # indexed by [u,r,a,m]
    B2 = jnp.einsum('uram,um->ura',B2_df, f_um) # indexed by [u,r,a]
    B2_dbu = jnp.einsum('usr,um,mr,ra,a->usa',Sw, f_um,Mr_mr, alpha_r_basis, sigma_ar) # indexed by [u,r,a]

    # Need to build the matrix IBTB = (I + B^T B)
    # B^T * B will be Nq x Nq
    # Will be block diagonal on u except for the rows/cols for alpha_r, with a given alpha_r coupled to ur for all u.
    # So there is probably a faster way to do this inversion, but I'm just going to use dense-matrix routines.
    
    IBTB = jnp.eye(Nq)
    iu = jnp.arange(Nu)
    # Start with (B0,B0) section
    t1 = jnp.einsum('urk,url->ukl',B0,B0)
    t2 = jnp.zeros( (Nu,Nk,Nu,Nk), dtype=float)
    t2 = t2.at[iu,:,iu,:].set(t1)
    IBTB = IBTB.at[:N1,:N1].add(t2.reshape(N1,N1))
    # (B0,B1)
    t1 = jnp.einsum('urk,ur->uk',B0,B1)
    t2 = jnp.zeros( (Nu,Nk,Nu), dtype=float)
    t2 = t2.at[iu,:,iu].set(t1)
    IBTB = IBTB.at[:N1,N1:N2].set(t2.reshape(N1,Nu))
    IBTB = IBTB.at[N1:N2,:N1].set(t2.reshape(N1,Nu).T)  # (B1,B0)
    # (B0,B2)
    t1 = jnp.einsum('urk,ura->uka',B0,B2).reshape(N1,Nar)
    IBTB = IBTB.at[:N1,N2:].set(t1)
    IBTB = IBTB.at[N2:,:N1].set(t1.T)  # (B2,B0)
    # (B1,B1) - diagonal
    IBTB = IBTB.at[N1+iu,N1+iu].add(jnp.einsum('ur,ur->u',B1,B1))
    # (B1,B2)
    t1 = jnp.einsum('ur,ura->ua',B1,B2)
    IBTB = IBTB.at[N1:N2,N2:].set(t1)
    IBTB = IBTB.at[N2:,N1:N2].set(t1.T)  # (B2,B0)
    # (B2, B2) - diagonal
    IBTB = IBTB.at[N2:,N2:].add(jnp.einsum('ura,urb->ab',B2,B2))
    
    # Get the L matrix = Cholesky of t.
    L = jnp.linalg.cholesky(IBTB)

    # Determinant of IBTB easy from the triangular form:
    logdet = jnp.sum(jnp.log(jnp.diag(L)**2))

    # Actually want inverse of this:
    L = jnp.linalg.inv(L)

    # Calculate the vector L*B^T*D in three parts
    LBTD = jnp.einsum('quk,urk,ur->q',L[:,:N1].reshape(Nq,Nu,Nk), B0, Delta)
    LBTD = LBTD + jnp.einsum('qu,ur,ur->q',L[:,N1:N2], B1, Delta)
    LBTD = LBTD + jnp.einsum('qa,ura,ur->q',L[:,N2:], B2, Delta)
                         
    logp = -0.5*(logdet + jnp.sum(Delta*Delta) - jnp.sum(LBTD*LBTD))

    out = [logp]

    if return_qmap or return_derivs:
        LTLBTD = L.T @ LBTD  # Indexed by q
    if return_qmap:
        # Calculate the MAP values of marginalized parameters:
        qmap = [ LTLBTD[:N1].reshape(Nu,Nk) * sigma_s_uk,
                 LTLBTD[N1:N2] * sigma_alpha_u + alpha_u0,
                 LTLBTD[N2:] * sigma_ar + ar0]
        out.append(qmap)

    if return_derivs:
        ########## Derivs w.r.t. f_um and b_u. ###########
        # Former are indexed by (um), the latter by (u) in trailing indices
        # Needed for all derivs:
        LTL = L.T @ L  # [q,q]
    
        # Need B_df^T B, B_df^T Delta, B^T Delta_df, Delta^T Delta_df
        ## Easiest first:
        logp_df = -jnp.einsum('urm,ur->um',Delta_df, Delta) 
        logp_dbu = -jnp.einsum('ur,ur->u',Delta_dbu, Delta)  
    
        # Then B^T * Delta_df which will be indexed by (q,x), dotted into LTLBTD
        # df first, doing 3 parts of B
        logp_df = logp_df + jnp.einsum('uk,urk,urm->um',LTLBTD[:N1].reshape(Nu,Nk), B0, Delta_df) 
        logp_df = logp_df + jnp.einsum('u,ur,urm->um',LTLBTD[N1:N2], B1, Delta_df)  
        logp_df = logp_df + jnp.einsum('a,ura,urm->um',LTLBTD[N2:], B2, Delta_df)

        # dbu
        logp_dbu = logp_dbu + jnp.einsum('uk,urk,ur->u',LTLBTD[:N1].reshape(Nu,Nk), B0, Delta_dbu) 
        logp_dbu = logp_dbu + jnp.einsum('u,ur,ur->u',LTLBTD[N1:N2], B1, Delta_dbu)  
        logp_dbu = logp_dbu + jnp.einsum('a,ura,ur->u',LTLBTD[N2:], B2, Delta_dbu)

        # Then B^T_df * Delta which will be indexed by (q,x), dotted into LTLBTD
        # df first, doing 3 parts of B
        logp_df = logp_df + jnp.einsum('uk,urkm,ur->um',LTLBTD[:N1].reshape(Nu,Nk), B0_df, Delta) 
        logp_df = logp_df + jnp.einsum('u,urm,ur->um',LTLBTD[N1:N2], B1_df, Delta)  
        logp_df = logp_df + jnp.einsum('a,uram,ur->um',LTLBTD[N2:], B2_df, Delta)

        # dbu
        logp_dbu = logp_dbu + jnp.einsum('uk,urk,ur->u',LTLBTD[:N1].reshape(Nu,Nk), B0_dbu, Delta) 
        # B1_dbu=0, skip it
        logp_dbu = logp_dbu + jnp.einsum('a,ura,ur->u',LTLBTD[N2:], B2_dbu, Delta)

        ## Now need B^T_df B + B^T B_df
        # This will need dimensions (q,q,u,m), and we'll split q into its 3 parts.
        BTB_df = jnp.zeros((Nq,Nq,Nu,Nm), dtype=float)

        # (B0,B0)
        t1 = jnp.einsum('urkm,url->uklm',B0_df, B0)  # (uk,ul,um)
        # Add the k/l transpose
        t1 = t1 + jnp.swapaxes(t1,1,2)
        t2 = jnp.zeros((Nu,Nk,Nu,Nk,Nu,Nm), dtype=float)
        t2 = t2.at[iu,:,iu,:,iu,:].set(t1).reshape(Nu*Nk,Nu*Nk,Nu,Nm)
        BTB_df = BTB_df.at[:N1,:N1,:,:].set(t2)

        # (B0,B1)
        t1 = jnp.einsum('urkm,ur->ukm',B0_df, B1)  # (uk,u,um)
        t1 = t1 + jnp.einsum('urk,urm->ukm',B0,B1_df)
        t2 = jnp.zeros((Nu,Nk,Nu,Nu,Nm), dtype=float)
        t2 = t2.at[iu,:,iu,iu,:].set(t1).reshape(Nu*Nk,Nu,Nu,Nm)
        BTB_df = BTB_df.at[:N1,N1:N2,:,:].set(t2)
        BTB_df = BTB_df.at[N1:N2,:N1,:,:].set(jnp.swapaxes(t2,0,1))
    
        # (B0,B2)
        t1 = jnp.einsum('urkm,ura->ukam',B0_df, B2)  # (uk,a,um)
        t1 = t1 + jnp.einsum('urk,uram->ukam',B0,B2_df)
        t2 = jnp.zeros((Nu,Nk,Nar,Nu,Nm), dtype=float)
        t2 = t2.at[iu,:,:,iu,:].set(t1).reshape(Nu*Nk,Nar,Nu,Nm)
        BTB_df = BTB_df.at[:N1,N2:,:,:].set(t2)
        BTB_df = BTB_df.at[N2:,:N1,:,:].set(jnp.swapaxes(t2,0,1))
    
        # (B1,B1)
        t1 = 2 * jnp.einsum('urm,ur->um',B1_df, B1)  # (u,u,um) - self-conjugate
        BTB_df = BTB_df.at[N1+iu,N1+iu,iu,:].set(t1)

        # (B1,B2)
        t1 = jnp.einsum('urm, ura->uam',B1_df, B2) # (u,a,um)
        t1 = t1 + jnp.einsum('ur, uram->uam',B1, B2_df) # (u,a,um)
        BTB_df = BTB_df.at[N1+iu,N2:,iu,:].set(t1)
        BTB_df = BTB_df.at[N2:,N1+iu,iu,:].set(jnp.swapaxes(t1,0,1))

        # (B2,B2)
        t1 = jnp.einsum('uram, urb->abum',B2_df, B2) # (a,b,um)
        t1 = t1 + np.swapaxes(t1,0,1)
        BTB_df = BTB_df.at[N2:,N2:,:,:].set(t1)

        # Add derivative of logdet of (I+B^TB):
        logp_df = logp_df - 0.5*jnp.einsum('pq,qpum->um',LTL,BTB_df) 
        # And the last term of derivative:
        logp_df = logp_df - 0.5*jnp.einsum('p,pqum,q->um',LTLBTD, BTB_df, LTLBTD)

        ### Do the dbu derivs
        ## Now need B^T_df B + B^T B_df
        # This will need dimensions (q,q,u), and we'll split q into its 3 parts.
        BTB_dbu = jnp.zeros((Nq,Nq,Nu), dtype=float)

        # (B0,B0)
        t1 = jnp.einsum('urk,url->ukl',B0_dbu, B0)  # (uk,ul,u)
        # Add the k/l transpose
        t1 = t1 + jnp.swapaxes(t1,1,2)
        t2 = jnp.zeros((Nu,Nk,Nu,Nk,Nu), dtype=float)
        t2 = t2.at[iu,:,iu,:,iu].set(t1).reshape(Nu*Nk,Nu*Nk,Nu)
        BTB_dbu = BTB_dbu.at[:N1,:N1,:].set(t2)

        # (B0,B1)
        t1 = jnp.einsum('urk,ur->uk',B0_dbu, B1)  # (uk,u,u)
        # B1_dbu is zero, can skip.
        t2 = jnp.zeros((Nu,Nk,Nu,Nu), dtype=float)
        t2 = t2.at[iu,:,iu,iu].set(t1).reshape(Nu*Nk,Nu,Nu)
        BTB_dbu = BTB_dbu.at[:N1,N1:N2,:].set(t2)
        BTB_dbu = BTB_dbu.at[N1:N2,:N1,:].set(jnp.swapaxes(t2,0,1))
    
        # (B0,B2)
        t1 = jnp.einsum('urk,ura->uka',B0_dbu, B2)  # (uk,a,u)
        t1 = t1 + jnp.einsum('urk,ura->uka',B0,B2_dbu)
        t2 = jnp.zeros((Nu,Nk,Nar,Nu), dtype=float)
        t2 = t2.at[iu,:,:,iu].set(t1).reshape(Nu*Nk,Nar,Nu)
        BTB_dbu = BTB_dbu.at[:N1,N2:,:].set(t2)
        BTB_dbu = BTB_dbu.at[N2:,:N1,:].set(jnp.swapaxes(t2,0,1))
    
        # (B1,B1) - zero derivs w.r.t. b_u

        # (B1,B2)
        t1 = jnp.einsum('ur, ura->ua',B1, B2_dbu) # (u,a,u)
        BTB_dbu = BTB_dbu.at[N1+iu,N2:,iu].set(t1)
        BTB_dbu = BTB_dbu.at[N2:,N1+iu,iu].set(jnp.swapaxes(t1,0,1))

        # (B2,B2)
    
        t1 = jnp.einsum('ura, urb->abu',B2_dbu, B2) # (a,b,u)
        t1 = t1 + jnp.swapaxes(t1,0,1)  # Add transpose
        BTB_dbu = BTB_dbu.at[N2:,N2:,:].set(t1)

        # Add derivative of logdet of (I+B^TB):
        logp_dbu = logp_dbu - 0.5*jnp.einsum('pq,qpu->u',LTL,BTB_dbu)  

        # And the last term of derivative:
        logp_dbu = logp_dbu - 0.5*jnp.einsum('p,pqu,q->u',LTLBTD, BTB_dbu, LTLBTD)

        out.append( [logp_df, logp_dbu])
    return out

##################################
def logpwz_dense(f_um, b_u,
           alpha_u0, sigma_alpha_u, 
           b_r, alpha_r_basis, ar0, sigma_ar,
           Sys_kr, sigma_s_uk, 
           w, Sw,
           A_mr, Mu_mr, Mr_mr, z_r,
           return_qmap=False,
           return_derivs=False):
    '''Calculate the log of the probability of observing wz data, and derivatives.
    The f_mu and b_u values are considered free parameters, and b_r as fixed.
    The values of the sys coefficients s_uk, and the magnification parameters
    alpha_u and ar (specifying alpha_r), are nuisance parameters with Gaussian priors that will
    be marginalized on the fly within this routine.

    The reference bins r can concatenate measurements from different spectroscopic surveys'
    bins.  

    ** The w covariance and the `Sw` are assumed in this case to be dense, with indices 
    ** (u,u',r,r').

    Parameters:
    `f_um`:       Fraction of galaxies from unknown set u that lie in redshift kernel m,
                  shape (Nu,Nm)
    `b_u`:        Bias values for galaxies in u, shape (Nu)
    `b_r`:        Bias values for reference galaxy subset r, shape (Nr)
    `alpha_u0, sigma_alpha_u`: Mean and sigma of Gaussian priors on magnification
                  coefficients of unknown sources bin u, shape (Nu) each.
    `alpha_r_basis`:  Matrix of shape (Nr, Nar) defining basis functions over zr
                  for alpha_r, such that alpha_r = alpha_r_basis @ (ar0 + ar).
    `ar0, sigma_ar`:  Means and std dev's of Gaussian priors on the basis coefficients
                  of the alpha_r values.
    `Sys_kr`:     Value of Sys function k at redshift of bin r, shape (Nk,Nr).
    `sigma_s_uk`: Prior uncertainties on sys-error coefficients s_uk.  The means
                  of priors are assumed to be zero, shape (Nu,Nk).
    `w`:          Array of cross-correlations between unknown and reference bins,
                  shape (Nu,Nr).
    `Sw`:         "Square root" of the covariance matrix for w, produced in
                  routine above, shape (Nu, Nu, Nr, Nr).
    `A_mr, Mu_mr, Mr_mr`: Theory matrices integrating over DM clustering for the
                  clustering and magnification terms of model for w, as calculated
                  in routines above.  Each has shape (Nm, Nr)
    `return_qmap`: If True, second return value is list [s_uk, alpha_u, ar] of
                  the maximizing values of marginalized parameters, with
                  shapes (Nu,Nk), (Nu), (Nar) respectively.
    `return_derivs`: If True, last return value is list [ dlogp_df, dlogp_dbu]
                  with shapes (Nu,Nm), (Nu).

    Returns:
    `logp`:       Scalar value of log p(w | f_um, b_u, b_r)
    qmap list:    Max posterior values of q, if `return_qmap==True`
    derivs:       Derivatives of logp list, if `return_derivs==True`'''
    
    
    # The explicit parameter vector x will be concatenation of f_um and b_u.  We
    # won't actually need that vector
    # Total of Nu x Nm + Nu.
    Nu, Nm = f_um.shape
    Nr = b_r.shape[0]
    Nk = Sys_kr.shape[0]
    Nw = Nu*Nr
    Nar = alpha_r_basis.shape[1]  # Number of controlling params for alpha_r

    # We'll want derivs w.r.t. non-marginalized parameters f_um and b_u (could do b_r later),
    # so I will construct in two parts: first, _df is with respect to
    # f_um and will be indexed by [u,m].
    # Second part noted as _dbu will be indexed by [u].

    # w0 is the model for w evaluated at the mean values of the marginalized parameters q
    # w0_df is its derivs w.r.t. f_um:
    w0_df = jnp.einsum('u,r,mr->urm',b_u,  b_r,A_mr) + \
            jnp.einsum('u, r, mr->urm', alpha_u0, b_r, Mu_mr) + \
            jnp.einsum('u, ra, a, mr->urm', b_u, alpha_r_basis, ar0, Mr_mr)    # indexed by [u,r,m]


    # Value:
    w0 = jnp.einsum('um,urm->ur',f_um, w0_df)  # indexed by [u,r]

    # The Delta vector holds the values of sqrt(cov_w)^{-1}*(w - w0)
    Delta = jnp.einsum('uvrs,vs->ur',Sw,w-w0)  # Indexed by [u,r]


    # The W_q matrix gives derivatives of w_model w.r.t. marginalized parameters q,
    # times the standard deviation of each q.
    # The implicit (marginalized) parameters will concatenate:
    # * q0 = s_uk (Nu * Nk entries)
    # * q1 = alpha_u (Nu entries)
    # * q2 = ar (Nar entries, determining alpha_r)
    #
    N1 = Nu*Nk   # Index of the first element of q1
    N2 = N1 + Nu # Index of first element of q2
    Nq = N2 + Nar # Total size of marginalization
    
    # We will construct the combination B = Sw * w_q * D_q
    # I will keep these components of q in distinct arrays because they are
    # diagonal on different combinations of indices
    # 
    # First the s_uk values.  Note duplication (diagonal) of u index in w_ur and s_uk
    B0_df = jnp.einsum('uvsr,v,r,mr,kr,vk->uskvm',Sw,b_u, b_r, A_mr, Sys_kr,sigma_s_uk) # indexed by [u,r,k,u',m]
    # where u',m are the indices of the f_um derivative and (u',k) index the nuisance parameters s_uk.
    B0 = jnp.einsum('urkvm,vm->urvk',B0_df, f_um) # indexed by [u,r,u',k]  

    # Next the alpha_u terms
    B1_df = jnp.einsum('uvsr,r,mr,v->usvm',Sw,b_r, Mu_mr, sigma_alpha_u) # indexed by [u,r,u',m], u' indexes both b_u and f_um
    B1 = jnp.einsum('urvm,vm->urv',B1_df, f_um) # indexed by [u,r,u'] with u' indexing b_u
    
    # And the alpha_r terms
    B2_df = jnp.einsum('uvsr,v,mr,ra, a->usavm',Sw,b_u, Mr_mr, alpha_r_basis, sigma_ar) # indexed by [u,r,a,u',m]
    B2 = jnp.einsum('uravm,vm->ura',B2_df, f_um) # indexed by [u,r,a]

    # Concatenate the parts of B and put into 2d
    B = jnp.concatenate( (B0.reshape(Nu*Nr, Nu*Nk),
                          B1.reshape(Nu*Nr, Nu),
                          B2.reshape(Nu*Nr, Nar)), axis=1)


    # Need to build the matrix IBTB = (I + B^T B)
    # B^T * B will be Nq x Nq
    # Will be block diagonal on u except for the rows/cols for alpha_r, with a given alpha_r coupled to ur for all u.
    # So there is probably a faster way to do this inversion, but I'm just going to use dense-matrix routines.
    
    IBTB = jnp.eye(Nq) + B.T@B
    
    # Get the L matrix = Cholesky of t.
    L = jnp.linalg.cholesky(IBTB)

    # Determinant of IBTB easy from the triangular form:
    logdet = jnp.sum(jnp.log(jnp.diag(L)**2))

    # Actually want inverse of this:
    L = jnp.linalg.inv(L)

    # Calculate the vector L*B^T*D in three parts
    LBTD = jnp.einsum('qp,yp,y->q',L, B, Delta.flatten())
                         
    logp = -0.5*(logdet + jnp.sum(Delta*Delta) - jnp.sum(LBTD*LBTD))

    out = [logp]

    if return_qmap or return_derivs:
        LTLBTD = L.T @ LBTD  # Indexed by q
    if return_qmap:
        # Calculate the MAP values of marginalized parameters:
        qmap = [ LTLBTD[:N1].reshape(Nu,Nk) * sigma_s_uk,
                 LTLBTD[N1:N2] * sigma_alpha_u + alpha_u0,
                 LTLBTD[N2:] * sigma_ar + ar0]
        out.append(qmap)

    if return_derivs:
        ########## Derivs w.r.t. f_um and b_u. ###########
        # Former are indexed by (um), the latter by (u) in trailing indices
        # Needed for all derivs:
        LTL = L.T @ L  # [q,q]
    
        # Derivs w.r.t. b_u
        w0_dbu = jnp.einsum('um,r,mr->ur', f_um,  b_r, A_mr) + \
                 jnp.einsum('um, ra, a, mr->ur', f_um, alpha_r_basis, ar0, Mr_mr)  # indexed by [u,r]

        # Delta derivatives w.r.t. parameters:
        Delta_df = jnp.einsum('uvrs,vsm->urvm',Sw,-w0_df)  # Indexed by [u,r,u',m] where (u',m) index f
        Delta_dbu = jnp.einsum('uvrs,vs->urv',Sw,-w0_dbu)  # Indexed by [u,r,u'], where (u') indexes b_u ????

        # Need B_df^T B, B_df^T Delta, B^T Delta_df, Delta^T Delta_df
        ## Easiest first:
        logp_df = -jnp.einsum('urvm,ur->vm',Delta_df, Delta) # Vector over (u,m) of f's
        logp_dbu = -jnp.einsum('urv,ur->v',Delta_dbu, Delta)  # Vector over u of b_u
    
        # Fill in the parts of dB/du
        B0_dbu = jnp.einsum('uvsr,vm,r,mr,kr,vk->uskv',Sw,f_um, b_r,A_mr,Sys_kr,sigma_s_uk) # Indexed by [u,r,k,u'],
        # with u' indexing the b_u derivatives.
        # B1_dbu = 0.
        B2_dbu = jnp.einsum('uvsr,vm,mr,ra,a->usav',Sw, f_um,Mr_mr, alpha_r_basis, sigma_ar) # indexed by [u,r,a,u']

        B_dbu = jnp.zeros( (Nu,Nr,Nq,Nu), dtype=float)
        tmp = jnp.zeros( (Nu,Nr,Nu,Nk,Nu), dtype=float)
        iu = jnp.arange(Nu)
        ## !!! Something very trick here: when two "advanced indices" (=arrays)
        ## are separated by a slice, the "advanced" index is promoted to
        ## be first in the array, whereas when they are consecutive,
        ## the advanced index retains its order.
        ## So tmp.at[:,:,iu,:,iu,:] does *not* do what we expect.
        ## It needs input with the iu index *first*.
        ## https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
        tmp = tmp.at[:,:,iu,:,iu].set(jnp.moveaxis(B0_dbu,3,0))
        B_dbu = B_dbu.at[:,:,:N1,:].set(tmp.reshape(Nu,Nr,N1,Nu))
        B_dbu = B_dbu.at[:,:,N2:,:].set(B2_dbu)
        B_dbu = B_dbu.reshape(Nu*Nr, Nq, Nu)
        
        
        # Fill in the parts of dB/df
        B_df = jnp.zeros( (Nu,Nr,Nq,Nu,Nm), dtype=float)
        tmp = jnp.zeros( (Nu,Nr,Nu,Nk,Nu,Nm), dtype=float)
        ## See note above.
        tmp = tmp.at[:,:,iu,:,iu,:].set(jnp.moveaxis(B0_df,3,0))
        B_df = B_df.at[:,:,:N1,:,:].set(tmp.reshape(Nu,Nr,N1,Nu,Nm))
        
        tmp = jnp.zeros( (Nu,Nr,Nu,Nu,Nm), dtype=float)
        tmp = tmp.at[:,:,iu,iu,:].set(B1_df)
        B_df = B_df.at[:,:,N1:N2,:,:].set(tmp)
         
        B_df = B_df.at[:,:,N2:,:,:].set(B2_df)
        B_df = B_df.reshape(Nu*Nr, Nq, Nu,Nm)
                                            
        # Then B^T * Delta_df which will be indexed by (q,x), dotted into LTLBTD
        logp_df = logp_df + jnp.einsum('q,xq,xvm->vm',LTLBTD, B, Delta_df.reshape(Nu*Nr,Nu,Nm))

        # dbu
        logp_dbu = logp_dbu + jnp.einsum('q,xq,xu->u',LTLBTD, B, Delta_dbu.reshape(Nu*Nr,Nu))

        # Then B^T_df * Delta which will be indexed by (q,x), dotted into LTLBTD
        logp_df = logp_df + jnp.einsum('q,xqum,x->um',LTLBTD, B_df, Delta.flatten()) 

        # dbu
        logp_dbu = logp_dbu + jnp.einsum('q,xqu,x->u',LTLBTD, B_dbu, Delta.flatten()) 

        ## Now need B^T_df B + B^T B_df
        # This will need dimensions (q,q,u,m), and we'll split q into its 3 parts.
        BTB_df = jnp.einsum('xqum,xp->pqum',B_df, B)  # (q,q',u,m) output indices
        BTB_df = BTB_df + jnp.swapaxes(BTB_df,0,1)

        # Add derivative of logdet of (I+B^TB):
        logp_df = logp_df - 0.5*jnp.einsum('pq,qpum->um',LTL,BTB_df) 
        # And the last term of derivative:
        logp_df = logp_df - 0.5*jnp.einsum('p,pqum,q->um',LTLBTD, BTB_df, LTLBTD)

        ## Now B^T_dbu B + B^T B_dbu
        BTB_dbu = jnp.einsum('xqu,xp->pqu',B_dbu, B)  # (q,q',u) output indices
        BTB_dbu = BTB_dbu + jnp.swapaxes(BTB_dbu,0,1)

        # Add derivative of logdet of (I+B^TB):
        logp_dbu = logp_dbu - 0.5*jnp.einsum('pq,qpu->u',LTL,BTB_dbu)  

        # And the last term of derivative:
        logp_dbu = logp_dbu - 0.5*jnp.einsum('p,pqu,q->u',LTLBTD, BTB_dbu, LTLBTD)

        out.append( [logp_df, logp_dbu])
    return out
