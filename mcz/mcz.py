# Some implementations in JAX for the PZWZ likelihood
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre

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
    at each reference redshift r.  Note that the coefficients at k=0 are actually
    coming from the degree=1 Legendre polynomial, i.e. the constant is skipped.
    Arguments:
    `zr0, dzr`:  Central redshift of first reference bin and increment between them
    `Nr`:        Number of reference bins
    `Nk`:        Number of terms (order) of the systematic error function S_k(r)
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
    out = np.array( [ak[k] * eval_legendre(k+1, u) for k in range(Nk)] )

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
    we need for logpwz (D_q^-1 * U^T_w in notes).'''
    
    # Take eigenvals/vecs for each u
    s,U = jnp.linalg.eigh(cov_w)
    # Sw is a "square root" of inverse of cov_w, so cov_w^{-1} = Sw.T @ Sw
    Sw = jnp.einsum('ur,usr->urs',s**(-0.5), U)  # indexed by (u,r,r') now.
    return Sw

def logpwz(f_um, b_u, b_r,
           alpha_u0, alpha_r0,
           sigma_s_uk, sigma_alpha_u, sigma_alpha_r,
           w, Sw,
           Sys_kr, A_mr, Mu_mr, Mr_mr):
    '''Calculate the log of the probability of observing wz data, and derivatives.
    The f_mu and b_u values are considered free parameters, and b_r as fixed.
    The values of the sys coefficients s_uk, and the magnification parameters
    alpha_u and alpha_r, are nuisance parameters with Gaussian priors that will
    be marginalized on the fly within this routine.

    The reference bins r can concatenate measurements from different spectroscopic surveys'
    bins.  ??? Code needs an update if different Sys coefficients s_uk will be needed for
    distinct surveys, i.e. k indexes over multiple surveys.  There will need to be
    a matrix saying which k's apply to which r's.???

    It is assumed that there is no covariance between w_ur values with distinct u's, i.e.
    Cov(w_{ur}, w_{u'r'}) = 0 if u!=u'.

    Parameters:
    `f_um`:       Fraction of galaxies from unknown set u that lie in redshift kernel k,
                  shape (Nu,Nk)
    `b_u`:        Bias values for galaxies in u, shape (Nu)
    `b_r`:        Bias values for reference galaxy subset r, shape (Nr)
    `sigma_s_uk`: Prior uncertainties on sys-error coefficients s_uk.  The means
                  of priors are assumed to be zero, shape (Nu,Nk).
    `alpha_u0, sigma_alpha_u`: Mean and sigma of Gaussian priors on magnification
                  coefficients of unknown sources bin u, shape (Nu) each.
    `alpha_r0, sigma_alpha_r`: Mean and sigma of Gaussian priors on magnification
                  coefficients of reference objects bin r, shape (Nr) each.
    `w`:          Array of cross-correlations between unknown and reference bins,
                  shape (Nu,Nr).
    `Sw`:         "Square root" of the covariance matrix for w, produced in
                  routine above, shape (Nu, Nr, Nr).
    `Sys_kr`:     Value of Sys function k at redshift of bin r, shape (Nk,Nr).
    `A_mr, Mu_mr, Mr_mr`: Theory matrices integrating over DM clustering for the
                  clustering and magnification terms of model for w, as calculated
                  in routines above.  Each has shape (Nm, Nr)
    Returns:
    `logp`:       Scalar value of log p(w | f_um, b_u, b_r)
    `logp_df`:    Derivatives of logp w.r.t. components f_um, shape (Nu,Nm)
    `logp_dbu`:   Derivatives of logp w.r.t. b_u values, shape (Nu).'''
    
    '''Calculate log p(w | f_um, b_u) given fixed values of b_r, marginalizing
    over the nuisance parameters s_uk (systematic for bins of u), alpha_u, and 
    alpha_r.'''

    '''cov_w is assumed to be diagonal in unknowns' bins u, so indexed by (u,r,r'),
    and symmetric in r/r'.'''

    
    # The explicit parameter vector x will be concatenation of f_um and b_u.  We
    # won't actually need that vector
    # Total of Nu x Nm + Nu.
    Nu, Nm = f_um.shape
    Nr = b_r.shape[0]
    Nk = Sys_kr.shape[0]
    Nw = Nu*Nr


    # We'll want derivs w.r.t. non-marginalized parameters f_um and b_u (could do b_r later),
    # so I will construct in two parts: first, _df is with respect to
    # f_um and will be indexed by [u,m].
    # Second part noted as _dbu will be indexed by [u].

    # w0 is the model for w evaluated at the mean values of the marginalized parameters q
    # w0_df is its derivs w.r.t. f_um:
    w0_df = jnp.einsum('u,r,mr->urm',b_u,  b_r,A_mr) + \
            jnp.einsum('u, r, mr->urm', alpha_u0, b_r, Mu_mr) + \
            jnp.einsum('u, r, mr->urm', b_u, alpha_r0, Mr_mr)    # indexed by [u,r,m]

    # Derivs w.r.t. b_u
    w0_dbu = jnp.einsum('um,r,mr->ur', f_um,  b_r, A_mr) + \
             jnp.einsum('um, r, mr->ur', f_um, alpha_r0, Mr_mr)  # indexed by [u,r]

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
    # * q0 = s_uk (Nu * Nk entries, ???subdivide r by reference set R???)
    # * q1 = alpha_u (Nu entries)
    # * q2 = alpha_r (Nr entries,  ???subdivide r by reference set R???)
    #
    N1 = Nu*Nk   # Index of the first element of q1
    N2 = N1 + Nu # Index of first element of q2
    Nq = N2 + Nr # Total size of marginalization
    
    # We will construct the combination B = Sw * w_q * D_q
    # I will keep these components of q in distinct arrays because they are
    # diagonal on different combinations of indices
    # 
    # First the s_uk values.  Note duplication (diagonal) of u index in w_ur and s_uk
    B0_df = jnp.einsum('usr,u,r,mr,kr,uk->uskm',Sw,b_u, b_r, A_mr, Sys_kr,sigma_s_uk) # indexed by [u,r,k,m]
    B0 = jnp.einsum('urkm,um->urk',B0_df, f_um) # indexed by [u,r,k]
    B0_dbu = jnp.einsum('usr,um,r,mr,kr,uk->usk',Sw,f_um, b_r,A_mr,Sys_kr,sigma_s_uk) # Indexed by [u,r,k]

    ### WORKS print('B0:',B0[1,6,1], B0_dbu[1,6,1]) ###

    # Next the alpha_u terms
    B1_df = jnp.einsum('usr,r,mr,u->usm',Sw,b_r, Mu_mr, sigma_alpha_u) # indexed by [u,r,m]
    B1 = jnp.einsum('urm,um->ur',B1_df, f_um) # indexed by [u,r]
    # B1_dbu = 0.
    
    # And the alpha_r terms
    B2_df = jnp.einsum('usr,u,mr,r->usm',Sw,b_u, Mr_mr, sigma_alpha_r) # indexed by [u,r,m]
    B2 = jnp.einsum('urm,um->ur',B2_df, f_um) # indexed by [u,r]
    B2_dbu = jnp.einsum('usr,um,mr,r->us',Sw, f_um,Mr_mr, sigma_alpha_r) # indexed by [u,r]

    ### WORKS print('B2:',B2[1,6], B2_dbu[1,6],'->',B2[1,6]+0.1*B2_dbu[1,6]) ###

    # Need to build the matrix IBTB = (I + B^T B)
    # B^T * B will be Nq x Nq
    # Will be block diagonal on u except for the rows/cols for alpha_r, with a given alpha_r coupled to ur for all u.
    # So there is probably a faster way to do this inversion, but I'm just going to use dense-matrix routines.
    
    IBTB = jnp.eye(Nq)
    iu = jnp.arange(Nu)
    ir = jnp.arange(Nr)
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
    t1 = jnp.einsum('urk,ur->ukr',B0,B2).reshape(N1,Nr)
    IBTB = IBTB.at[:N1,N2:].set(t1)
    IBTB = IBTB.at[N2:,:N1].set(t1.T)  # (B2,B0)
    # (B1,B1) - diagonal
    IBTB = IBTB.at[N1+iu,N1+iu].add(jnp.einsum('ur,ur->u',B1,B1))
    # (B1,B2)
    t1 = B1 * B2  # [u,r] * [u,r] -> [u,r]
    IBTB = IBTB.at[N1:N2,N2:].set(t1)
    IBTB = IBTB.at[N2:,N1:N2].set(t1.T)  # (B2,B0)
    # (B2, B2) - diagonal
    IBTB = IBTB.at[N2+ir,N2+ir].add(jnp.einsum('ur,ur->r',B2,B2))
    
    # Get the L matrix = Cholesky of t.
    L = jnp.linalg.cholesky(IBTB)

    # Determinant of IBTB easy from the triangular form:
    logdet = jnp.sum(jnp.log(jnp.diag(L)**2))

    # Actually want inverse of this:
    L = jnp.linalg.inv(L)

    # Calculate the vector L*B^T*D in three parts
    LBTD = jnp.einsum('quk,urk,ur->q',L[:,:N1].reshape(Nq,Nu,Nk), B0, Delta)
    LBTD = LBTD + jnp.einsum('qu,ur,ur->q',L[:,N1:N2], B1, Delta)
    LBTD = LBTD + jnp.einsum('qr,ur,ur->q',L[:,N2:], B2, Delta)
                         
    logp = -0.5*(logdet + jnp.sum(Delta*Delta) - jnp.sum(LBTD*LBTD))

    ########## Derivs w.r.t. f_um ###########
    # Needed for all derivs:
    LTL = L.T @ L  # [q,q]
    LTLBTD = L.T @ LBTD  # Indexed by q
    
    # Need B_df^T B, B_df^T Delta, B^T Delta_df, Delta^T Delta_df
    ## Easiest first:
    logp_df = -jnp.einsum('urm,ur->um',Delta_df, Delta) 
    logp_dbu = -jnp.einsum('ur,ur->u',Delta_dbu, Delta)  
    
    # Then B^T * Delta_df which will be indexed by (q,x), dotted into LTLBTD
    # df first, doing 3 parts of B
    logp_df = logp_df + jnp.einsum('uk,urk,urm->um',LTLBTD[:N1].reshape(Nu,Nk), B0, Delta_df) 
    logp_df = logp_df + jnp.einsum('u,ur,urm->um',LTLBTD[N1:N2], B1, Delta_df)  
    logp_df = logp_df + jnp.einsum('r,ur,urm->um',LTLBTD[N2:], B2, Delta_df)

    # dbu
    logp_dbu = logp_dbu + jnp.einsum('uk,urk,ur->u',LTLBTD[:N1].reshape(Nu,Nk), B0, Delta_dbu) 
    logp_dbu = logp_dbu + jnp.einsum('u,ur,ur->u',LTLBTD[N1:N2], B1, Delta_dbu)  
    logp_dbu = logp_dbu + jnp.einsum('r,ur,ur->u',LTLBTD[N2:], B2, Delta_dbu)

    # Then B^T_df * Delta which will be indexed by (q,x), dotted into LTLBTD
    # df first, doing 3 parts of B
    logp_df = logp_df + jnp.einsum('uk,urkm,ur->um',LTLBTD[:N1].reshape(Nu,Nk), B0_df, Delta) 
    logp_df = logp_df + jnp.einsum('u,urm,ur->um',LTLBTD[N1:N2], B1_df, Delta)  
    logp_df = logp_df + jnp.einsum('r,urm,ur->um',LTLBTD[N2:], B2_df, Delta)

    # dbu
    logp_dbu = logp_dbu + jnp.einsum('uk,urk,ur->u',LTLBTD[:N1].reshape(Nu,Nk), B0_dbu, Delta) 
    # B1_dbu=0, skip it
    logp_dbu = logp_dbu + jnp.einsum('r,ur,ur->u',LTLBTD[N2:], B2_dbu, Delta)

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
    t1 = jnp.einsum('urkm,ur->ukrm',B0_df, B2)  # (uk,r,um)
    t1 = t1 + jnp.einsum('urk,urm->ukrm',B0,B2_df)
    t2 = jnp.zeros((Nu,Nk,Nr,Nu,Nm), dtype=float)
    t2 = t2.at[iu,:,:,iu,:].set(t1).reshape(Nu*Nk,Nr,Nu,Nm)
    BTB_df = BTB_df.at[:N1,N2:,:,:].set(t2)
    BTB_df = BTB_df.at[N2:,:N1,:,:].set(jnp.swapaxes(t2,0,1))
    
    # (B1,B1)
    t1 = 2 * jnp.einsum('urm,ur->um',B1_df, B1)  # (u,u,um) - self-conjugate
    BTB_df = BTB_df.at[N1+iu,N1+iu,iu,:].set(t1)

    # (B1,B2)
    t1 = jnp.einsum('urm, ur->urm',B1_df, B2) # (u,r,um)
    t1 = t1 + jnp.einsum('ur, urm->urm',B1, B2_df) # (u,r,um)
    BTB_df = BTB_df.at[N1+iu,N2:,iu,:].set(t1)
    BTB_df = BTB_df.at[N2:,N1+iu,iu,:].set(jnp.swapaxes(t1,0,1))

    # (B2,B2)
    t1 = 2 * jnp.einsum('urm, ur->rum',B2_df, B2) # (r,um) - self-conjugate
    BTB_df = BTB_df.at[N2+ir,N2+ir,:,:].set(t1)

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
    t1 = jnp.einsum('urk,ur->ukr',B0_dbu, B2)  # (uk,r,u)
    t1 = t1 + jnp.einsum('urk,ur->ukr',B0,B2_dbu)
    t2 = jnp.zeros((Nu,Nk,Nr,Nu), dtype=float)
    t2 = t2.at[iu,:,:,iu].set(t1).reshape(Nu*Nk,Nr,Nu)
    BTB_dbu = BTB_dbu.at[:N1,N2:,:].set(t2)
    BTB_dbu = BTB_dbu.at[N2:,:N1,:].set(jnp.swapaxes(t2,0,1))
    
    # (B1,B1) - zero derivs w.r.t. b_u

    # (B1,B2)
    t1 = jnp.einsum('ur, ur->ur',B1, B2_dbu) # (u,r,u)
    BTB_dbu = BTB_dbu.at[N1+iu,N2:,iu].set(t1)
    BTB_dbu = BTB_dbu.at[N2:,N1+iu,iu].set(jnp.swapaxes(t1,0,1))

    # (B2,B2)
    
    t1 = 2 * jnp.einsum('ur, ur->ru',B2_dbu, B2) # (r,u) - self-conjugate
    BTB_dbu = BTB_dbu.at[N2+ir,N2+ir,:].set(t1)

    # Add derivative of logdet of (I+B^TB):
    logp_dbu = logp_dbu - 0.5*jnp.einsum('pq,qpu->u',LTL,BTB_dbu)  

    ### BTB derivs work
    ### ii = N2+6 
    ### WORKS! print(BTB[ii,ii],BTB_dbu[ii,ii,1],BTB[ii,ii]+0.1*BTB_dbu[ii,ii,1]) ###
    ### ii = Nk + 2 
    ### jj = N2+6 
    ### print(BTB[ii,jj],BTB_dbu[ii,jj,1],BTB[ii,jj]+0.1*BTB_dbu[ii,jj,1]) ###
    
    ### WORKS! print(-0.5*logdet, -0.5*jnp.einsum('pq,qpu->u',LTL,BTB_dbu)[1],-0.5*(logdet+0.1*jnp.einsum('pq,qpu->u',LTL,BTB_dbu)[1])) #### debug
    ### WORKS! print(jnp.sum(Delta*Delta), 2*jnp.einsum('ur,ur->u',Delta_dbu, Delta)) #### debug
    
    
    # And the last term of derivative:
    logp_dbu = logp_dbu - 0.5*jnp.einsum('p,pqu,q->u',LTLBTD, BTB_dbu, LTLBTD)

    
    return logp, logp_df, logp_dbu


          
        


