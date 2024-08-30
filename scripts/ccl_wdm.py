#!/usr/bin/env python

# Script that will use CCL to make lookup tables for
# cosmological distances and w_dm quantity for WZ
# analysis

import sys
import numpy as np
import pyccl as ccl

import astropy.constants as c
import astropy.units as u
c_km_s = c.c.to(u.km/u.s).value

# DES Y3 3x2 + All ext
cosmo = ccl.Cosmology(h=0.68, Omega_b=0.0487, Omega_c=0.306-0.0487, sigma8=0.804, n_s=0.965,
                     Neff=3.046)
# Pure Planck2018
h = 0.674
cosmo = ccl.Cosmology(h=h, Omega_b=0.02233/h/h, Omega_c=0.1198/h/h, A_s=2.09e-9, n_s=0.965,
                     Neff=3.046)

# William's choice
h = 0.67
cosmo = ccl.Cosmology(h=h, Omega_b=0.045, Omega_c=0.27, A_s=2.1e-9, n_s=0.96,
                     Neff=3.046)

c_over_H0 = c_km_s / (100 * h)   # Hubble scale at z=0, in Mpc

def plx(x, lmax=1000):
    '''Calculate Legendre up to order lmax using recurrence.
    Prepends a dimension for ell to input x values.'''
    out = []
    out.append(np.ones_like(x))
    out.append(np.array(x))
    for i in range(2,lmax):
        out.append(((2*i-1)*x*out[-1] - (i-1)*out[-2]) / i)
    return np.array(out)

def w_dm(z, lmax=int(1e5)):
    '''Calculate the projected DM clustering at redshift z
    weighted over angles as 1/theta'''
    a = 1/(1+z)
    chi = ccl.comoving_angular_distance(cosmo,a)
    ell = np.arange(lmax)
    # Range of angles used is 1.5-5 Mpc (no h???)
    theta = np.linspace(1.5,5,100) / chi
    weight = 1/theta
    weight /= np.sum(weight) # Normalize to unit integral
    
    # Get the Legendre integral d_theta W(theta) P_l(cos(theta))
    costh = np.cos(theta)
    ell_weight = np.einsum('ij,j->i',plx(costh,lmax=lmax), weight)

    # Integrate over power spectrum
    integrand = ell_weight * ccl.nonlin_matter_power(cosmo, k=(ell+0.5)/chi, a=a) \
       * (2*ell+1) / 4 / np.pi
    return np.sum(integrand)   / (chi*chi*c_over_H0) * ccl.h_over_h0(cosmo, a)

z = np.arange(0.01,4.02,0.01)
w = []
for zz in z:
    w.append(w_dm(zz, lmax=int(1e5)))

a = 1/(1+z)
np.savez('ccl_wdm',z=z,wdm=w,cda=ccl.comoving_angular_distance(cosmo,a),c_over_H=c_over_H0/ccl.h_over_h0(cosmo,a))

sys.exit()
