import os
import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
#from docopt import docopt

from astropy import units as u
from astropy import constants
import scipy.optimize as sop
from scipy.interpolate import interp1d
from numpy.polynomial import Chebyshev as Pfit

#args = docopt(__doc__)
plot=True

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

### Read MESA file
p = mr.MesaData('profile_solar.data')
mass = p.mass[::-1] * u.M_sun
r = 10**(p.logR[::-1]) * u.R_sun
r_Rsol = 10**(p.logR[::-1]) * u.R_sun
mass, r = mass.cgs, r.cgs
rho = 10**p.logRho[::-1] * u.g / u.cm**3
P = 10**p.logP[::-1] * u.g / u.cm / u.s**2
T = 10**p.logT[::-1] * u.K
opacity = p.opacity[::-1] * (u.cm**2 / u.g)
Luminosity = p.luminosity[::-1] * u.L_sun
Luminosity = Luminosity.cgs
mlt_conv_vel = p.conv_vel * u.cm / u.s
g = constants.G.cgs*mass/r**2

conv_lum = p.conv_L_div_L[::-1] * Luminosity

grad_rad = p.gradr[::-1]
grad_ad  = (p.gradr_div_grada[::-1])**(-1) * grad_rad


rad_cond = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho * opacity) # = rho * cp * rad_diffusivity
rad_cond = rad_cond.cgs



#grad = dlnT/dlnP = (dlnT/dr)*(dr/dlnP) = (P/T)*(dT/dr)*(dr/dP)
my_grad_rad = 3*opacity*P*Luminosity / (16 * constants.sigma_sb.cgs * g * T**4 * 4 * np.pi * r**2)


bot_cz = sop.brentq(interp1d(r, grad_rad - grad_ad), (0.6*u.R_sun).cgs.value, (0.8*u.R_sun).cgs.value)
bot_cz_Rsol = sop.brentq(interp1d(r_Rsol, grad_rad - grad_ad), 0.6, 0.8)


bot_plot = bot_cz - (0.1*u.R_sun).cgs.value
top_plot = bot_cz + (0.1*u.R_sun).cgs.value
good = (r.value >= bot_plot)*(r.value <= top_plot)

bot_plot_Rsol = bot_cz_Rsol - 0.1
top_plot_Rsol = bot_cz_Rsol + 0.1


grad_T_rad = -(T/P)*rho*g*grad_rad
grad_T_ad  = -(T/P)*rho*g*grad_ad

rho_interface = interp1d(r, rho)(bot_cz)
R_interface = P/(rho_interface*T)
bouss_grad_T_rad = -g*grad_rad/R_interface
bouss_grad_T_ad  = -g*grad_ad/R_interface
bouss_grad_T_rad = -(T/P)*rho_interface*g*grad_rad
bouss_grad_T_ad  = -(T/P)*rho_interface*g*grad_ad




plt.figure()
plt.axvline(bot_cz_Rsol, c='k', lw=0.5)
plt.plot(r_Rsol, grad_rad, c='r', label=r'$\nabla_{\rm{rad}}$')
#plt.plot(r_Rsol, my_grad_rad, ls='--', c='k')
plt.plot(r_Rsol, grad_ad, c='b', label=r'$\nabla_{\rm{ad}}$')
plt.ylim(0, 3)
plt.xlim(bot_plot_Rsol, top_plot_Rsol)
plt.ylabel(r'$\nabla$')
plt.xlabel(r'$r/R_{\odot}$')
plt.savefig('nabla_plot.png', dpi=300, bbox_inches='tight')

plt.figure()
avg_mag_diff = np.sum((np.abs((grad_T_rad - grad_T_ad)*r**2*np.gradient(r)))[good])/(top_plot**3-bot_plot**3)
plt.axvline(bot_cz_Rsol, c='k', lw=0.5)
plt.plot(r_Rsol, -grad_T_rad, label=r'$\frac{\partial T_{\rm{rad}}}{\partial z}$', c='r')
plt.plot(r_Rsol, -grad_T_ad, label=r'$\frac{\partial T_{\rm{ad}}}{\partial z}$', c='b')
#plt.plot(r_Rsol, -bouss_grad_T_rad, label=r'$\frac{\partial T_{\rm{rad}}}{\partial z}$, $\rho = \rho_{\rm{interface}}$', c='r', ls='--')
#plt.plot(r_Rsol, -bouss_grad_T_ad, label=r'$\frac{\partial T_{\rm{ad}}}{\partial z}$, $\rho = \rho_{\rm{interface}}$', c='b', ls='--')
plt.yscale('log')
plt.xlim(bot_plot_Rsol, top_plot_Rsol)
plt.legend()
plt.ylim(avg_mag_diff.value, avg_mag_diff.value*10)
plt.ylabel(r'$-\frac{\partial T}{\partial z} = \frac{ \rho g T}{P} \nabla$')
plt.xlabel(r'$r/R_{\odot}$')
plt.savefig('gradT_plot.png', dpi=300, bbox_inches='tight')

plt.figure()
avg_mag = np.sum((np.abs(rad_cond*r**2*np.gradient(r)))[good])/(top_plot**3-bot_plot**3)
plt.axvline(bot_cz_Rsol, c='k', lw=0.5)
plt.plot(r_Rsol, rad_cond, c='k')
plt.yscale('log')
plt.xlim(bot_plot_Rsol, top_plot_Rsol)
plt.ylim(avg_mag.value/10, avg_mag.value*10)
plt.ylabel(r'$k = \frac{16\, \sigma_{\rm{SB}}\, T^3}{3\, \rho\, \kappa}$')
plt.xlabel(r'$r/R_{\odot}$')
plt.savefig('conductivity_plot.png', dpi=300, bbox_inches='tight')

plt.figure()
avg_mag = np.sum((np.abs(rad_cond*r**2*np.gradient(r)))[good])/(top_plot**3-bot_plot**3)
L_ad = -4*np.pi*r**2 * rad_cond * grad_T_ad
plt.axvline(bot_cz_Rsol, c='k', lw=0.5)
plt.plot(r_Rsol, Luminosity, c='orange', label=r'$L_{\odot}$')
plt.plot(r_Rsol, conv_lum, c='k', label=r'$L_{\rm{conv}}$')
plt.plot(r_Rsol, L_ad, c='indigo', label=r'$L_{\rm{cond, ad}} = -4\pi r^2 k_{\rm{rad}} \frac{\partial T_{\rm{ad}}}{\partial z}$')
plt.xlim(bot_plot_Rsol, top_plot_Rsol)
plt.ylim(0, 2*Luminosity.max().value)
plt.legend()
plt.ylabel(r'$L (erg/s)$')
plt.xlabel(r'$r/R_{\odot}$')
plt.savefig('luminosity_plot.png', dpi=300, bbox_inches='tight')



#plt.figure()
#plt.plot(r_Rsol, np.abs(grad_T_ad - grad_T_rad), label='mesa', c='r')
##plt.plot(r_Rsol, np.abs(bouss_grad_T_ad - bouss_grad_T_rad), label=r'$\rho = \rho_{\rm{interface}}$', c='b')
#plt.yscale('log')
#plt.xlim(bot_plot_Rsol, top_plot_Rsol)
#plt.legend()
#plt.ylim(avg_mag_diff.value/100, avg_mag_diff.value*100)
#plt.ylabel(r'$|\frac{\partial T_{\rm{ad}}}{\partial z} - \frac{\partial T_{\rm{ad}}}{\partial z}|$')
#plt.xlabel(r'$r/R_{\odot}$')
#
#plt.show()
#
