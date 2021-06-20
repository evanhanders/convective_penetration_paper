from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from scipy.optimize import brentq
from scipy.interpolate import interp1d

import pandas as pd

col_width = 3.25
golden_ratio = 1.61803398875

#d = directory = 'erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart4/'
d = directory = 'new_erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.4/'
filename = directory + '/avg_profs/averaged_avg_profs.h5'

S = float(d.split('_S')[-1].split('_')[0])
P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
Re = float(d.split('_Re')[-1].split('_')[0])
Lz = float(d.split('_Lz')[-1].split('_')[0])

fig = plt.figure(figsize=(2*col_width, 0.7*col_width/golden_ratio))
ax1 = fig.add_axes([0.00,    0, 0.27, 1])
ax2 = fig.add_axes([0.38,    0, 0.27, 1])
ax3 = fig.add_axes([0.75,    0, 0.27, 1])

with h5py.File(filename, 'r') as f:
    enstrophy_profiles = f['enstrophy'][()]
    fconv_profiles = f['F_conv'][()]
    f_ke_profiles = f['F_KE_p'][()]
    grad_profiles = -f['T_z'][()]
    grad_ad_profiles = -f['T_ad_z'][()]
    grad_rad_profiles = -f['T_rad_z'][()]
    z = f['z'][()]

N = -5
dissipation = enstrophy_profiles[N,:]/Re
fconv       = fconv_profiles[N,:]
ke_eqn_rhs  = fconv - dissipation


ax1.axhline(0, c='k', lw=0.5)
ax1.plot(z, dissipation, c='indigo', label='$\omega^2 / \mathcal{R}$')
ax1.plot(z, fconv, c='red', label=r'$F_{\rm{conv}}$')
ax1.plot(z, ke_eqn_rhs, c='blue', label=r'$F_{\rm{conv}}$ - $\omega^2 / \mathcal{R}$')
ax1.legend(loc='best', fontsize=8)
ax1.set_ylim(-0.25, 0.25)
ax1.set_ylabel('KE sources')

f_ke_p = f_ke_profiles[N,:]

ax2.axhline(0, c='k', lw=0.5)
ax2.plot(z, f_ke_p, c='blue')
ax2.axvline(brentq(interp1d(z, f_ke_p), 0.05, 0.9), c='k', ls='--')
ax2.set_ylabel(r'$F_{\rm{KE}}$')


grad = grad_profiles[N,:]
grad_ad = grad_ad_profiles[N,:]
grad_rad = grad_rad_profiles[N,:]

ax3.axhline(1, c='k', lw=0.5)
ax3.plot(z, grad/grad_ad, c='green', label=r'$\nabla/\nabla_{\rm{ad}}$')
ax3.plot(z, grad_rad/grad_ad, c='red', label=r'$\nabla_{\rm{rad}}/\nabla_{\rm{ad}}$')
ax3.legend(loc='best', fontsize=8)
ax3.set_ylim(0.75, 1.2)
ax3.set_ylabel(r'$\nabla/\nabla_{\rm{ad}}$')


for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, Lz)
    ax.set_xlabel('z')

fig.savefig('theory_profiles.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/theory_profiles.pdf', dpi=300, bbox_inches='tight')
