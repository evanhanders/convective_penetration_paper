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

brewer_green  = np.array(( 27,158,119, 255))/255
brewer_orange = np.array((217, 95,  2, 255))/255
brewer_purple = np.array((117,112,179, 255))/255

Re = 6.4e3
P = 4
S = 1e3
Lz = 2
filename = 'turbulent_profiles/re6.4e3_p4e0_s1e3_avg_profiles.h5'
with h5py.File(filename, 'r') as f:
    enstrophy_profiles = f['enstrophy'][()]
    fconv_profiles = f['F_conv'][()]
    f_ke_profiles = f['F_KE_p'][()]
    grad_profiles = -f['T_z'][()]
    grad_ad_profiles = -f['T_ad_z'][()]
    grad_rad_profiles = -f['T_rad_z'][()]
    z = f['z'][()]

N = -2
dissipation = enstrophy_profiles[N,:]/Re
fconv       = fconv_profiles[N,:]
ke_eqn_rhs  = fconv - dissipation

grad     = grad_profiles[N,:]
grad_ad  = grad_ad_profiles[N,:]
grad_rad = grad_rad_profiles[N,:]


fig_grad = plt.figure(figsize=(col_width, col_width/golden_ratio))
fig_theory = plt.figure(figsize=(col_width, 1.5*col_width/golden_ratio))

ax_grad = fig_grad.add_subplot(1,1,1)

ax_flux   = fig_theory.add_axes([0.00, 0.50, 1, 0.5])
ax_source = fig_theory.add_axes([0.00, 0.00, 1, 0.5])


ax_grad.axvline(1.044, c='k', lw=0.5, ls='--')
ax_grad.axvline(1.39, c='k', lw=0.5)
ax_grad.axhline(1, c=brewer_purple, lw=1.5, label=r'$\nabla_{\rm{ad}}$')
ax_grad.plot(z, grad_rad/grad_ad, c=brewer_orange, label=r'$\nabla_{\rm{rad}}$', lw=1.5)
ax_grad.plot(z, grad/grad_ad, c=brewer_green, label=r'$\nabla$', lw=2)
ax_grad.legend(fontsize=10, loc='lower left')
ax_grad.set_ylim(0.75, 1.2)
ax_grad.set_ylabel(r'$\nabla/\nabla_{\rm{ad}}$')
ax_grad.text(0.5, 1.1, 'CZ', ha='center', va='center')
ax_grad.text(1.044 + (0.39-0.044)/2, 1.1, 'PZ', ha='center', va='center')
ax_grad.text(1.39 + 0.305, 1.1, 'RZ', ha='center', va='center')


ax_source.axhline(0, c='k', lw=0.5)
ax_source.plot(z, dissipation/0.2, c=brewer_purple, label='$\overline{\Phi}$', lw=2)
ax_source.plot(z, fconv/0.2, c=brewer_orange, label=r'$\overline{\mathcal{B}}$', lw=2)
#ax_source.plot(z, ke_eqn_rhs/0.2, c='blue', label=r'$\overline{\mathcal{B}} - \overline{\Phi}$')
ax_source.legend(loc='best', fontsize=10)
ax_source.set_ylim(-0.5, 1.25)
ax_source.set_ylabel('KE sources', labelpad=0)
ax_source.axvline(1.044, c='k', lw=0.5, ls='--')
ax_source.axvline(1.39, c='k', lw=0.5)

f_ke_p = f_ke_profiles[N,:]

ax_flux.axhline(0, c='k', lw=0.5)
ax_flux.plot(z, f_ke_p/0.2, c=brewer_green, lw=2)
#ax_flux.axvline(brentq(interp1d(z, f_ke_p), 0.05, 0.9), c='k', ls='--')
#ax_flux.axvline(1.42, c='k')
ax_flux.set_ylabel(r'$\overline{\mathcal{F}}$')
ax_flux.axvline(1.044, c='k', lw=0.5, ls='--')
ax_flux.axvline(1.39, c='k', lw=0.5)


grad = grad_profiles[N,:]
for ax in [ax_source, ax_flux, ax_grad]:
    ax.set_xlim(0, Lz)
    ax.set_xlabel(r'$z$')

ax_flux.set_xticklabels(())

fig_grad.savefig('grad_profiles.png', dpi=300, bbox_inches='tight')
fig_grad.savefig('../manuscript/grad_profiles.pdf', dpi=300, bbox_inches='tight')
fig_theory.savefig('theory_profiles.png', dpi=300, bbox_inches='tight')
fig_theory.savefig('../manuscript/theory_profiles.pdf', dpi=300, bbox_inches='tight')
