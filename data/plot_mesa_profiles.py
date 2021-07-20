from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from scipy.optimize import brentq
from scipy.interpolate import interp1d

import mesa_reader as mr
from astropy import units as u
from astropy import constants

col_width = 3.25
golden_ratio = 1.61803398875

brewer_green  = np.array(( 27,158,119, 255))/255
brewer_orange = np.array((217, 95,  2, 255))/255
brewer_purple = np.array((117,112,179, 255))/255

penetration_run_file = 'MESA/PZ.data'#mesa_penetration_profile67.data'
standard_run_file = 'MESA/no_pz.data'#mesa_penetration_profile67.data'
pf = mr.MesaData(penetration_run_file)
sf = mr.MesaData(standard_run_file)

fig_mesa = plt.figure(figsize=(col_width, 1.5*col_width/golden_ratio))

ax_grad   = fig_mesa.add_axes([0.00, 0.50, 1, 0.5])
ax_cs = fig_mesa.add_axes([0.00, 0.00, 1, 0.5])

pz_mass = pf.mass[::-1] * constants.M_sun
pz_radius = pf.radius[::-1] * constants.R_sun
pz_rho    = 10**(pf.logRho[::-1]) * (u.g/u.cm**3)
pz_P      = 10**(pf.logP[::-1]) * (u.g / u.cm / u.s**2)
gravity   = constants.G.cgs*pz_mass/pz_radius**2
gradP     = pz_rho*gravity
grad_ln_P = gradP/pz_P
HP        = 1/grad_ln_P
penetration = pf.penetration_fraction[::-1]

good_pz = (penetration == 1)*(pz_radius.cgs/constants.R_sun.cgs < 0.85)

r    = pf.radius[::-1]
grad = pf.gradT[::-1]
grad_rad = pf.gradr[::-1]
grad_ad  = pf.grada[::-1]
cs_pz = pf.csound[::-1]


boundary = grad_rad - grad_ad > 0
r_schwarz = r[boundary][0]
r_bot_pz  = r[good_pz][0]
HP_bot = HP[r == r_schwarz]
r_pz   = r_schwarz - r_bot_pz
#ax_grad.axvline(r_schwarz)
#ax_grad.axvline(r_bot_pz)

print(r_pz, HP_bot.cgs/constants.R_sun.cgs)


r_standard  = sf.radius[::-1]
cs_standard = sf.csound[::-1]
cs_standard_func = interp1d(r_standard, cs_standard, fill_value='extrapolate', bounds_error=False)

#ax_grad.axvline(1.044, c='k', lw=0.5, ls='--')
#ax_grad.axvline(1.39, c='k', lw=0.5)
ax_grad.plot(r, grad_ad, c=brewer_purple, lw=1.5, label=r'$\nabla_{\rm{ad}}$')
ax_grad.plot(r, grad_rad, c=brewer_orange, label=r'$\nabla_{\rm{rad}}$', lw=1.5)
ax_grad.plot(r, grad, c=brewer_green, label=r'$\nabla$', lw=2)
ax_grad.legend(fontsize=10, loc='upper left')
ax_grad.set_ylabel(r'$\nabla$')
ax_grad.set_ylim(0, 1)



#ax_cs.plot(r, 1 - cs_pz/cs_standard_func(r), c=brewer_purple, lw=2)
ax_cs.plot(r, 1 - cs_standard_func(r)/cs_pz, c=brewer_purple, lw=2)
#ax_cs.set_ylabel(r'$(c_{\rm{std}} - c_{\rm{PZ}})/c_{\rm{std}}$')
ax_cs.set_ylabel(r'$(c_{\rm{PZ}} - c_{\rm{std}})/c_{\rm{PZ}}$')
#ax_cs.set_ylim(-2e-2, 2e-3)
ax_cs.set_ylim(0, 2e-2)
#ax_cs.set_yticks((-0.015, -0.01, -0.005, 0))
ax_cs.set_yticks((0, 0.005, 0.01, 0.015))

for ax in [ax_cs, ax_grad]:
    ax.set_xlim(0.65, 0.8)

ax_cs.set_xlabel('$r (R_{\odot})$')
ax_grad.set_xticklabels(())


fig_mesa.savefig('mesa_profiles.png', dpi=300, bbox_inches='tight')
fig_mesa.savefig('../manuscript/mesa_profiles.pdf', dpi=300, bbox_inches='tight')
