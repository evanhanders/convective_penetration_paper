import matplotlib
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import h5py

import dedalus.public as de
plt.style.use('apj')


# Get data from files
dynamics_file = 'turbulence_slices/erf_step_3D_Re3.2e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_256x256x256_predictive0.46/slices/slices_s5.h5'
profile_file = 'turbulence_slices/erf_step_3D_Re3.2e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_256x256x256_predictive0.46/profiles/profiles_s5.h5'

with h5py.File(profile_file, 'r') as f:
    z = f['scales/z/1.0'][()].squeeze()
    mean_T1 = np.mean(f['tasks/T1'][()][:,:].squeeze(), axis=0)
    grad = -np.mean(f['tasks/T_z'][()][:,:].squeeze(), axis=0)
    grad_ad = -np.mean(f['tasks/T_ad_z'][()][:,:].squeeze(), axis=0)
    grad_rad = -np.mean(f['tasks/T_rad_z'][()][:,:].squeeze(), axis=0)

    departure_frac = 0.5
    departed_points = (z > 1)*(grad > grad_ad - departure_frac * (grad_ad - grad_rad))
    departure_point = z[departed_points][-1]
#    plt.plot(z, grad_ad)
#    plt.plot(z, grad_rad)
#    plt.plot(z, grad)
#    plt.show()


# Set up figure subplots
fig = plt.figure(figsize=(3.5, 6.5))
cax_specs = [( 0,   0.95,   0.47,   0.04),
             ( 0.53, 0.95,   0.47,   0.04)]
ax_specs = [( 0,   0.78,    0.47,   0.12),
            ( 0,   0.52,    0.47,   0.25),
            ( 0,   0.26,    0.47,   0.25),
            ( 0,   0,       0.47,   0.25),
            ( 0.53, 0.78,   0.47,   0.12),
            ( 0.53, 0.52,   0.47,   0.25),
            ( 0.53, 0.26,   0.47,   0.25),
            ( 0.53, 0,      0.47,   0.25),
            ]
axs  = [fig.add_axes(spec) for spec in ax_specs]
caxs = [fig.add_axes(spec) for spec in cax_specs]

Lz = 2
Lx = Ly = 2*Lz
nx = ny = nz = 256

# Dedalus domain for interpolation
x_basis = de.Fourier('x',   nx, interval=[0, Lx], dealias=1)
y_basis = de.Fourier('y',   ny, interval=[0, Ly], dealias=1)
z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
horiz_domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
hires_scales = 4

horiz_field = horiz_domain.new_field()
vert_field  = vert_domain.new_field()

zz_tb, xx_tb = np.meshgrid(vert_domain.grid(-1, scales=hires_scales), vert_domain.grid(0, scales=hires_scales))
yy_mid, xx_mid = np.meshgrid(horiz_domain.grid(-1, scales=hires_scales), horiz_domain.grid(0, scales=hires_scales))

plot_ind = 5
with h5py.File(dynamics_file, 'r') as f:
    #Temp plots
    vert_field['g'] = f['tasks']['T1_y_mid'][plot_ind,:].squeeze()
    vert_field['g'] -= mean_T1
    vert_field.set_scales(hires_scales, keep_data=True)
    T_cbar_scale = np.max(np.abs(vert_field['g']))/10
    pT = axs[0].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='RdBu_r', rasterized=True, shading="nearest", vmin=-T_cbar_scale, vmax=T_cbar_scale)
    vert_field.set_scales(1)

    for ind, zv in zip((1, 2, 3),(1.2, 1, 0.5)):
        horiz_field['g'] = f['tasks']['T1_z_{}'.format(zv)][plot_ind,:].squeeze()
        horiz_field['g'] -= np.mean(horiz_field['g'])
        horiz_field.set_scales(hires_scales, keep_data=True)
        axs[ind].pcolormesh(xx_mid,  yy_mid,  np.copy(horiz_field['g']),  cmap='RdBu_r', rasterized=True, shading="nearest", vmin=-T_cbar_scale, vmax=T_cbar_scale)
        horiz_field.set_scales(1)
        axs[ind].axhline(Lx/2, c='k', lw=0.5)

    #Vel plots
    vert_field['g'] = f['tasks']['w_y_mid'][plot_ind,:].squeeze()
    vert_field.set_scales(hires_scales, keep_data=True)
    w_cbar_scale = np.max(np.abs(vert_field['g']))
    pW = axs[4].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='PuOr_r', rasterized=True, shading="nearest", vmin=-w_cbar_scale, vmax=w_cbar_scale)
    vert_field.set_scales(1)

    for ind, zv in zip((5, 6, 7),(1.2, 1, 0.5)):
        horiz_field['g'] = f['tasks']['w_z_{}'.format(zv)][plot_ind,:].squeeze()
        horiz_field['g'] -= np.mean(horiz_field['g'])
        horiz_field.set_scales(hires_scales, keep_data=True)
        axs[ind].pcolormesh(xx_mid,  yy_mid,  np.copy(horiz_field['g']),  cmap='PuOr_r', rasterized=True, shading="nearest", vmin=-w_cbar_scale, vmax=w_cbar_scale)
        horiz_field.set_scales(1)
        axs[ind].axhline(Lx/2, c='k', lw=0.5)

cbar_T = plt.colorbar(pT, cax=caxs[0], orientation='horizontal')
cbar_w = plt.colorbar(pW, cax=caxs[1], orientation='horizontal')

caxs[0].text(0.5, 0.5, r"$T'$", transform=caxs[0].transAxes, va='center', ha='center')
caxs[1].text(0.5, 0.5, r"$w$", transform=caxs[1].transAxes, va='center', ha='center')

for ind in [0, 4]:
    ax = axs[ind]
    ax.axhline(1, c='k')
    ax.axhline(departure_point, c='k', ls='--')


for ax in axs:
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(0, Lx)


fig.savefig('turbulent_dynamics_panels.png', dpi=300, bbox_inches='tight')