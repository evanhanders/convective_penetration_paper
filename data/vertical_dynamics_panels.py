import matplotlib
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import h5py

import dedalus.public as de
plt.style.use('apj')


# Get data from files
dynamics_file = 'turbulence_slices/erf_step_3D_Re6.4e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_384x384x384/slices_s9.h5'
profile_file  = 'turbulence_slices/erf_step_3D_Re6.4e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_384x384x384/profiles_s9.h5'
#dynamics_file = 'turbulence_slices/erf_step_3D_Re3.2e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_256x256x256_predictive0.46/slices/slices_s5.h5'
#profile_file = 'turbulence_slices/erf_step_3D_Re3.2e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_256x256x256_predictive0.46/profiles/profiles_s5.h5'

with h5py.File(profile_file, 'r') as f:
    z = f['scales/z/1.0'][()].squeeze()
    mean_T1 = np.mean(f['tasks/T1'][()][:,:].squeeze(), axis=0)
    T1_fluc = np.mean(f['tasks/T1_fluc'][()][:,:].squeeze(), axis=0)
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
fig = plt.figure(figsize=(6.5, 2))
cax_specs = [( 0,   0.92,   0.47,   0.08),
             ( 0.53, 0.92,   0.47,   0.08)]
ax_specs = [
            ( 0,   0,       0.47,   0.8),
            ( 0.53, 0,      0.47,   0.8),
            ]
axs  = [fig.add_axes(spec) for spec in ax_specs]
caxs = [fig.add_axes(spec) for spec in cax_specs]

Lz = 2
Lx = Ly = 2*Lz
nx = ny = nz = 384

# Dedalus domain for interpolation
x_basis = de.Fourier('x',   nx, interval=[0, Lx], dealias=1)
y_basis = de.Fourier('y',   ny, interval=[0, Ly], dealias=1)
z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
horiz_domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
hires_scales = 4

horiz_field = horiz_domain.new_field()
vert_field  = vert_domain.new_field()


#x_shift = 1
x = vert_domain.grid(0, scales=hires_scales)
#x -= x_shift
#x[x < 0] += Lx

zz_tb, xx_tb = np.meshgrid(vert_domain.grid(-1, scales=hires_scales), x)



plot_ind = -2
with h5py.File(dynamics_file, 'r') as f:
    #Temp plots
    vert_field['g'] = f['tasks']['T1_y_mid'][plot_ind,:].squeeze()
    vert_field['g'] -= mean_T1
    vert_field['g'] /= T1_fluc 
    vert_field.set_scales(hires_scales, keep_data=True)
    T_cbar_scale = np.max(np.abs(vert_field['g']))/3
    pT = axs[1].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='RdBu_r', rasterized=True, shading="nearest", vmin=-T_cbar_scale, vmax=T_cbar_scale)
    vert_field.set_scales(1)

    #Vel plots
    vert_field['g'] = f['tasks']['w_y_mid'][plot_ind,:].squeeze()
    vert_field.set_scales(hires_scales, keep_data=True)
    w_cbar_scale = np.max(np.abs(vert_field['g']))/1.5
    pW = axs[0].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='PuOr_r', rasterized=True, shading="nearest", vmin=-w_cbar_scale, vmax=w_cbar_scale)
    vert_field.set_scales(1)

cbar_w = plt.colorbar(pW, cax=caxs[0], orientation='horizontal')
cbar_T = plt.colorbar(pT, cax=caxs[1], orientation='horizontal')

caxs[0].text(0.5, 0.5, r"$w$", transform=caxs[0].transAxes, va='center', ha='center', fontsize=10)
caxs[1].text(0.5, 0.4, r"$T' / \overline{|T'|}$", transform=caxs[1].transAxes, va='center', ha='center', fontsize=8)

for ind in [0, 1]:
    ax = axs[ind]
    ax.axhline(1, lw=0.5, ls='--', color='xkcd:dark grey')
    ax.axhline(departure_point,  color='dimgrey')


for ax in axs:
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(0, Lx)


fig.savefig('vertical_dynamics_panels.png', dpi=300, bbox_inches='tight')
fig.savefig('../manuscript/vertical_dynamics_panels.pdf', dpi=300, bbox_inches='tight')
