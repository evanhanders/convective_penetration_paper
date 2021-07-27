import matplotlib
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import h5py

import dedalus.public as de
plt.style.use('apj')

read_raw_files = False
if read_raw_files:
    erf_plot_ind = 12 
    linear_plot_ind = 16
    # Get data from files
    linear_dynamics_file = 'turbulence_slices/linear_3D_Re8e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_128x128x256/slices_s49.h5'
    linear_profile_file  = 'turbulence_slices/linear_3D_Re8e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_128x128x256/profiles_s49.h5'
    erf_dynamics_file    = 'turbulence_slices/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart6/slices_s234.h5'
    erf_profile_file     = 'turbulence_slices/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart6/profiles_s234.h5'

    with h5py.File(linear_profile_file, 'r') as f:
        linear_z = f['scales/z/1.0'][()].squeeze()
        linear_mean_T1 = np.mean(f['tasks/T1'][()][:,:].squeeze(), axis=0)
        linear_T1_fluc = np.mean(f['tasks/T1_fluc'][()][:,:].squeeze(), axis=0)
        grad = -np.mean(f['tasks/T_z'][()][:,:].squeeze(), axis=0)
        grad_ad = -np.mean(f['tasks/T_ad_z'][()][:,:].squeeze(), axis=0)
        grad_rad = -np.mean(f['tasks/T_rad_z'][()][:,:].squeeze(), axis=0)

        departure_frac = 0.5
        departed_points = (linear_z > 1)*(grad > grad_ad - departure_frac * (grad_ad - grad_rad))
        linear_departure_point = linear_z[departed_points][-1]

    with h5py.File(erf_profile_file, 'r') as f:
        erf_z = f['scales/z/1.0'][()].squeeze()
        erf_mean_T1 = np.mean(f['tasks/T1'][()][:,:].squeeze(), axis=0)
        erf_T1_fluc = np.mean(f['tasks/T1_fluc'][()][:,:].squeeze(), axis=0)
        grad = -np.mean(f['tasks/T_z'][()][:,:].squeeze(), axis=0)
        grad_ad = -np.mean(f['tasks/T_ad_z'][()][:,:].squeeze(), axis=0)
        grad_rad = -np.mean(f['tasks/T_rad_z'][()][:,:].squeeze(), axis=0)

        departure_frac = 0.5
        departed_points = (erf_z > 1)*(grad > grad_ad - departure_frac * (grad_ad - grad_rad))
        erf_departure_point = erf_z[departed_points][-1]


    with h5py.File('turbulence_slices/linear_erf_lessturb_plot_data.h5', 'w') as f:
        f['linear_z'] = linear_z
        f['linear_mean_T1'] = linear_mean_T1
        f['linear_T1_fluc'] = linear_T1_fluc
        f['linear_departure_point'] = linear_departure_point
        with h5py.File(linear_dynamics_file, 'r') as df:
            f['linear_T1_field'] = df['tasks']['T1_y_mid'][linear_plot_ind,:].squeeze()
        f['erf_z'] = erf_z
        f['erf_mean_T1'] = erf_mean_T1
        f['erf_T1_fluc'] = erf_T1_fluc
        f['erf_departure_point'] = erf_departure_point
        with h5py.File(erf_dynamics_file, 'r') as df:
            f['erf_T1_field'] = df['tasks']['T1_y_mid'][erf_plot_ind,:].squeeze()


with h5py.File('turbulence_slices/linear_erf_lessturb_plot_data.h5', 'r') as f:
    linear_z               = f['linear_z'][()]
    linear_mean_T1         = f['linear_mean_T1'][()]
    linear_T1_fluc         = f['linear_T1_fluc'][()]
    linear_departure_point = f['linear_departure_point'][()]
    linear_T1_field        = f['linear_T1_field'][()]
    erf_z               = f['erf_z'][()]
    erf_mean_T1         = f['erf_mean_T1'][()]
    erf_T1_fluc         = f['erf_T1_fluc'][()]
    erf_departure_point = f['erf_departure_point'][()]
    erf_T1_field        = f['erf_T1_field'][()]


# Set up figure subplots
fig = plt.figure(figsize=(3.25, 4.25))
cax_specs = [( 0,   0.94,   1.00,   0.06)]
ax_specs = [
            ( 0,   0.45,    1.00,   0.35),
            ( 0,   0.00,    1.00,   0.35),
            ]
axs  = [fig.add_axes(spec) for spec in ax_specs]
caxs = [fig.add_axes(spec) for spec in cax_specs]

Lz = 2
Lx = Ly = 2*Lz
nz = 256
linear_nx = 128
erf_nx    = 64

# Dedalus domain for interpolation
linear_x_basis = de.Fourier('x',   linear_nx, interval=[0, Lx], dealias=1)
erf_x_basis    = de.Fourier('x',   erf_nx, interval=[0, Lx], dealias=1)
z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
linear_vert_domain = de.Domain([linear_x_basis, z_basis], grid_dtype=np.float64)
erf_vert_domain    = de.Domain([erf_x_basis, z_basis], grid_dtype=np.float64)
hires_scales = 4

linear_vert_field  = linear_vert_domain.new_field()
erf_vert_field  = erf_vert_domain.new_field()

l_zz, l_xx = np.meshgrid(linear_vert_domain.grid(-1, scales=hires_scales), linear_vert_domain.grid(0, scales=hires_scales) )
e_zz, e_xx = np.meshgrid(erf_vert_domain.grid(-1, scales=hires_scales),    erf_vert_domain.grid(0, scales=hires_scales)    )

T_cbar_scale = 3.0
linear_vert_field['g'] = linear_T1_field
linear_vert_field['g'] -= linear_mean_T1
linear_vert_field['g'] /= linear_T1_fluc 
linear_vert_field.set_scales(hires_scales, keep_data=True)
pT = axs[1].pcolormesh(l_xx,  l_zz,  np.copy(linear_vert_field['g']),  cmap='RdBu_r', rasterized=True, shading="nearest", vmin=-T_cbar_scale, vmax=T_cbar_scale)
cbar_T = plt.colorbar(pT, cax=caxs[0], orientation='horizontal')



erf_vert_field['g'] = erf_T1_field
erf_vert_field['g'] -= erf_mean_T1
erf_vert_field['g'] /= erf_T1_fluc 
erf_vert_field.set_scales(hires_scales, keep_data=True)
pT = axs[0].pcolormesh(e_xx,  e_zz,  np.copy(erf_vert_field['g']),  cmap='RdBu_r', rasterized=True, shading="nearest", vmin=-T_cbar_scale, vmax=T_cbar_scale)

caxs[0].text(0.5, 0.4, r"$T' / \overline{|T'|}$", transform=caxs[0].transAxes, va='center', ha='center', fontsize=10)

for ind in [0, 1]:
    ax = axs[ind]
axs[0].axhline(1.04, lw=0.5, ls='--', color='xkcd:dark grey')
axs[1].axhline(0.1, lw=0.5, ls='--', color='xkcd:dark grey')
axs[1].axhline(1, lw=0.5, ls='--', color='xkcd:dark grey')
axs[0].axhline(erf_departure_point,  color='dimgrey')
axs[1].axhline(linear_departure_point,  color='dimgrey')

axs[0].text(0.5, 1.1, r'Case I, $\mathcal{R} = 400$, $\mathcal{P}_D = 4$, $\mathcal{S} = 10^3$', ha='center', va='center', fontsize=10, transform=axs[0].transAxes)
axs[1].text(0.5, 1.1, r'Case II, $\mathcal{R} = 800$, $\mathcal{P}_L = 4$, $\mathcal{S} = 10^3$', ha='center', va='center', fontsize=10, transform=axs[1].transAxes)


for ax in axs:
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(0, Lx)

fig.savefig('lessturb_dynamics_panels.png', dpi=300, bbox_inches='tight')
fig.savefig('../manuscript/lessturb_dynamics_panels.pdf', dpi=300, bbox_inches='tight')
