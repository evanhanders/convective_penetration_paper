import matplotlib
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import h5py

import dedalus.public as de
from plotpal.file_reader import SingleFiletypePlotter as SFP


# Get data from files
start_file = 1
n_files = np.inf
root_dir = '../noslip_erf_Re_cut/erf_step_3D_Re3.2e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_256x256x256_restart/'
plotter_slices = SFP(root_dir, file_dir='slices', fig_name='talk_snapshots', start_file=start_file, n_files=n_files, distribution='even')
plotter_profiles = SFP(root_dir, file_dir='profiles', fig_name='talk_snapshots', start_file=start_file, n_files=n_files, distribution='even')

with h5py.File('{:s}/top_cz/data_top_cz.h5'.format(root_dir), 'r') as f:
    Ls = f['Ls'][()]
    L_d09s = f['L_d09s'][()]

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
nx = ny = nz = 256

# Dedalus domain for interpolation
x_basis = de.Fourier('x',   nx, interval=[0, Lx], dealias=1)
z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
vert_domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
hires_scales = 4

vert_field  = vert_domain.new_field()

#x_shift = 1
x = vert_domain.grid(0, scales=hires_scales)
#x -= x_shift
#x[x < 0] += Lx
count = 0
zz_tb, xx_tb = np.meshgrid(vert_domain.grid(-1, scales=hires_scales), x)
if not plotter_slices.idle:
    prof_fields = ['T1', 'T1_fluc']
    plotter_profiles.set_read_fields(['z'], prof_fields)
    while plotter_slices.files_remain(['x', 'z'], ['T1_y_mid', 'w_y_mid']):
        bases, tasks, write_num, sim_time = plotter_slices.read_next_file()
        pbases, ptasks, pwrite_num, psim_time = plotter_profiles.read_next_file()

        for i in range(sim_time.shape[0]):
            print('writing {}'.format(i))
            #Temp plots
            vert_field['g'] = tasks['T1_y_mid'][i,:].squeeze()
            vert_field['g'] -= ptasks['T1'][i,0,0,:][None,:]
            vert_field['g'] /= ptasks['T1_fluc'][i,0,0,:][None,:]
            vert_field.set_scales(hires_scales, keep_data=True)
            T_cbar_scale = 2.5#np.max(np.abs(vert_field['g']))/3
            pT = axs[1].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='RdBu_r', rasterized=True, shading="nearest", vmin=-T_cbar_scale, vmax=T_cbar_scale)
            vert_field.set_scales(1)

            #Vel plots
            vert_field['g'] = tasks['w_y_mid'][i,:].squeeze()
            vert_field.set_scales(hires_scales, keep_data=True)
            w_cbar_scale = 1#np.max(np.abs(vert_field['g']))/1.5
            pW = axs[0].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='PuOr_r', rasterized=True, shading="nearest", vmin=-w_cbar_scale, vmax=w_cbar_scale)
            vert_field.set_scales(1)

            cbar_w = plt.colorbar(pW, cax=caxs[0], orientation='horizontal')
            cbar_T = plt.colorbar(pT, cax=caxs[1], orientation='horizontal')

            caxs[0].text(0.5, 0.5, r"$w$", transform=caxs[0].transAxes, va='center', ha='center', fontsize=10)
            caxs[1].text(0.5, 0.4, r"$T' / \overline{|T'|}$", transform=caxs[1].transAxes, va='center', ha='center', fontsize=8)

            for ind in [0, 1]:
                ax = axs[ind]
                ax.axhline(Ls, lw=0.5, ls='--', color='xkcd:dark grey')
                ax.axhline(L_d09s[count],  color='dimgrey')
            count += 1


            for ax in axs:
                ax.set_xticks(())
                ax.set_yticks(())
                ax.set_xlim(0, Lx)

            fig.savefig('{:s}/vertical_dynamics_panels_{:06d}.png'.format(plotter_slices.out_dir, write_num[i]), dpi=300, bbox_inches='tight')

            for ax in axs:
                ax.clear()
            for ax in caxs:
                ax.clear()
