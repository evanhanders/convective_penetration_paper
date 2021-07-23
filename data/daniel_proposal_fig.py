import matplotlib
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import h5py

import glob

import dedalus.public as de
plt.style.use('apj')


# Get data from files
dynamics_file = 'turbulence_slices/erf_step_3D_Re6.4e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_384x384x384/slices_s8.h5'
profile_file  = 'turbulence_slices/erf_step_3D_Re6.4e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_384x384x384/profiles_s8.h5'
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
fig = plt.figure(figsize=(6.5, 1.5))
cax_specs = [( 0,   0.8,   0.27,   0.08),
             ]
ax_specs = [
            ( 0,    0,       0.27,   0.7),
            ( 0.37, 0,      0.27,   1),
            ( 0.73, 0,      0.27,   1),
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
    #Vel plots
    vert_field['g'] = f['tasks']['w_y_mid'][plot_ind,:].squeeze()
    vert_field.set_scales(hires_scales, keep_data=True)
    w_cbar_scale = np.max(np.abs(vert_field['g']))/1.5
    pW = axs[0].pcolormesh(xx_tb,  zz_tb,  np.copy(vert_field['g']),  cmap='PuOr_r', rasterized=True, shading="nearest", vmin=-w_cbar_scale, vmax=w_cbar_scale)
    vert_field.set_scales(1)

cbar_w = plt.colorbar(pW, cax=caxs[0], orientation='horizontal')

caxs[0].text(0.5, -0.7, r"$w$", transform=caxs[0].transAxes, va='center', ha='center', fontsize=10)
caxs[0].xaxis.set_ticks_position('top')
#caxs[0].set_label(r'$w$')

axs[0].set_xticks(())
axs[0].set_yticks(())
axs[0].set_xlim(0, Lx)
axs[0].set_xlabel('x')
axs[0].set_ylabel('z')
axs[0].axhline(departure_point,  color='dimgrey')

#Panel 2

d = directory = 'new_erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.4/'
#d = directory = 'new_erf_Re_cut/erf_step_3D_Re6.4e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_384x384x384/'
filename = directory + '/avg_profs/averaged_avg_profs.h5'

S = float(d.split('_S')[-1].split('_')[0])
P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
Re = float(d.split('_Re')[-1].split('_')[0])
Lz = float(d.split('_Lz')[-1].split('_')[0])

with h5py.File(filename, 'r') as f:
    grad_profiles = -f['T_z'][()]
    grad_ad_profiles = -f['T_ad_z'][()]
    grad_rad_profiles = -f['T_rad_z'][()]
    z = f['z'][()]



N = -1
grad = grad_profiles[N,:]
grad_ad = grad_ad_profiles[N,:]
grad_rad = grad_rad_profiles[N,:]
departure_frac = 0.5
departed_points = (z > 1)*(grad > grad_ad - departure_frac * (grad_ad - grad_rad))
departure_point = z[z > z[departed_points][-1]][0]


axs[1].plot(z, grad_ad/grad_ad, c='k', label=r'$\nabla_{\rm{ad}}$')
axs[1].plot(z, grad_rad/grad_ad, c='red', label=r'$\nabla_{\rm{rad}}$')
axs[1].plot(z, grad/grad_ad, c='green', label=r'$\nabla$')
axs[1].legend(loc='best', fontsize=8, borderaxespad=0.25)
axs[1].set_ylim(0.75, 1.2)
#axs[1].set_ylim(0.75*grad_ad.min(), 1.2*grad_ad.min())
axs[1].set_ylabel(r'$\nabla/\nabla_{\rm{ad}}$')
axs[1].axvline(departure_point,  color='dimgrey')


axs[1].set_xlim(0, Lz)
axs[1].set_xlabel('z')


axs[1].text(0.28, 0.9, r"CZ", transform=axs[1].transAxes, va='center', ha='center', fontsize=10)
axs[1].text(0.63, 0.9, r"PZ", transform=axs[1].transAxes, va='center', ha='center', fontsize=10)
axs[1].text(0.88, 0.9, r"RZ", transform=axs[1].transAxes, va='center', ha='center', fontsize=10)


# Panel 3
dirs = glob.glob('new_erf*/erf*')

data = []
for d in dirs:
    S = float(d.split('_S')[-1].split('_')[0])
    P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
    Re = float(d.split('_Re')[-1].split('_')[0])
    Lz = float(d.split('_Lz')[-1].split('_')[0])
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        theory_f = f['f_theory_cz'][()]
        z2 = f['z2'][()]
        Ls = np.mean(f['Ls'][()])
        tot_time = times[-1] - times[0]
        time_window = np.min((500, tot_time/2))
        if Re == 6.4e3:
            good_times = times < 170
        else:
            good_times = times > (times[-1] - time_window)
        mean_L_d01 = np.mean(L_d01s[good_times])
        mean_L_d05 = np.mean(L_d05s[good_times])
        mean_L_d09 = np.mean(L_d09s[good_times])
        mean_f = np.mean(theory_f[good_times])
        mean_z2 = np.mean(z2[good_times])
        if Re == 6.4e3:
            print(mean_L_d01 - Ls, mean_L_d05 - Ls, mean_L_d09 - Ls)
    if '3D' in d:
        threeD = True
    else:
        threeD = False
    if 'erf' in d:
        erf = True
    else:
        erf = False
    if 'AE' in d:
        ae = True
    else:
        ae = False

    data.append((S, P, Re, threeD, mean_L_d01, mean_L_d05, mean_L_d09, mean_f, Ls, Lz, erf, ae, mean_z2))
#    print(d)
#    print("                 ",S, P, Re, mean_L_d01, mean_L_d05, mean_L_d09, mean_f)
data = np.array(data)

S = data[:,0]
P = data[:,1]
Re = data[:,2]
threeD = data[:,3]
L_d01 = data[:,4]
L_d05 = data[:,5]
L_d09 = data[:,6]
theory_f = data[:,7]
Ls = data[:,8]
Lz = data[:,9]
erf = data[:,10]
ae = data[:,11]
z2 = data[:,12]
Lcz_norm = Ls# - z2
axs[1].axvline(np.mean(Lcz_norm), lw=0.5, ls='--', color='xkcd:dark grey')
axs[0].axhline(np.mean(Lcz_norm), lw=0.5, ls='--', color='xkcd:dark grey')

good = (erf == 1) * (P == 4) * (Re == 4e2) * (ae == 0) * (S < 1e4)
#axs[2].scatter(S[good], (L_d01[good] - Ls[good])/Lcz_norm[good], c='k', label=r"Simulations ($\delta_p$)", zorder=1, marker='^')
#axs[2].scatter(S[good], (L_d09[good] - Ls[good])/Lcz_norm[good], c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=1, marker='v')
axs[2].scatter(S[good], (L_d05[good] - Ls[good])/Lcz_norm[good], c='k', zorder=1, marker='o')
axs[2].set_xscale('log')
#axs[2].set_title('$\mathcal{S}|_{\mathcal{P} = 4, \mathcal{R} = 400}$')
axs[2].set_ylim(0, 0.6)
axs[2].set_xlabel('stiffness')
axs[2].set_ylabel(r'$\delta_P/L_{\rm{CZ}}$')
axs[2].set_yticks((0, 0.2, 0.4, 0.6))



fig.savefig('proposal_panels.png', dpi=300, bbox_inches='tight')
fig.savefig('../manuscript/proposal_panels.pdf', dpi=300, bbox_inches='tight')
