from collections import OrderedDict
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from scipy.interpolate import interp1d
import pandas as pd


col_width = 3.25
golden_ratio = 1.61803398875

dirs = ["erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart2",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart3",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart4"]

fig = plt.figure(figsize=(2*col_width, col_width/golden_ratio))
ax1 = fig.add_axes([0,    0.67, 0.45, 0.33])
ax2 = fig.add_axes([0,    0.34, 0.45, 0.33])
ax3 = fig.add_axes([0,    0.00, 0.45, 0.34])
ax4 = fig.add_axes([0.50,    0, 0.45, 1])


data = OrderedDict()
S = float(dirs[0].split('_S')[-1].split('_')[0])
P = float(dirs[0].split('_Pr')[0].split('_P')[-1].split('_')[0])
Re = float(dirs[0].split('_Re')[-1].split('_')[0])
Lz = float(dirs[0].split('_Lz')[-1].split('_')[0])

k_rz = 0.2 / (S * P)
t_diff = 1/k_rz

if '3D' in dirs[0]:
    threeD = True
else:
    threeD = False
if 'erf' in dirs[0]:
    erf = True
else:
    erf = False

times = []
L_d01s = []
L_d05s = []
L_d09s = []
grad_above = []
grad_rad_above = []
theory_f_enstrophy = []
theory_f_flux = []
enstrophy_Ls_cz = []
grad = []
prof_times = []
for i, d in enumerate(dirs):
    if i == 0:
        skip_points = 100
    else:   
        skip_points = 0
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times.append(f['times'][()][skip_points:])
        L_d01s.append(f['L_d01s'][()][skip_points:])
        L_d05s.append(f['L_d05s'][()][skip_points:])
        L_d09s.append(f['L_d09s'][()][skip_points:])
        grad_above.append(f['grad_above'][()][skip_points:])
        grad_rad_above.append(f['grad_rad_above'][()][skip_points:])
        theory_f_enstrophy.append(f['f_theory_enstrophy'][()][skip_points:])
        theory_f_flux.append(f['f_theory_flux'][()][skip_points:])
        Ls = f['Ls'][()]
        enstrophy_Ls_cz.append(f['enstrophy_Ls'][()][skip_points:])
    with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
        z = f['z'][()]
        grad_ad = -f['T_ad_z'][()][0,:].squeeze()
        grad_rad = -f['T_rad_z'][()][0,:].squeeze()

        grad.append(-f['T_z'][()])
        prof_times.append(f['T_z_times'][()])



Lcz_norm = Ls - 0.2
times          = np.array(np.concatenate(times))
L_d01s         = np.array(np.concatenate(L_d01s))
L_d05s         = np.array(np.concatenate(L_d05s))
L_d09s         = np.array(np.concatenate(L_d09s))
grad_above     = np.array(np.concatenate(grad_above))
grad_rad_above = np.array(np.concatenate(grad_rad_above))
grad           = np.array(np.concatenate(grad, axis=0))
prof_times     = np.array(np.concatenate(prof_times))
theory_f_enstrophy     = np.array(np.concatenate(theory_f_enstrophy))
theory_f_flux          = np.array(np.concatenate(theory_f_flux))
enstrophy_Ls_cz = np.array(np.concatenate(enstrophy_Ls_cz))


f_data = dict()
f_data['sim_time'] = times
f_data['enstrophy'] = enstrophy_Ls_cz
f_df = pd.DataFrame(data=f_data)
rolledf = f_df.rolling(window=50, min_periods=25).mean()

final_f = np.mean(enstrophy_Ls_cz[-500:])
error_f = np.abs(1 - rolledf['enstrophy']/final_f)
time_cross_f = times[error_f < 0.01][0]

delta_d05 = (L_d05s - Ls)/Lcz_norm
final_d05 = np.mean(delta_d05[-500:])
error_d05 = np.abs(1 - (delta_d05/final_d05))
time_cross_d05 = times[error_d05 < 0.01][0]


cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=times.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

colors = [sm.to_rgba(t) for t in times]
for i in range(int(len(times)/10)-1):
#    ax1.plot(times[i*10:(i+1)*10], (L_d01s[i*10:(i+1)*10]-Ls)/Lcz_norm, c=sm.to_rgba(times[int((i+0.5)*10)]))
    ax1.plot(times[i*10:(i+1)*10]/t_diff, L_d05s[i*10:(i+1)*10]-Ls, c=sm.to_rgba(times[int((i+0.5)*10)]))
    ax2.plot(times[i*10:(i+1)*10], ((np.abs(grad_above)-np.abs(grad_rad_above))/np.abs(grad_rad_above))[i*10:(i+1)*10], c=sm.to_rgba(times[int((i+0.5)*10)]))
    ax3.plot(rolledf['sim_time'][i*10:(i+1)*10], rolledf['enstrophy'][i*10:(i+1)*10]/Re, c=sm.to_rgba(times[int((i+0.5)*10)]))
#    ax3.plot(times[i*10:(i+1)*10], theory_f_enstrophy[i*10:(i+1)*10], c=sm.to_rgba(times[int((i+0.5)*10)]))
ax1.axvline(time_cross_d05/t_diff, color='xkcd:dark grey')
ax3.axvline(time_cross_f, color='xkcd:dark grey')
ax1.set_ylim(0, 0.6)
ax1.set_xlim(0, times.max()/t_diff)
ax2.set_xlim(0, times.max())
ax3.set_xlim(0, times.max())
ax1.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('top')
ax2.set_xticklabels(())
ax1.set_yticks((0.2, 0.4, 0.6))
ax1.set_ylabel(r'$\delta_{0.5}$')
ax2.set_ylabel(r'$\left\langle\frac{\nabla - \nabla_{\rm{rad}}}{\nabla_{\rm{rad}}}\right\rangle_{\rm{PZ}}$')
ax3.set_ylabel(r'$\langle\frac{\omega^2}{\mathcal{R}}\rangle_{\rm{CZ}}$')
#ax3.set_ylabel(r'$\mathcal{R}^{-1}\int_0^{L_s}\overline{\omega^2}dz$')
ax3.set_xlabel('simulation time (freefall units)')
ax1.set_xlabel('simulation time (diffusion units)')


delta_grad = grad_ad - grad_rad
delta_grad_func = interp1d(z, delta_grad, bounds_error=False, fill_value='extrapolate')
y_min = (grad_ad[-1] - delta_grad[-1]*1.5)/grad_ad[-1]
y_max = (grad_ad[-1] + delta_grad[-1]*0.25)/grad_ad[-1]
ax4.plot(z, grad_ad/grad_ad, c='grey', ls='--')
ax4.plot(z, grad_rad/grad_ad, c='grey')
for i in range(grad.shape[0]):
    ax4.plot(z, grad[i,:]/grad_ad, c=sm.to_rgba(np.mean(prof_times[i])))


for i in range(int(len(times)/10)-1):
    if times[(i+1)*10] < 160:
        continue
    grad_01 = grad_ad[-1] - 0.1*delta_grad_func(L_d01s)
    grad_05 = grad_ad[-1] - 0.5*delta_grad_func(L_d05s)
    grad_09 = grad_ad[-1] - 0.9*delta_grad_func(L_d09s)
    ax4.plot(L_d01s[i*10:(i+1)*10], grad_01[i*10:(i+1)*10]/grad_ad[-1], c='r')
    ax4.plot(L_d05s[i*10:(i+1)*10], grad_05[i*10:(i+1)*10]/grad_ad[-1], c='k')
    ax4.plot(L_d09s[i*10:(i+1)*10], grad_09[i*10:(i+1)*10]/grad_ad[-1], c='r')
ax4.set_ylim(y_min, y_max)
ax4.set_xlim(0.9, 1.65)
ax4.set_xlabel('z')
ax4.set_ylabel(r'$\nabla/\nabla_{\rm{ad}}$')
ax4.yaxis.set_ticks_position('right')
ax4.yaxis.set_label_position('right')

plt.savefig('time_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/time_evolution.pdf', dpi=300, bbox_inches='tight')
