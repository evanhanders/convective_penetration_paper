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
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart4",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart5",
        ]

fig = plt.figure(figsize=(2*col_width, col_width/golden_ratio))
ax1 = fig.add_axes([0,    0.67, 0.45, 0.33])
ax2 = fig.add_axes([0,    0.34, 0.45, 0.33])
ax3 = fig.add_axes([0,    0.00, 0.45, 0.34])
ax4 = fig.add_axes([0.50,    0, 0.45, 1])

#Parse input parmeters
data = OrderedDict()
S = float(dirs[0].split('_S')[-1].split('_')[0])
P = float(dirs[0].split('_Pr')[0].split('_P')[-1].split('_')[0])
Re = float(dirs[0].split('_Re')[-1].split('_')[0])
Lz = float(dirs[0].split('_Lz')[-1].split('_')[0])
threeD = True
erf = True

k_rz = 0.2 / (S * P)
t_diff = 1/k_rz

#Read in scalar and profile data
times = []
L_d01s = []
L_d05s = []
L_d09s = []
theory_f = []
theory_xi = []
grad = []
prof_times = []
for i, d in enumerate(dirs):
    if i == 0:
        skip_points = 100
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        dictionary = dict()
        for k in ['times', 'L_d01s', 'L_d05s', 'L_d09s', 'modern_f', 'modern_xi']:
            dictionary[k] = f[k][()]
            if k != 'times':
                dictionary[k] = dictionary[k][np.unique(dictionary['times'], return_index=True)[1]]
        dictionary['times'] = np.unique(dictionary['times'])
        if i > 0:
            #clean out some time artifacts from restarting simulations
            skip_points = int(np.sum(dictionary['times'] <= times[-1][-1]))
        
        times.append(dictionary['times'][skip_points:])
        L_d01s.append(dictionary['L_d01s'][skip_points:])
        L_d05s.append(dictionary['L_d05s'][skip_points:])
        L_d09s.append(dictionary['L_d09s'][skip_points:])
        theory_f.append(dictionary['modern_f'][skip_points:])
        theory_xi.append(dictionary['modern_xi'][skip_points:])
        Ls = f['Ls'][()]
    with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
        z = f['z'][()]
        grad_ad = -f['T_ad_z'][()][0,:].squeeze()
        grad_rad = -f['T_rad_z'][()][0,:].squeeze()

        grad.append(-f['T_z'][()])
        prof_times.append(f['T_z_times'][()])

Lcz_norm = Ls - 0.2 # for erf case, maybe necessary.
times          = np.array(np.concatenate(times))
L_d01s         = np.array(np.concatenate(L_d01s))
L_d05s         = np.array(np.concatenate(L_d05s))
L_d09s         = np.array(np.concatenate(L_d09s))
theory_f       = np.array(np.concatenate(theory_f))
theory_xi      = np.array(np.concatenate(theory_xi))

grad           = np.array(np.concatenate(grad, axis=0))
prof_times     = np.array(np.concatenate(prof_times))

#Take rolling averages of theory values
f_data = dict()
f_data['sim_time'] = times
f_data['f'] = theory_f
f_data['xi'] = theory_xi
f_df = pd.DataFrame(data=f_data)
rolledf = f_df.rolling(window=200, min_periods=25).mean()
roll_times = rolledf['sim_time']

#Find convergence times
final_f  = np.mean(theory_f[-1000:])
error_f = np.abs(1 - rolledf['f']/final_f)
time_cross_f = times[error_f < 0.01][0]

final_xi = np.mean(theory_xi[-1000:])
error_xi = np.abs(1 - rolledf['xi']/final_xi)
time_cross_xi = times[error_xi < 0.01][0]

delta_d05 = (L_d05s - Ls)
final_d05 = np.mean(delta_d05[-1000:])
error_d05 = np.abs(1 - (delta_d05/final_d05))
time_cross_d05 = times[error_d05 < 0.01][0]

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=times.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

#Plot time traces
colors = [sm.to_rgba(t) for t in times]
ax1.axhline(final_d05, c='xkcd:dark grey', lw=0.5)
ax2.axhline(final_f,   c='xkcd:dark grey', lw=0.5)
ax3.axhline(final_xi,  c='xkcd:dark grey', lw=0.5)
#ax1.plot(times/t_diff, L_d05s-Ls,     c='k')
#ax2.plot(roll_times,   rolledf['f'] , c='k')
#ax3.plot(roll_times,   rolledf['xi'], c='k')
for i in range(int(len(times)/10)-1):
    ax1.plot(times[i*10:(i+1)*10]/t_diff, L_d05s[i*10:(i+1)*10]-Ls,     c=sm.to_rgba(times[int((i+0.5)*10)]))
    ax2.plot(roll_times[i*10:(i+1)*10],   rolledf['f'][i*10:(i+1)*10],  c=sm.to_rgba(times[int((i+0.5)*10)]))
    ax3.plot(roll_times[i*10:(i+1)*10],   rolledf['xi'][i*10:(i+1)*10], c=sm.to_rgba(times[int((i+0.5)*10)]))
ax1.axvline(time_cross_d05/t_diff, color='xkcd:dark grey')
ax2.axvline(time_cross_f, color='xkcd:dark grey')
ax3.axvline(time_cross_xi, color='xkcd:dark grey')
ax1.set_ylim(0, 0.6)
ax1.set_xlim(0, times.max()/t_diff)
ax2.set_xlim(0, times.max())
ax3.set_xlim(0, times.max())
ax1.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('top')
ax2.set_xticklabels(())
ax1.set_yticks((0.2, 0.4, 0.6))
ax1.set_ylabel(r'$\delta_{0.5}$')
ax2.set_ylabel(r'$f$')
ax3.set_ylabel(r'$\xi$')
ax3.set_xlabel('simulation time (freefall units)')
ax1.set_xlabel('simulation time (diffusion units)')



#Plot grad profiles
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
