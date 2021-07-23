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

with h5py.File('temporal_data/erf_re4e2_p4e0_s1e3_longevolution.h5', 'r') as in_f:
    d01s = in_f['scalars/d01'][100:] 
    d05s = in_f['scalars/d05'][100:] 
    d09s = in_f['scalars/d09'][100:] 
    times = in_f['scalars/times'][100:]
    f = in_f['scalars/f'][100:]
    xi = in_f['scalars/xi'][100:]  
    Ls = in_f['scalars/Ls'][()]
    z    = in_f['profiles/z'][()]
    grad = in_f['profiles/grad'][()]  
    prof_times = in_f['profiles/times'][()] 
    grad_ad = in_f['profiles/grad_ad'][()] 
    grad_rad = in_f['profiles/grad_rad'][()] 

S = 1e3
P = 4
Re = 4e2
Lz = 2

k_rz = 0.2 / (S * P)
t_diff = 1/k_rz

fig = plt.figure(figsize=(2*col_width, col_width/golden_ratio))
ax1 = fig.add_axes([0,    0.67, 0.45, 0.33])
ax2 = fig.add_axes([0,    0.34, 0.45, 0.33])
ax3 = fig.add_axes([0,    0.00, 0.45, 0.34])
ax4 = fig.add_axes([0.50,    0, 0.45, 1])


#Take rolling averages of theory values
f_data = dict()
f_data['sim_time'] = times
f_data['f'] = f
f_data['xi'] = xi
f_df = pd.DataFrame(data=f_data)
rolledf = f_df.rolling(window=200, min_periods=25).mean()
roll_times = rolledf['sim_time']

#Find convergence times
final_f  = np.mean(f[-1000:])
error_f = np.abs(1 - rolledf['f']/final_f)
time_cross_f = times[error_f < 0.01][0]

final_xi = np.mean(xi[-1000:])
error_xi = np.abs(1 - rolledf['xi']/final_xi)
time_cross_xi = times[error_xi < 0.01][0]

final_d05 = np.mean(d05s[-1000:])
error_d05 = np.abs(1 - d05s/final_d05)
time_cross_d05 = times[error_d05 < 0.01][0]

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=times.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

#Plot time traces
colors = [sm.to_rgba(t) for t in times]
ax1.axhline(final_d05, c='xkcd:dark grey', lw=0.5)
ax2.axhline(final_f,   c='xkcd:dark grey', lw=0.5)
ax3.axhline(final_xi,  c='xkcd:dark grey', lw=0.5)
N_points = len(times)
N_color  = 50
for i in range(N_color):
    lower = int(N_points*i/N_color)
    upper = int(N_points*(i+1)/N_color)-1
    ax1.plot(times[lower:upper]/t_diff, d05s[lower:upper],          c=sm.to_rgba(times[upper]))#, rasterized=True)
    ax2.plot(roll_times[lower:upper],   rolledf['f'][lower:upper],  c=sm.to_rgba(times[upper]))#, rasterized=True)
    ax3.plot(roll_times[lower:upper],   rolledf['xi'][lower:upper], c=sm.to_rgba(times[upper]))#, rasterized=True)
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
    if i % 2 == 0:
        continue
    ax4.plot(z, grad[i,:]/grad_ad, c=sm.to_rgba(np.mean(prof_times[i])))#, rasterized=True)

grad_01 = grad_ad[-1] - 0.1*delta_grad_func(Ls + d01s)
grad_05 = grad_ad[-1] - 0.5*delta_grad_func(Ls + d05s)
grad_09 = grad_ad[-1] - 0.9*delta_grad_func(Ls + d09s)
ax4.plot(Ls + d01s, grad_01/grad_ad[-1], c='r')
ax4.plot(Ls + d05s, grad_05/grad_ad[-1], c='k')
ax4.plot(Ls + d09s, grad_09/grad_ad[-1], c='r')
ax4.set_ylim(y_min, y_max)
ax4.set_xlim(0.9, 1.65)
ax4.set_xlabel('z')
ax4.set_ylabel(r'$\nabla/\nabla_{\rm{ad}}$')
ax4.yaxis.set_ticks_position('right')
ax4.yaxis.set_label_position('right')
ax4.axvline(Ls, c='k', ls='--')

plt.savefig('time_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/time_evolution.pdf', dpi=300, bbox_inches='tight')
