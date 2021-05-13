from collections import OrderedDict
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from scipy.interpolate import interp1d


col_width = 3.25
golden_ratio = 1.61803398875

d = "erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/"

fig = plt.figure(figsize=(col_width, 2*col_width/golden_ratio))
ax1 = fig.add_axes([0, 0.52, 1, 0.45])
ax2 = fig.add_axes([0, 0.03, 1, 0.45])


data = OrderedDict()
S = float(d.split('_S')[-1].split('_')[0])
P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
Re = float(d.split('_Re')[-1].split('_')[0])
Lz = float(d.split('_Lz')[-1].split('_')[0])

if '3D' in d:
    threeD = True
else:
    threeD = False
if 'erf' in d:
    erf = True
else:
    erf = False

with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
    times = f['times'][()]
    z_departure = f['z_departure'][()]
    z_overshoot = f['z_overshoot'][()]

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=times.max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

colors = [sm.to_rgba(t) for t in times]
for i in range(int(len(times)/10)-1):
    ax1.plot(times[i*10:(i+1)*10], z_departure[i*10:(i+1)*10]-1, c=sm.to_rgba(times[int((i+0.5)*10)]))
    ax1.plot(times[i*10:(i+1)*10], z_overshoot[i*10:(i+1)*10]-1, c=sm.to_rgba(times[int((i+0.5)*10)]))
ax1.set_ylim(0, 0.3)
ax1.set_xlim(0, times.max())
ax1.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('top')
ax1.set_xlabel('simulation time')
ax1.set_ylabel(r'$\delta_p$, $\delta_{\rm{ov}}$')


with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
    z = f['z'][()]
    grad = -f['T_z'][()]
    grad_ad = -f['T_ad_z'][()][0,:].squeeze()
    grad_rad = -f['T_rad_z'][()][0,:].squeeze()
    prof_times = f['T_z_times'][()]

delta_grad = grad_ad - grad_rad
delta_grad_func = interp1d(z, delta_grad, bounds_error=False, fill_value='extrapolate')
y_min = (grad_ad[-1] - delta_grad[-1]*1.5)/grad_ad[-1]
y_max = (grad_ad[-1] + delta_grad[-1]*1.1)/grad_ad[-1]
ax2.plot(z, grad_ad/grad_ad, c='grey', ls='--')
ax2.plot(z, grad_rad/grad_ad, c='grey')
for i in range(grad.shape[0]):
    ax2.plot(z, grad[i,:]/grad_ad, c=sm.to_rgba(np.mean(prof_times[i])))


for i in range(int(len(times)/10)-1):
    if times[(i+1)*10] < 160:
        continue
    grad_dep = grad_ad[-1] - 0.1*delta_grad_func(z_departure)
    grad_ov = grad_ad[-1] - 0.9*delta_grad_func(z_overshoot)
    ax2.plot(z_departure[i*10:(i+1)*10], grad_dep[i*10:(i+1)*10]/grad_ad[-1], c='r')
    ax2.plot(z_overshoot[i*10:(i+1)*10], grad_ov[i*10:(i+1)*10]/grad_ad[-1], c='r')
#    ax2.plot(z_departure[i*10:(i+1)*10], grad_dep[i*10:(i+1)*10]/grad_ad[-1], c=sm.to_rgba(times[int((i+0.5)*10)]))
#    ax2.plot(z_overshoot[i*10:(i+1)*10], grad_ov[i*10:(i+1)*10]/grad_ad[-1], c=sm.to_rgba(times[int((i+0.5)*10)]))
ax2.set_ylim(y_min, y_max)
ax2.set_xlim(0.9, 1.35)
ax2.set_xlabel('z')
ax2.set_ylabel(r'$\nabla/\nabla_{\rm{ad}}$')

plt.savefig('time_evolution.png', dpi=300, bbox_inches='tight')
