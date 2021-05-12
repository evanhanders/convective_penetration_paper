import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
page_width = 6.5
golden_ratio = 1.61803398875

dirs = glob.glob('linear*/lin*/') + glob.glob('erf*/erf*/')

data = []
for d in dirs:
    print(d)
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        z_departure = f['z_departure'][()]
        z_overshoot = f['z_overshoot'][()]
        tot_time = times[-1] - times[0]
        good_times = times > (times[-1] - 500)
        mean_departure = np.mean(z_departure[good_times])
        mean_overshoot = np.mean(z_overshoot[good_times])
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
    if 'AE' in d:
        ae = True
    else:
        ae = False

    data.append((S, P, Re, threeD, mean_departure, mean_overshoot, Lz, erf, ae))
data = np.array(data)

S = data[:,0]
P = data[:,1]
Re = data[:,2]
threeD = data[:,3]
z_top = data[:,4]
z_ov = data[:,5]
z_av = (z_top + z_ov)/2
Lz = data[:,6]
erf = data[:,7]
ae = data[:,8]

fig = plt.figure(figsize=(page_width, page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.04, 0, 0.29, 1])
ax2 = fig.add_axes([0.40, 0, 0.29, 1])
ax3 = fig.add_axes([0.69, 0, 0.29, 1])


good = (erf == 1) * (Re == 4e2) * (S == 1e3) * (ae == 0)
ax1.scatter(P[good], z_ov[good] - 1, c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=100, marker='v')
ax1.scatter(P[good], z_top[good] - 1, c='k', label=r"Simulations ($\delta_p$)", zorder=100, marker='^')
ax1.scatter(P[good], z_av[good] - 1, c='k', zorder=100, marker='o')

x = np.logspace(-3, 2, 100)
y1 = 0.11*x
ax1.plot(x, y1, c='orange', label=r'Theory$\,\,(0.125 \mathcal{P}_E)$')
ax1.legend(frameon=True, fontsize=8)#False)
ax1.set_xlabel('$\mathcal{P}_E$')
ax1.set_title('$\mathcal{P}_E|_{\mathcal{R} = 400, \mathcal{S} = 10^3}$')
ax1.set_ylabel('$\delta_p$')
ax1.set_xlim(0, 21)
ax1.set_ylim(0, 3)

good = (erf == 1) * (P == 4) * (S == 1e3) * (ae == 0)
ax2.scatter(Re[good], z_top[good] - 1, c='k', label=r"Simulations ($\delta_p$)", zorder=100, marker='^')
ax2.scatter(Re[good], z_ov[good] - 1, c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=100, marker='v')
ax2.scatter(Re[good], z_av[good] - 1, c='k', zorder=100, marker='o')
#ax2.scatter(Re[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
ax2.set_xscale('log')
ax2.set_xlabel('$\mathcal{R}$')
ax2.set_title('$\mathcal{R}|_{\mathcal{P} = 4, \mathcal{S} = 10^3}$')
ax2.set_ylim(0.35, 0.65)

#ax2_in = ax2.inset_axes([0.3, 0.6, 0.6, 0.35])
#ax2_in.scatter(Re[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
#ax2_in.set_xscale('log')
#ax2_in.set_ylim(0.4, 0.6)
#
#extra_re = [3.2e3, 6.4e3]
#extra_delta = [ 0.45, 0.44]
#ax2_in.scatter(extra_re, extra_delta, c='k', zorder=100)

good = (erf == 1) * (P == 4) * (Re == 4e2) * (ae == 0)
ax3.scatter(S[good], z_top[good] - 1, c='k', label=r"Simulations ($\delta_p$)", zorder=100, marker='^')
ax3.scatter(S[good], z_ov[good] - 1, c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=100, marker='v')
ax3.scatter(S[good], z_av[good] - 1, c='k', zorder=100, marker='o')
#ax3.scatter(S[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
ax3.set_xscale('log')
ax3.set_xlabel('$\mathcal{S}$')
ax3.set_title('$\mathcal{S}|_{\mathcal{P} = 4, \mathcal{R} = 400}$')
ax3.set_ylim(0.35, 0.65)

#ax3_in = ax3.inset_axes([0.3, 0.6, 0.6, 0.35])
#ax3_in.scatter(S[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
#ax3_in.set_xscale('log')
#ax3_in.set_ylim(0.4, 0.6)


ax1.axhline(0.35, c='k', lw=0.5)
ax1.axhline(0.65, c='k', lw=0.5)


ax2.set_yticklabels(())
ax3.yaxis.set_ticks_position('right')
ax3.yaxis.set_label_position('right')
ax2.yaxis.set_ticks_position('right')
ax2.tick_params(axis='y', direction='in')
ax1.set_ylabel('$\delta_p$')

plt.savefig('erf_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
