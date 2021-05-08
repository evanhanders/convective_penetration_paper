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
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        z_departure = f['z_departure'][()]
        tot_time = times[-1] - times[0]
        good_times = times > (times[-1] - 1000)
        mean_departure = np.mean(z_departure[good_times])
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

    data.append((S, P, Re, threeD, mean_departure, Lz, erf))
data = np.array(data)

S = data[:,0]
P = data[:,1]
Re = data[:,2]
threeD = data[:,3]
z_top = data[:,4]
Lz = data[:,5]
erf = data[:,6]

fig = plt.figure(figsize=(page_width, col_width/golden_ratio))

ax1 = fig.add_axes([0.04, 0, 0.32, 1])
ax2 = fig.add_axes([0.36, 0, 0.32, 1])
ax3 = fig.add_axes([0.68, 0, 0.32, 1])


good = (erf == 1) * (Re == 4e2) * (S == 1e3)
ax1.scatter(P[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)

x = np.logspace(-3, 2, 100)
y1 = 0.125*x
ax1.plot(x, y1, c='orange', label=r'Theory$\,\,(0.125 \mathcal{P}_E)$')
ax1.legend(frameon=False)
ax1.set_xlabel('$\mathcal{P}_E|_{\mathcal{R} = 400, \mathcal{S} = 10^3}$')
ax1.set_ylabel('$\delta_p$')
ax1.set_xlim(0, 21)
ax1.set_ylim(0, 3)

good = (erf == 1) * (P == 4) * (S == 1e3)
ax2.scatter(Re[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
ax2.set_xscale('log')
ax2.set_xlabel('$\mathcal{R}|_{\mathcal{P} = 4, \mathcal{S} = 10^3}$')
ax2.set_ylim(0, 3)

ax2_in = ax2.inset_axes([0.3, 0.6, 0.6, 0.35])
ax2_in.scatter(Re[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
ax2_in.set_xscale('log')
ax2_in.set_ylim(0.4, 0.6)

good = (erf == 1) * (P == 4) * (Re == 4e2)
ax3.scatter(S[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
ax3.set_xscale('log')
ax3.set_xlabel('$\mathcal{S}|_{\mathcal{P} = 4, \mathcal{R} = 400}$')
ax3.set_ylim(0, 3)

ax3_in = ax3.inset_axes([0.3, 0.6, 0.6, 0.35])
ax3_in.scatter(S[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
ax3_in.set_xscale('log')
ax3_in.set_ylim(0.4, 0.6)




ax2.set_yticklabels(())
ax3.set_yticklabels(())

plt.savefig('erf_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
