from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
golden_ratio = 1.61803398875

dirs = glob.glob('erf_AE_cut/erf*/')
dirs.append('erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.55/')



plt.figure(figsize=(col_width, col_width/golden_ratio))

data = OrderedDict()
for d in dirs:
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
    data[d] = [S, P, Re, threeD, Lz, erf, times, z_departure, z_overshoot]

for d in dirs:
    times = np.copy(data[d][-3])
    z_departure = data[d][-2]
    z_overshoot = data[d][-1]
    if 'predictive0.4_noiters/' in d:
        color = 'blue'
        times -= times[-1]
    elif 'schwarzschild/' in d or 'schwarzschild_restart/' in d:
        color = 'k'
        for d2 in dirs:
            times2 = data[d2][-3]
            if 'schwarzschild_restart/' in d2:
                times -= times2[-1]
                print(d, d2, times[-1])
    elif 'predictive0.7/' in d or 'pertUpContinued/' in d:
        color = 'green'
        for d2 in dirs:
            times2 = data[d2][-3]
            if 'pertUpContinued/' in d2:
                times -= times2[-1]
    elif 'predictive0.4/' in d:
        color = 'indigo'
        times -= times[-1]
    elif 'erf_P_cut' in d:
        color = 'orange'
        times -= times[-1]
    else:
        color = 'red'
        times -= times[-1]

    if data[d][1] == 4:
        plt.plot(times, z_departure-1, color=color)
        plt.plot(times, z_overshoot-1, color=color)

plt.ylim(0, 0.7)
plt.show()
#
#S = data[:,0]
#P = data[:,1]
#Re = data[:,2]
#threeD = data[:,3]
#z_top = data[:,4]
#z_ov  = data[:,5]
#z_av = (z_top + z_ov)/2
#Lz = data[:,6]
#erf = data[:,7]
#
#plt.figure(figsize=(col_width, col_width/golden_ratio))
#good = (erf == 0)
#plt.scatter(P[good], z_top[good] - 1, c='k', label=r"Simulations ($\delta_p$)", zorder=100, marker='^')
#plt.scatter(P[good], z_ov[good] - 1, c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=100, marker='v')
#plt.scatter(P[good], z_av[good] - 1, c='k', zorder=100, marker='o')
#plt.xscale('log')
#plt.yscale('log')
#
#x = np.logspace(-3, 2, 100)
#y1 = 0.23*x**(1/2)
#plt.plot(x, y1, c='orange', label=r'Theory$\,\,(0.25 \mathcal{P}_L^{1/2})$')
#plt.legend(frameon=True, fontsize=8, loc='upper left')
#plt.xlabel('$\mathcal{P}_L$')
#plt.ylabel('$\delta_p$')
#plt.xlim(8e-3, 2e1)
#
#plt.savefig('linear_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
