import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
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

plt.figure(figsize=(col_width, col_width/golden_ratio))
good = (erf == 0)
plt.scatter(P[good], z_top[good] - 1, c='k', label="Simulations", zorder=100)
plt.xscale('log')
plt.yscale('log')

x = np.logspace(-3, 2, 100)
y1 = 0.23*x**(1/2)
plt.plot(x, y1, c='orange', label=r'Theory$\,\,(0.25 \mathcal{P}_L^{1/2})$')
plt.legend(frameon=False)
plt.xlabel('$\mathcal{P}_L$')
plt.ylabel('$\delta_p$')
plt.xlim(8e-3, 2e1)

plt.savefig('linear_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
