import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
golden_ratio = 1.61803398875

dirs = glob.glob('erf_Re_cut*/erf*/') + glob.glob('erf_P_cut*/erf*/')

data = []
for d in dirs:
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        cz_vel = f['cz_velocities'][()]
        tot_time = times[-1] - times[0]
        good_times = times > times[0] + tot_time/2
        mean_vel = np.mean(cz_vel[good_times])
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

    data.append((S, P, Re, threeD, mean_vel, Lz, erf))
data = np.array(data)

S = data[:,0]
P = data[:,1]
Re = data[:,2]
threeD = data[:,3]
vel = data[:,4]
Lz = data[:,5]
erf = data[:,6]

plt.figure(figsize=(col_width, col_width/golden_ratio))
good = (erf == 1) * (P == 4)
plt.scatter(Re[good], vel[good], c='k')
plt.xscale('log')
#plt.yscale('log')

plt.show()

#plt.savefig('linear_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
#plt.savefig('../manuscript/linear_3D_penetration_depths.pdf', dpi=300, bbox_inches='tight')
