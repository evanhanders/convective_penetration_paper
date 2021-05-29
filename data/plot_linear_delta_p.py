import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
golden_ratio = 1.61803398875

dirs = glob.glob('linear*/lin*/')

data = []
for d in dirs:
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        enstrophy_f = f['f_theory_enstrophy'][()]
        Ls = f['Ls'][()]
        tot_time = times[-1] - times[0]
        time_window = np.min((500, tot_time/2))
        good_times = times > (times[-1] - time_window)
        mean_L_d01 = np.mean(L_d01s[good_times])
        mean_L_d05 = np.mean(L_d05s[good_times])
        mean_L_d09 = np.mean(L_d09s[good_times])
        mean_enstrophy_f = np.mean(enstrophy_f[good_times])
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

    data.append((S, P, Re, threeD, mean_L_d01, mean_L_d05, mean_L_d09, mean_enstrophy_f, Ls, Lz, erf, ae))
    print(d)
    print("                 ",S, P, Re, mean_L_d01, mean_L_d05, mean_L_d09, mean_enstrophy_f)
data = np.array(data)

S = data[:,0]
P = data[:,1]
Re = data[:,2]
threeD = data[:,3]
L_d01 = data[:,4]
L_d05 = data[:,5]
L_d09 = data[:,6]
enstrophy_f = data[:,7]
Ls = data[:,8]
Lz = data[:,9]
erf = data[:,10]
ae = data[:,11]

Lcz_norm = Ls - 0.2 #account for flux change at base of CZ.
theory   = np.sqrt(P*(1 - enstrophy_f))

line_P = np.logspace(-2, 2, 100)
approx_f = 0.925
line_theory = np.sqrt(line_P*(1-approx_f))

plt.figure(figsize=(col_width, col_width/golden_ratio))
good = (erf == 0)
plt.scatter(P[good], (L_d09[good] - Ls[good])/Lcz_norm[good], c='k', label=r"$\delta_{0.9}$", zorder=1, marker='v')
plt.scatter(P[good], (L_d05[good] - Ls[good])/Lcz_norm[good], c='k', label=r"$\delta_{0.5}$", zorder=1, marker='o')
plt.scatter(P[good], (L_d01[good] - Ls[good])/Lcz_norm[good], c='k', label=r"$\delta_{0.1}$", zorder=1, marker='^')
#plt.scatter(P[good], theory[good], c='orange', label=r'$\sqrt{\mathcal{P}(1 - \langle f \rangle)}$', marker='x')
plt.plot(line_P, line_theory, c='orange', zorder=0, label='theory')
plt.xscale('log')
plt.yscale('log')

#x = np.logspace(-3, 2, 100)
#y1 = 0.23*x**(1/2)
#plt.plot(x, y1, c='orange', label=r'Theory$\,\,(0.25 \mathcal{P}_L^{1/2})$')
plt.legend(frameon=True, fontsize=8, loc='upper left')
plt.xlabel('$\mathcal{P}_L$')
plt.ylabel(r'$\delta_p/\tilde{L_s}$')
plt.xlim(8e-3, 2e1)

plt.savefig('linear_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/linear_3D_penetration_depths.pdf', dpi=300, bbox_inches='tight')
