import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from matplotlib.patches import ConnectionPatch

col_width = 3.25
page_width = 6.5
golden_ratio = 1.61803398875

dirs = glob.glob('noslip_erf_S_cut/erf*')
dirs += ['noslip_erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256/']

data = []
for d in dirs:
    S = float(d.split('_S')[-1].split('_')[0])
    P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
    Re = float(d.split('_Re')[-1].split('_')[0])
    Lz = float(d.split('_Lz')[-1].split('_')[0])
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        modern_f = f['modern_f'][()]
        modern_xi = f['modern_xi'][()]
        fconv_Ls = f['fconv_Ls'][()]
        Ls = np.mean(f['Ls'][()])
        tot_time = times[-1] - times[0]
        time_window = np.min((500, tot_time/2))
        good_times = times > (times[-1] - time_window)
        mean_L_d01 = np.mean(L_d01s[good_times])
        mean_L_d05 = np.mean(L_d05s[good_times])
        mean_L_d09 = np.mean(L_d09s[good_times])
        mean_f  = np.mean(modern_f[good_times])
        mean_xi = np.mean(modern_xi[good_times])
        mean_fconv_Ls = np.mean(fconv_Ls[good_times])
    if 'erf' in d:
        erf = True
    else:
        erf = False

    data.append((S, P, Re, Ls, Lz, erf, mean_L_d01, mean_L_d05, mean_L_d09, mean_f, mean_xi, mean_fconv_Ls))
data = np.array(data)

S = data[:,0]
P = data[:,1]
Re = data[:,2]
Ls = data[:,3]
Lz = data[:,4]
erf = data[:,5]

L_d01 = data[:,6]
L_d05 = data[:,7]
L_d09 = data[:,8]
theory_f  = data[:,9]
theory_xi = data[:,10]
fconv_Ls = data[:,11]

fig = plt.figure(figsize=(page_width, page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.00, 0., 0.46, 1])
ax2 = fig.add_axes([0.54, 0., 0.46, 1])


good = (erf == 1) * (Re == 4e2) * (P == 4)
#Lcz_erf = (Ls*(fconv_Ls/(0.2*Ls)))[good]
ax1.scatter(S[good], (L_d09[good] - Ls[good]), c='k', label=r"$\delta_{0.9}$", zorder=1, marker='v')
ax1.scatter(S[good], (L_d05[good] - Ls[good]), c='k', label=r"$\delta_{0.5}$", zorder=1, marker='o')
ax1.scatter(S[good], (L_d01[good] - Ls[good]), c='k', label=r"$\delta_{0.1}$", zorder=1, marker='^')
leg1 = ax1.legend(frameon=True, fontsize=8, framealpha=0.6, loc='upper right')
ax1.set_xscale('log')

ax1.set_xlabel(r'$\mathcal{S}$')
ax1.set_ylabel(r'$\delta_{\rm{p}}$')

smooth_S = np.logspace(1, 4, 100)
ax2.plot(smooth_S, 1.5*smooth_S**(-1/2), c='orange', label=r'$\mathcal{S}^{-1/2}$')
ax2.scatter(S[good], L_d09[good] - L_d01[good], c='k')
ax2.legend(loc='lower left', frameon=True, fontsize=8, framealpha=0.6)
ax2.set_yscale('log')
ax2.set_xlim(80, 4000)
ax2.set_ylim(2e-2, 3e-1)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\mathcal{S}$')
ax2.set_ylabel(r'$\delta_{0.9} - \delta_{0.1}$')
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')


plt.savefig('parameters_vs_s.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/parameters_vs_s.pdf', dpi=300, bbox_inches='tight')
