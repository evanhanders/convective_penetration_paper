from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

import pandas as pd

col_width = 3.25
golden_ratio = 1.61803398875

dirs = glob.glob('erf_AE_cut/erf*/')



fig = plt.figure(figsize=(2*col_width, col_width/golden_ratio))
ax1 = fig.add_axes([0.00,    0.5, 0.33, 0.5])
ax2 = fig.add_axes([0.00,    0.0, 0.33, 0.5])
ax3 = fig.add_axes([0.33,    0.5, 0.33, 0.5])
ax4 = fig.add_axes([0.33,    0.0, 0.33, 0.5])
ax5 = fig.add_axes([0.66,    0.5, 0.33, 0.5])
ax6 = fig.add_axes([0.66,    0.0, 0.33, 0.5])

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
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        Ls = f['Ls'][()]
        f_theory_enstrophy = f['enstrophy_Ls'][()]/Re


    data[d] = [S, P, Re, threeD, Lz, erf, times, L_d01s, L_d05s, L_d09s, f_theory_enstrophy, Ls]

dirs_p1 = ["erf_AE_cut/erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
           "erf_AE_cut/erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart/"]

dirs_p2 = ["erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
           "erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart/",
           "erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart2/"]

dirs_p4 = ["erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart2/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart3/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart4/"]

for d in dirs:
    P = data[d][1]
    times = np.copy(data[d][6])
    L_d01 = data[d][7]
    L_d05 = data[d][8]
    L_d09 = data[d][9]
    f_enstrophy    = data[d][10]
    N_skip = 0

    if 'predictive0.4_noiters/' in d or 'pertUpContinued/' in d:
        color = 'blue'
        times -= times[-1]
        continue
    elif 'schwarzschild/' in d:
        color = 'k'
        zorder = 100

        if P == 4:
            N_skip = 100
            these_dirs = dirs_p4
        elif P == 1:
            N_skip = 300
            these_dirs = dirs_p1
        elif P == 2:
            N_skip = 150
            these_dirs = dirs_p2
        times = []
        L_d01 = []
        L_d05 = []
        L_d09 = []
        f_enstrophy = []
        for d in these_dirs:
            times.append(np.copy(data[d][6]))
            L_d01.append(data[d][7])
            L_d05.append(data[d][8])
            L_d09.append(data[d][9])
            f_enstrophy.append(data[d][10])
        times = np.array(np.concatenate(times))
        L_d01 = np.array(np.concatenate(L_d01))
        L_d05 = np.array(np.concatenate(L_d05))
        L_d09 = np.array(np.concatenate(L_d09))
        f_enstrophy = np.array(np.concatenate(f_enstrophy))

        times -= times[-1]
    elif 'schwarzschild_restart' in d:
        continue
    elif ('predictive0.7/' in d and P == 4) or ('predictive0.45' in d and P == 2) or ('predictive0.3' in d and P == 1):
        color = 'orange'
        zorder = 1
        times -= times[-1]
        if P == 1:
            N_skip = 0#10
        if P == 2:
            N_skip = 10
    elif ('predictive0.4/' in d and P == 4) or ('predictive0.1' in d and P == 2) or ('predictive0' in d and P == 1):
        color = 'green'
        zorder = 2
        times -= times[-1]
        if P == 1:
            N_skip = 10
        if P == 2:
            N_skip = 10
    else:
        zorder = 0
        color = 'red'
        times -= times[-1]
    times = times[N_skip:]
    L_d01 = L_d01[N_skip:]
    L_d05 = L_d05[N_skip:]
    L_d09 = L_d09[N_skip:]
    f_enstrophy = f_enstrophy[N_skip:]


    f_data = dict()
    f_data['sim_time'] = times
    f_data['f'] = f_enstrophy
    f_df = pd.DataFrame(data=f_data)
    rolledf = f_df.rolling(window=50, min_periods=25).mean()

    if P == 1:
        ax1.plot(times, (L_d01-Ls), color=color, zorder=zorder)
        ax1.plot(times, (L_d09-Ls), color=color, zorder=zorder)
        ax2.plot(rolledf['sim_time'], rolledf['f'], color=color, zorder=zorder)
    elif P == 2:
        ax3.plot(times, (L_d01-Ls), color=color, zorder=zorder)
        ax3.plot(times, (L_d09-Ls), color=color, zorder=zorder)
        ax4.plot(rolledf['sim_time'], rolledf['f'], color=color, zorder=zorder)
    elif P == 4:
        ax5.plot(times, (L_d01-Ls), color=color, zorder=zorder)
        ax5.plot(times, (L_d09-Ls), color=color, zorder=zorder)
        ax6.plot(rolledf['sim_time'], rolledf['f'], color=color, zorder=zorder)

for ax in [ax2, ax4, ax6]:
    ax.set_ylim(0.08, 0.15)
    ax.set_xlabel(r'$t - t_{\rm{end}}$')
for ax in [ax3, ax4, ax6]:
    ax.set_yticks(())
for ax in [ax1, ax3, ax5]:
    ax.set_xticks(())
    ax.set_ylim(0, 0.6)
    ax.set_yticks((0.2, 0.4, 0.6))
for ax in [ax3, ax5]:
    ax.set_yticklabels(())

ax5.yaxis.set_ticks_position('right')
ax1.set_ylabel(r'$\delta_{\rm{p}}$')
ax2.set_ylabel(r'$\langle \frac{\omega^2}{\mathcal{R}} \rangle_{\rm{CZ}}$')

ax1.text(0.03, 0.85, r'$\mathcal{P} = 1$', transform=ax1.transAxes)
ax3.text(0.03, 0.85, r'$\mathcal{P} = 2$', transform=ax3.transAxes)
ax5.text(0.03, 0.85, r'$\mathcal{P} = 4$', transform=ax5.transAxes)

fig.savefig('AE_time_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/AE_time_figure.pdf', dpi=300, bbox_inches='tight')
