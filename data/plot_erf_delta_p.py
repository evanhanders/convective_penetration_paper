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
theory   = P*(1 - enstrophy_f)

fig = plt.figure(figsize=(page_width, page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.04, 0, 0.29, 1])
ax2 = fig.add_axes([0.40, 0, 0.29, 1])
ax3 = fig.add_axes([0.69, 0, 0.29, 1])
second_plot_lower = 0.4
second_plot_upper = 0.7

line_P = np.linspace(0.5, 21, 100)
approx_f = 0.875
line_theory = line_P * (1 - approx_f)

good = (erf == 1) * (Re == 4e2) * (S == 1e3) * (ae == 0)
ax1.scatter(P[good], (L_d09[good] - Ls[good])/Lcz_norm[good], c='k', label=r"$\delta_{0.9}$", zorder=1, marker='v')
ax1.scatter(P[good], (L_d05[good] - Ls[good])/Lcz_norm[good], c='k', label=r"$\delta_{0.5}$", zorder=1, marker='o')
ax1.scatter(P[good], (L_d01[good] - Ls[good])/Lcz_norm[good], c='k', label=r"$\delta_{0.1}$", zorder=1, marker='^')
ax1.plot(line_P, line_theory, c='orange', label=r'theory'.format(approx_f), zorder=0)
#ax1.plot(line_P, line_theory, c='orange', label=r'theory ($f$ = {})'.format(approx_f), zorder=0)
#ax1.scatter(P[good], theory[good], c='orange', label=r'$\mathcal{P}(1 - \langle f \rangle)$', marker='x')
print(Ls[good])

#y1 = 0.11*x
ax1.legend(frameon=True, fontsize=8, framealpha=0.6)
ax1.set_xlabel('$\mathcal{P}_D$')
ax1.set_title('$\mathcal{P}_D|_{\mathcal{R} = 400, \mathcal{S} = 10^3}$')
ax1.set_ylabel(r'$\delta_{\rm{p}}/\tilde{L_s}$')
ax1.set_xlim(0, 11)
ax1.set_ylim(0, 1.5)

good = (erf == 1) * (P == 4) * (S == 1e3) * (ae == 0)
ax2.axhline(4*(1 - 0.875), c='orange', zorder=0)
ax2.scatter(Re[good], (L_d01[good] - Ls[good])/Lcz_norm[good], c='k', label=r"Simulations ($\delta_p$)", zorder=1, marker='^')
ax2.scatter(Re[good], (L_d09[good] - Ls[good])/Lcz_norm[good], c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=1, marker='v')
ax2.scatter(Re[good], (L_d05[good] - Ls[good])/Lcz_norm[good], c='k', zorder=1, marker='o')
#ax2.scatter(Re[good], theory[good], c='orange', label=r'$\mathcal{P}(1 - \langle f \rangle)$', marker='x')
ax2.set_xscale('log')
ax2.set_xlabel('$\mathcal{R}$')
ax2.set_title('$\mathcal{R}|_{\mathcal{P} = 4, \mathcal{S} = 10^3}$')
ax2.set_ylim(second_plot_lower, second_plot_upper)

#ax2_in = ax2.inset_axes([0.3, 0.6, 0.6, second_plot_lower])
#ax2_in.scatter(Re[good], L_d01[good] - Ls[good], c='k', label="Simulations", zorder=1)
#ax2_in.set_xscale('log')
#ax2_in.set_ylim(0.4, 0.6)
#
#extra_re = [3.2e3, 6.4e3]
#extra_delta = [ 0.45, 0.44]
#ax2_in.scatter(extra_re, extra_delta, c='k', zorder=1)

good = (erf == 1) * (P == 4) * (Re == 4e2) * (ae == 0)
ax3.axhline(4*(1 - 0.875), c='orange', zorder=0)
ax3.scatter(S[good], (L_d01[good] - Ls[good])/Lcz_norm[good], c='k', label=r"Simulations ($\delta_p$)", zorder=1, marker='^')
ax3.scatter(S[good], (L_d09[good] - Ls[good])/Lcz_norm[good], c='k', label=r"Simulations ($\delta_{\rm{ov}}$)", zorder=1, marker='v')
ax3.scatter(S[good], (L_d05[good] - Ls[good])/Lcz_norm[good], c='k', zorder=1, marker='o')
#ax3.scatter(S[good], theory[good], c='orange', label=r'$\mathcal{P}(1 - \langle f \rangle)$', marker='x')
#ax3.scatter(S[good], L_d01[good] - Ls[good], c='k', label="Simulations", zorder=1)
ax3.set_xscale('log')
ax3.set_xlabel('$\mathcal{S}$')
ax3.set_title('$\mathcal{S}|_{\mathcal{P} = 4, \mathcal{R} = 400}$')
ax3.set_ylim(second_plot_lower, second_plot_upper)

#ax3_in = ax3.inset_axes([0.3, 0.6, 0.6, second_plot_lower])
#ax3_in.scatter(S[good], L_d01[good] - 1, c='k', label="Simulations", zorder=1)
#ax3_in.set_xscale('log')
#ax3_in.set_ylim(0.4, 0.6)


ax1.axhline(second_plot_lower, c='k', lw=0.5)
ax1.axhline(second_plot_upper, c='k', lw=0.5)


ax2.set_yticklabels(())
ax3.yaxis.set_ticks_position('right')
ax3.yaxis.set_label_position('right')
ax2.yaxis.set_ticks_position('right')
ax2.tick_params(axis='y', direction='in')

plt.savefig('erf_3D_penetration_depths.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/erf_3D_penetration_depths.pdf', dpi=300, bbox_inches='tight')
