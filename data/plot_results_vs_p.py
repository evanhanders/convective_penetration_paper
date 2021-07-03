import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from matplotlib.patches import ConnectionPatch

col_width = 3.25
page_width = 6.5
golden_ratio = 1.61803398875

dirs = glob.glob('noslip_linear_P_cut/linear*')
dirs += glob.glob('noslip_erf_P_cut/erf*')

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

fig = plt.figure(figsize=(page_width, 2*page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.00, 0.5, 0.46, 0.5])
ax2 = fig.add_axes([0.54, 0.5, 0.46, 0.5])
ax1_1 = fig.add_axes([0.00, 0.25, 0.46, 0.25])
ax1_2 = fig.add_axes([0.00, 0.00, 0.46, 0.25])
ax2_1 = fig.add_axes([0.54, 0.25, 0.46, 0.25])
ax2_2 = fig.add_axes([0.54, 0.00, 0.46, 0.25])
second_plot_lower = 0.45
second_plot_upper = 0.9


good = (erf == 1) * (Re == 4e2) * (S == 1e3) 
#Lcz_erf = (Ls*(fconv_Ls/(0.2*Ls)))[good]
ax1.scatter(P[good], (L_d09[good] - Ls[good]), c='k', label=r"$\delta_{0.9}$", zorder=1, marker='v')
ax1.scatter(P[good], (L_d05[good] - Ls[good]), c='k', label=r"$\delta_{0.5}$", zorder=1, marker='o')
ax1.scatter(P[good], (L_d01[good] - Ls[good]), c='k', label=r"$\delta_{0.1}$", zorder=1, marker='^')
ax1_1.scatter(P[good], theory_f[good], c='k')
ax1_2.scatter(P[good], theory_xi[good], c='k')
leg1 = ax1.legend(frameon=True, fontsize=8, framealpha=0.6, loc='lower right')

line_P = np.linspace(0, 21, 100)
#line_f = 0.65
#line_xi = 0.55
#line_theory =  line_P * (1 - line_f) / (1 + line_P*line_f*line_xi)
#theory_plot = ax1.plot(line_P, line_theory, c='orange', label=r'theory ($f = {{{:.2f}}}, \xi = {{{:.2f}}}$)'.format(line_f, line_xi), zorder=0)
#ax1_1.axhline(line_f, c='orange', lw=0.5, zorder=0)
#ax1_2.axhline(line_xi, c='orange', lw=0.5, zorder=0)
line_theory = 0.13 * line_P
theory_plot = ax1.plot(line_P, line_theory, c='orange', label=r'$\propto \mathcal{P}_D$', zorder=0)

ax1_1.set_xlabel('$\mathcal{P}_D$')
ax1.set_title('$\mathcal{P}_D|_{\mathcal{R} = 400, \mathcal{S} = 10^3}$')
ax2.set_title('$\mathcal{P}_L|_{\mathcal{R} = 800, \mathcal{S} = 10^3}$')
ax1.set_ylabel(r'$\delta_{\rm{p}}$')
ax1_1.set_ylabel(r'$f$')
ax1_2.set_ylabel(r'$\xi$')
for ax in [ax1, ax1_1]:
    ax.set_xlim(0, 11)
ax1.set_ylim(7e-2, 1)
ax1.set_yticks((0.25, 0.5, 0.75, 1, 1.25))


leg2 = ax1.legend(frameon=True, fontsize=8, framealpha=0.6, loc='upper left')
#leg2 = ax1.legend(theory_plot,[r'theory ($f = {{{:.2f}}}, \xi = {{{:.2f}}}$)'.format(line_f, line_xi)],  frameon=True, fontsize=8, framealpha=0.6, loc='upper left')
leg2 = ax1.legend(theory_plot,[r'$\propto \mathcal{P}_D$'],  frameon=True, fontsize=8, framealpha=0.6, loc='upper left')
ax1.add_artist(leg1)


#Linear runs
good = (erf == 0) * (Re == 8e2) * (S == 1e3) 
#Lcz_linear = (Ls*(fconv_Ls/0.1)**(1/2))[good]
ax2.scatter(P[good], (L_d09[good] - Ls[good]), c='k', zorder=1, marker='v')
ax2.scatter(P[good], (L_d05[good] - Ls[good]), c='k', zorder=1, marker='o')
ax2.scatter(P[good], (L_d01[good] - Ls[good]), c='k', zorder=1, marker='^')
ax2_1.scatter(P[good], theory_f[good], c='k')
ax2_2.scatter(P[good], theory_xi[good], c='k')


line_P = np.logspace(-3, 2, 100)
#line_f = 0.78
#line_xi = 2/3
#zeta = (line_xi*line_f/2)*np.sqrt(line_P/(1-line_f))
#line_theory = np.sqrt(line_P * (1 - line_f)) * (np.sqrt(zeta**2 + 1) - zeta)
#ax2.plot(line_P, line_theory, c='orange', label=r'theory ($f = {{{:.2f}}}, \xi = {{{:.2f}}}$)'.format(line_f, line_xi), zorder=0)
#ax2.legend(frameon=True, fontsize=8, framealpha=0.6)
#ax2_1.axhline(line_f, c='orange', lw=0.5, zorder=0)
#ax2_2.axhline(line_xi, c='orange', lw=0.5, zorder=0)
line_theory = 0.25 * line_P**(1/2)
theory_plot = ax2.plot(line_P, line_theory, c='orange', label=r'$\propto \mathcal{P}_L^{1/2}$', zorder=0)
ax2.legend(frameon=True, fontsize=8, framealpha=0.6)





for ax in [ax1_1, ax2_1]:
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks((0.5, 0.7, 0.9))

for ax in [ax1_2, ax2_2]:
    ax.set_ylim(0.4, 0.85)

for ax in [ax1, ax1_1, ax1_2]:
    ax.set_xscale('log')
    ax.set_xlim(8e-1, 15)
ax1.set_yscale('log')

for ax in [ax2, ax2_1, ax2_2]:
    ax.set_xscale('log')
    ax.set_xlim(8e-3, 20)
ax2.set_yscale('log')


for ax in [ax1, ax1_1, ax2, ax2_1]:
    ax.set_xticklabels(())

ax1_2.set_xlabel(r'$\mathcal{P}_D$')
ax2_2.set_xlabel(r'$\mathcal{P}_L$')

plt.savefig('parameters_vs_p.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/parameters_vs_p.pdf', dpi=300, bbox_inches='tight')
