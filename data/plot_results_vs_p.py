import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
page_width = 6.5
golden_ratio = 1.61803398875


data = np.genfromtxt('scalar_table_data.csv', skip_header=1, delimiter=',', dtype=np.unicode_)
S  = np.array(data[:,0], dtype=np.float64)
P  = np.array(data[:,1], dtype=np.float64)
Re = np.array(data[:,2], dtype=np.float64)
Lz = np.array(data[:,3], dtype=np.float64)
erf = data[:,5]
sf  = data[:,6]
ae  = data[:,7]
d01 = np.array(data[:,10], dtype=np.float64)
d05 = np.array(data[:,11], dtype=np.float64)
d09 = np.array(data[:,12], dtype=np.float64)
f   = np.array(data[:,13], dtype=np.float64)
xi  = np.array(data[:,14], dtype=np.float64)

tmp = np.zeros_like(erf, dtype=bool)
for i in range(erf.shape[0]):
    if 'False' in erf[i]:
        tmp[i] = 0
    elif 'True' in erf[i]:
        tmp[i] = 1
erf = np.copy(tmp)

tmp = np.zeros_like(sf, dtype=bool)
for i in range(sf.shape[0]):
    if 'False' in sf[i]:
        tmp[i] = 0
    elif 'True' in sf[i]:
        tmp[i] = 1
sf = np.copy(tmp)

tmp = np.zeros_like(ae, dtype=bool)
for i in range(ae.shape[0]):
    if 'False' in ae[i]:
        tmp[i] = 0
    elif 'True' in ae[i]:
        tmp[i] = 1
ae = np.copy(tmp)



fig = plt.figure(figsize=(page_width, 3*page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.00, 0.6, 0.46, 0.4])
ax2 = fig.add_axes([0.54, 0.6, 0.46, 0.4])
ax1_1 = fig.add_axes([0.00, 0.30, 0.46, 0.3])
ax1_2 = fig.add_axes([0.00, 0.00, 0.46, 0.3])
ax2_1 = fig.add_axes([0.54, 0.30, 0.46, 0.3])
ax2_2 = fig.add_axes([0.54, 0.00, 0.46, 0.3])

good = (erf == 1) * (ae == 1) * (sf == 0) * (Re == 4e2) * (S == 1e3) 
#Lcz_erf = (Ls*(fconv_Ls/(0.2*Ls)))[good]
ax1.scatter(P[good], d09[good], c='k', label=r"$\delta_{0.9}$", zorder=1, marker='v')
ax1.scatter(P[good], d05[good], c='k', label=r"$\delta_{0.5}$", zorder=1, marker='o')
ax1.scatter(P[good], d01[good], c='k', label=r"$\delta_{0.1}$", zorder=1, marker='^')
ax1_1.scatter(P[good], f[good], c='k')
ax1_2.scatter(P[good], xi[good], c='k')
leg1 = ax1.legend(frameon=True, fontsize=10, framealpha=0.6, loc='lower right')

line_P = np.linspace(0, 21, 100)
line_theory = 0.13 * line_P
theory_plot = ax1.plot(line_P, line_theory, c='orange', label=r'$\propto \mathcal{P}_D$', zorder=0)

ax1_1.set_xlabel('$\mathcal{P}_D$')
ax1.set_title('$\mathcal{R} = 400, \mathcal{S} = 10^3$')
ax2.set_title('$\mathcal{R} = 800, \mathcal{S} = 10^3$')
ax1.set_ylabel(r'$\delta_{\rm{p}}$')
ax1_1.set_ylabel(r'$f$')
ax1_2.set_ylabel(r'$\xi$')
ax1.set_yticks((0.25, 0.5, 0.75, 1, 1.25))


leg2 = ax1.legend(frameon=True, fontsize=10, framealpha=0.6, loc='upper left')
leg2 = ax1.legend(theory_plot,[r'$\propto \mathcal{P}_D$'],  frameon=True, fontsize=10, framealpha=0.6, loc='upper left')
ax1.add_artist(leg1)


#Linear runs
good = (erf == 0) * (ae == 1) * (sf == 0) * (Re == 8e2) * (S == 1e3) 
ax2.scatter(P[good], d09[good], c='k', zorder=1, marker='v')
ax2.scatter(P[good], d05[good], c='k', zorder=1, marker='o')
ax2.scatter(P[good], d01[good], c='k', zorder=1, marker='^')
ax2_1.scatter(P[good], f[good], c='k')
ax2_2.scatter(P[good], xi[good], c='k')


line_P = np.logspace(-3, 2, 100)
line_theory = 0.25 * line_P**(1/2)
theory_plot = ax2.plot(line_P, line_theory, c='orange', label=r'$\propto \mathcal{P}_L^{1/2}$', zorder=0)
ax2.legend(frameon=True, fontsize=10, framealpha=0.6)


for ax in [ax1, ax2]:
    ax.set_ylim(8e-3, 2)


for ax in [ax1_1, ax2_1]:
    ax.set_ylim(0, 1)
    ax.set_yticks((0, 0.25, 0.5, 0.75))

for ax in [ax1_2, ax2_2]:
    ax.set_ylim(0, 1)
    ax.set_yticks((0, 0.25, 0.5, 0.75))

for ax in [ax1, ax1_1, ax1_2]:
    ax.set_xscale('log')
    ax.set_xlim(8e-2, 15)
ax1.set_yscale('log')

for ax in [ax2, ax2_1, ax2_2]:
    ax.set_xscale('log')
    ax.set_xlim(8e-3, 20)
ax2.set_yscale('log')


for ax in [ax1, ax1_1, ax2, ax2_1]:
    ax.tick_params(axis="x",direction="in", pad=-15, which='both')
    ax.set_xticklabels(())

ax1_2.set_xlabel(r'$\mathcal{P}_D$')
ax2_2.set_xlabel(r'$\mathcal{P}_L$')

plt.savefig('parameters_vs_p.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/parameters_vs_p.pdf', dpi=300, bbox_inches='tight')
