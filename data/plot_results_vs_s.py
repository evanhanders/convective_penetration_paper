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

fig = plt.figure(figsize=(page_width, page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.00, 0., 0.44, 1])
ax2 = fig.add_axes([0.56, 0., 0.44, 1])

good = (erf == 1) * (ae == 1) * (sf == 0) * (Re == 4e2) * (P == 4) 

ax1.scatter(S[good], d09[good], c='k', label=r"$\delta_{0.9}$", zorder=1, marker='v')
ax1.scatter(S[good], d05[good], c='k', label=r"$\delta_{0.5}$", zorder=1, marker='o')
ax1.scatter(S[good], d01[good], c='k', label=r"$\delta_{0.1}$", zorder=1, marker='^')
leg1 = ax1.legend(frameon=True, fontsize=8, framealpha=0.6, loc='upper right')
ax1.set_xscale('log')

ax1.set_xlabel(r'$\mathcal{S}$')
ax1.set_ylabel(r'$\delta_{\rm{p}}$')

smooth_S = np.logspace(1, 4, 100)
ax2.plot(smooth_S, 1.5*smooth_S**(-1/2), c='orange', label=r'$\mathcal{S}^{-1/2}$')
ax2.scatter(S[good], d09[good] - d01[good], c='k')
ax2.legend(loc='lower left', frameon=True, fontsize=8, framealpha=0.6)
ax2.set_yscale('log')
ax2.set_xlim(80, 4000)
ax2.set_ylim(2e-2, 3e-1)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\mathcal{S}$')
ax2.set_ylabel(r'$\delta_{0.9} - \delta_{0.1}$')


plt.savefig('parameters_vs_s.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/parameters_vs_s.pdf', dpi=300, bbox_inches='tight')
