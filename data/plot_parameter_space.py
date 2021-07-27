from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from scipy.optimize import brentq
from scipy.interpolate import interp1d

import pandas as pd

col_width = 3.25
golden_ratio = 1.61803398875

brewer_green  = np.array(( 27,158,119, 255))/255
brewer_orange = np.array((217, 95,  2, 255))/255
brewer_purple = np.array((117,112,179, 255))/255

data = np.genfromtxt('scalar_table_data.csv', skip_header=1, delimiter=',', dtype=np.unicode_)
S  = np.array(data[:,0], dtype=np.float64)
P  = np.array(data[:,1], dtype=np.float64)
Re = np.array(data[:,2], dtype=np.float64)
erf = data[:,5]
sf  = data[:,6]
ae  = data[:,7]
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

fig = plt.figure(figsize=(col_width, col_width/golden_ratio))

ax = fig.add_subplot(1,1,1)

cut1 = (ae == True)*(sf == False)*(erf == True)*(S == 1e3)
cut2 = (ae == True)*(sf == True)*(erf == True)*(S == 1e3)
cut3 = (ae == True)*(sf == False)*(erf == False)*(S == 1e3)
case2 = ax.scatter(P[cut3], Re[cut3], marker='d', c=brewer_orange, s=20, label='Case II')
case1sf = ax.scatter(P[cut2], Re[cut2], marker='o', edgecolor=brewer_purple, s=40, facecolor=(1,1,1,0), linewidth=0.5, label='Case I/SF')
case1 = ax.scatter(P[cut1], Re[cut1], marker='o', c=brewer_purple, s=10, label='Case I')

hh=[case1, case1sf, case2]
ax.legend(hh,[H.get_label() for H in hh], fontsize=9)


ax.scatter(4, 400, marker='s', s=80, edgecolor=brewer_purple, facecolor=(1,1,1,0))
ax.scatter(4, 800, marker='s', s=80, edgecolor=brewer_orange, facecolor=(1,1,1,0))

ax.scatter(4, 400, marker='x', s=60, c=brewer_green)
ax.set_xlabel(r'$\mathcal{P}$')
ax.set_ylabel(r'$\mathcal{R}$')




ax.set_xscale('log')
ax.set_yscale('log')

fig.savefig('parameter_space.png', dpi=300, bbox_inches='tight')
fig.savefig('../manuscript/parameter_space.pdf', dpi=300, bbox_inches='tight')
