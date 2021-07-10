"""
Outputs sim info for table

Usage:
    plot_slices.py <dirs>... [options]

"""

from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

import pandas as pd

from docopt import docopt
args = docopt(__doc__)

dirs = args['<dirs>']

data_chunks = []
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
        L_d09s = f['L_d09s'][()]
        enstrophy_Ls = f['enstrophy_Ls'][()]
        enstrophy_cz = f['enstrophy_cz'][()]
        fconv_Ls = f['fconv_Ls'][()]
        vel_Ls = f['vel_Ls'][()]

        try:
            modern_f = f['modern_f'][()]
            modern_xi = f['modern_xi'][()]
        except:
            L_pz = L_d09s - Ls
            dissipation_cz   = enstrophy_cz/Re
            dissipation_Ls   = enstrophy_Ls/Re
            dissipation_pz   = (dissipation_cz * L_d09s - dissipation_Ls * Ls) / L_pz
       
            modern_f         = dissipation_Ls/fconv_Ls
            modern_xi        = ( (dissipation_pz*L_pz) / (fconv_Ls*Ls) ) / (modern_f * (L_pz/Ls))

    if P == 0.03 and erf == False:
        N = 900
    else:
        N = 1000

    data_chunks.append((P, Re, S, times[-1], np.mean((L_d01s-Ls)[-N:]), np.mean((L_d05s-Ls)[-N:]), np.mean((L_d09s-Ls)[-N:]), np.mean(modern_f[-N:]), np.mean(modern_xi[-N:]), np.mean(vel_Ls[-N:]) ))
data_chunks.sort(key = lambda x: x[2])
data_chunks.sort(key = lambda x: x[1])
data_chunks.sort(key = lambda x: x[0])



for c in data_chunks:
    print('P: {}, R: {}, S: {},  t: {:.4e}, ds: {:.3f}, {:.3f}, {:.3f}, f: {:.2f}, xi: {:.2f}, u: {:.2f}'.format(*tuple(c)))
            

