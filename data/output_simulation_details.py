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

fields = dict()
fields['t_sim'] = 'times'
fields['d_01'] = 'L_d01s'
fields['d_05'] = 'L_d05s'
fields['d_09'] = 'L_d09s'
fields['f']  = 'modern_f'
fields['xi'] = 'modern_xi'
fields['Ls']    = 'Ls'
fields['volavg_enstrophy_Ls']    = 'enstrophy_Ls'
fields['volavg_enstrophy_CZ+PZ'] = 'enstrophy_cz'
fields['volavg_fconv_Ls']        = 'fconv_Ls'
fields['volavg_velocity_Ls'] = 'vel_Ls'

data_chunks = []
for dnum, d in enumerate(dirs):
    data = dict()
    S = float(d.split('_S')[-1].split('_')[0])
    P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
    Re = float(d.split('_Re')[-1].split('_')[0])
    Lz = float(d.split('_Lz')[-1].split('_')[0])
    resolution = d.split('Titer')[-1].split('_')[1].split('/')[0]


    if '3D' in d:
        threeD = True
    else:
        threeD = False
    if 'erf' in d:
        erf = True
    else:
        erf = False
    if 'stressfree' in d:
        sf = True
    else:
        sf = False
    if 'schwarzschild' in d:
        ae = False
    else:
        ae = True


    data['S'] = S
    data['P'] = P
    data['R'] = Re
    data['Lz'] = Lz
    data['3D'] = threeD
    data['erf'] = erf
    data['sf']  = sf
    data['ae'] = ae
    data['resolution'] = resolution

    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        N = len(f['times'][()])
        if N > 2000:
            N = 1000
        else:
            N /= 2
            N = int(N)
        try:
            modern_f = f['modern_f'][()]
            modern_xi = f['modern_xi'][()]
        except:
            #some older files don't have modern_f/modern_xi
            Ls = f['Ls'][()]
            L_d09s = f['L_d09s'][()]
            enstrophy_Ls = f['enstrophy_Ls'][()]
            enstrophy_cz = f['enstrophy_cz'][()]
            fconv_Ls = f['fconv_Ls'][()]

            L_pz = L_d09s - Ls
            dissipation_cz   = enstrophy_cz/Re
            dissipation_Ls   = enstrophy_Ls/Re
            dissipation_pz   = (dissipation_cz * L_d09s - dissipation_Ls * Ls) / L_pz
       
            modern_f         = dissipation_Ls/fconv_Ls
            modern_xi        = ( (dissipation_pz*L_pz) / (fconv_Ls*Ls) ) / (modern_f * (L_pz/Ls))


        for outk, ink in fields.items():
            if ink == 'times':
                data[outk] = f['times'][-1]
            elif ink == 'Ls':
                data[outk] = f['Ls'][()]
            else:
                if ink == 'modern_f':
                    series = modern_f
                elif ink == 'modern_xi':
                    series = modern_xi
                else:
                    series = f[ink][()]
                data[outk] = np.mean(series[-N:])
        for k in ['d_01', 'd_05', 'd_09']:
            data[k] -= data['Ls']
    data_list = []
    if dnum == 0:
        field_list = []
    for k, item in data.items():
        data_list.append(item)
        if dnum == 0:
            field_list.append(k)
    data_chunks.append(data_list)

data_chunks.sort(key = lambda x: x[2])
data_chunks.sort(key = lambda x: x[1])
data_chunks.sort(key = lambda x: x[0])
data_chunks.sort(key = lambda x: x[6])#stressfree at bottom
data_chunks.sort(key = lambda x: 1-x[5])#erf -- linear at bottom.
data_chunks.sort(key = lambda x: x[7])#ae -- schwarzschild at top.

file1 = open("scalar_table_data.csv","w")
for j, c in enumerate(data_chunks):
    c1 = []
    strbase = ''
    for i, d in enumerate(c):
        if i == 8:
            c1.append(d)
            strbase += '{:11}, '
        elif i in [4, 5, 6, 7]:
            c1.append('{}'.format(d))
            strbase += '{:5}, '
        elif i in [0, 1, 2]:
            c1.append('{:.1e}'.format(d))
            strbase += '{:7}, '
        elif i in [16, 17]:
            c1.append('{:.3f}'.format(d))
            strbase += '{:7}, '
        elif i == 3:
            c1.append('{:.1f}'.format(d))
            strbase += '{:3}, '
        elif i in [len(c) - 1]:
            c1.append('{:.3f}'.format(d))
            strbase += '{}\n'
        elif i == 9:
            c1.append('{}'.format(int(d)))
            strbase += '{:5}, '
        else:
            c1.append('{:.3f}'.format(d))
            strbase += '{:5}, '
    if j == 0:
        print(strbase.format(*tuple(field_list)))
        file1.write(strbase.format(*tuple(field_list)))
    print(strbase.format(*tuple(c1)))
    file1.write(strbase.format(*tuple(c1)))
file1.close()
            

