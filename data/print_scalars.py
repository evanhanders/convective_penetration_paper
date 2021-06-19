import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
golden_ratio = 1.61803398875

dirs = glob.glob('new_linear*/lin*/') + glob.glob('new_erf*/erf*/')

data = []
print('{:5s}, {:5s}, {:5s}, {:10s}, {:10s}, {:10s}'.format('P', 'R', 'S', 'type', 'f', 'Lcz'))
for d in dirs:
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        theory_f = f['f_theory_cz'][()]
        Ls = f['Ls'][()]
        z2 = f['z2'][()]
        L_cz = Ls - z2
        tot_time = times[-1] - times[0]
        time_window = np.min((500, tot_time/2))
        good_times = times > (times[-1] - time_window)
        mean_L_d01 = np.mean(L_d01s[good_times])
        mean_L_d05 = np.mean(L_d05s[good_times])
        mean_L_d09 = np.mean(L_d09s[good_times])
        mean_f = np.mean(theory_f[good_times])
        mean_L_cz = np.mean(L_cz[good_times])
    S = float(d.split('_S')[-1].split('_')[0])
    P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
    Re = float(d.split('_Re')[-1].split('_')[0])
    Lz = float(d.split('_Lz')[-1].split('_')[0])
    if '3D' in d:
        threeD = True
    else:
        threeD = False
    if 'erf' in d:
        dirtype = 'erf'
    else:
        dirtype = 'linear' 
    if 'AE' in d:
        ae = True
    else:
        ae = False

    data.append((S, P, Re, threeD, mean_L_d01, mean_L_d05, mean_L_d09, mean_f, Ls, Lz, dirtype, ae))
    print('{:5.0f}, {:5.0f}, {:5.0f}, {:10s}, {:10f}, {:10f}'.format(P, Re, S, dirtype, mean_f, mean_L_cz))

