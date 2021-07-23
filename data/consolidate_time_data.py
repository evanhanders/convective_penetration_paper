from collections import OrderedDict
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

from scipy.interpolate import interp1d
import pandas as pd


dirs_p1 = ["erf_AE_cut/erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
           "erf_AE_cut/erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart/"]

dirs_p1_down = ['noslip_erf_P_cut/erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0/']
dirs_p1_up   = ['erf_AE_cut/erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.3']

dirs_p2 = ["erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
           "erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart/",
           "erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart2/",
           "erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart3/"]

dirs_p2_down = ['noslip_erf_P_cut/not_in_use/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.1/']
dirs_p2_up   = ['erf_AE_cut/erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.45']

dirs_p4 = ["erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart2/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart3/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart4/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart5/",
        "erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_schwarzschild_restart6/"]

dirs_p4_down = ['noslip_erf_P_cut/not_in_use/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.4/']
dirs_p4_up   = ['erf_AE_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.7']


name_dirs_p1      = 'erf_re4e2_p1e0_s1e3_longevolution'
name_dirs_p1_down = 'erf_re4e2_p1e0_s1e3_pertDown'
name_dirs_p1_up   = 'erf_re4e2_p1e0_s1e3_pertUp'
name_dirs_p2      = 'erf_re4e2_p2e0_s1e3_longevolution'
name_dirs_p2_down = 'erf_re4e2_p2e0_s1e3_pertDown'
name_dirs_p2_up   = 'erf_re4e2_p2e0_s1e3_pertUp'
name_dirs_p4      = 'erf_re4e2_p4e0_s1e3_longevolution'
name_dirs_p4_down = 'erf_re4e2_p4e0_s1e3_pertDown'
name_dirs_p4_up   = 'erf_re4e2_p4e0_s1e3_pertUp'

dir_lists = [   dirs_p1,
                dirs_p1_down,
                dirs_p1_up,
                dirs_p2,
                dirs_p2_down,
                dirs_p2_up,
                dirs_p4,
                dirs_p4_down,
                dirs_p4_up ]

name_lists = [  name_dirs_p1,
                name_dirs_p1_down,
                name_dirs_p1_up,
                name_dirs_p2,
                name_dirs_p2_down,
                name_dirs_p2_up,
                name_dirs_p4,
                name_dirs_p4_down,
                name_dirs_p4_up ]


for dirs, this_trace_name in zip(dir_lists, name_lists):
    #Parse input parmeters
    data = OrderedDict()
    S = float(dirs[0].split('_S')[-1].split('_')[0])
    P = float(dirs[0].split('_Pr')[0].split('_P')[-1].split('_')[0])
    Re = float(dirs[0].split('_Re')[-1].split('_')[0])
    Lz = float(dirs[0].split('_Lz')[-1].split('_')[0])
    threeD = True
    erf = True

    k_rz = 0.2 / (S * P)
    t_diff = 1/k_rz

    #Read in scalar and profile data
    times = []
    L_d01s = []
    L_d05s = []
    L_d09s = []
    theory_f = []
    theory_xi = []
    grad = []
    prof_times = []
    for i, d in enumerate(dirs):
        if i == 0 and 'longevolution' in d:
            skip_points = 100
        else:
            skip_points = 0
        with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
            dictionary = dict()
            for k in ['times', 'L_d01s', 'L_d05s', 'L_d09s']:
                dictionary[k] = f[k][()]
                if k != 'times':
                    dictionary[k] = dictionary[k][np.unique(dictionary['times'], return_index=True)[1]]
            try:
                for k in ['modern_f', 'modern_xi']:
                    dictionary[k] = f[k][()]
                    dictionary[k] = dictionary[k][np.unique(dictionary['times'], return_index=True)[1]]

            except:
                #some older files don't have modern_f/modern_xi
                Ls = f['Ls'][()]
                L_d09s_tmp = f['L_d09s'][()]
                enstrophy_Ls = f['enstrophy_Ls'][()]
                enstrophy_cz = f['enstrophy_cz'][()]
                fconv_Ls = f['fconv_Ls'][()]

                L_pz = L_d09s_tmp - Ls
                dissipation_cz   = enstrophy_cz/Re
                dissipation_Ls   = enstrophy_Ls/Re
                dissipation_pz   = (dissipation_cz * L_d09s_tmp - dissipation_Ls * Ls) / L_pz
       
                modern_f         = dissipation_Ls/fconv_Ls
                modern_xi        = ( (dissipation_pz*L_pz) / (fconv_Ls*Ls) ) / (modern_f * (L_pz/Ls))
                dictionary['modern_f'] = modern_f
                dictionary['modern_xi'] = modern_xi
                for k in ['modern_f', 'modern_xi']:
                    dictionary[k] = dictionary[k][np.unique(dictionary['times'], return_index=True)[1]]



            dictionary['times'] = np.unique(dictionary['times'])
            if i > 0:
                #clean out some time artifacts from restarting simulations
                skip_points = int(np.sum(dictionary['times'] <= times[-1][-1]))
            
            times.append(dictionary['times'][skip_points:])
            L_d01s.append(dictionary['L_d01s'][skip_points:])
            L_d05s.append(dictionary['L_d05s'][skip_points:])
            L_d09s.append(dictionary['L_d09s'][skip_points:])
            theory_f.append(dictionary['modern_f'][skip_points:])
            theory_xi.append(dictionary['modern_xi'][skip_points:])
            Ls = f['Ls'][()]
        with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
            z = f['z'][()]
            grad_ad = -f['T_ad_z'][()][0,:].squeeze()
            grad_rad = -f['T_rad_z'][()][0,:].squeeze()

            grad.append(-f['T_z'][()])
            prof_times.append(f['T_z_times'][()])

    times          = np.array(np.concatenate(times))
    L_d01s         = np.array(np.concatenate(L_d01s))
    L_d05s         = np.array(np.concatenate(L_d05s))
    L_d09s         = np.array(np.concatenate(L_d09s))
    theory_f       = np.array(np.concatenate(theory_f))
    theory_xi      = np.array(np.concatenate(theory_xi))

    grad           = np.array(np.concatenate(grad, axis=0))
    prof_times     = np.array(np.concatenate(prof_times))

    with h5py.File('temporal_data/{}.h5'.format(this_trace_name), 'w') as f:
        f['scalars/Ls']  = Ls
        f['scalars/d01'] = L_d01s - Ls
        f['scalars/d05'] = L_d05s - Ls
        f['scalars/d09'] = L_d09s - Ls
        f['scalars/times'] = times
        f['scalars/f']     = theory_f
        f['scalars/xi']    = theory_xi
        f['profiles/z']     = z
        f['profiles/grad']  = grad
        f['profiles/times'] = prof_times
        f['profiles/grad_ad'] = grad_ad
        f['profiles/grad_rad'] = grad_rad

