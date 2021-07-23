import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

import dedalus.public as de
from scipy.interpolate import interp1d
from scipy.optimize import brentq

col_width = 3.25
page_width = 6.5
golden_ratio = 1.61803398875

dirs = glob.glob('noslip_linear_P_cut/*P1e0*/')
dirs.sort(key = lambda x: float(x.split('_Re')[-1].split('_')[0]))

#nz = 1024
#Lz = 2
#z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
#domain = de.Domain([z_basis], grid_dtype=np.float64)
#ke_rhs = domain.new_field()
#ke_flux = domain.new_field()
#domain_z = domain.grid(-1)

cmap = "viridis_r"
norm = mpl.colors.Normalize(vmin=np.log10(50), vmax=3.5)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

data = []
for i, d in enumerate(dirs):
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
        fconv_cz = f['fconv_cz'][()]
        Ls = np.mean(f['Ls'][()])
        tot_time = times[-1] - times[0]
        time_window = np.min((500, tot_time/2))
        good_times = times > (times[-1] - time_window)
        mean_L_d01 = np.mean(L_d01s[good_times])
        mean_L_d05 = np.mean(L_d05s[good_times])
        mean_L_d09 = np.mean(L_d09s[good_times])
        mean_f  = np.mean(modern_f[good_times])
        std_f   = np.std(modern_f[good_times])/np.sqrt(len(modern_f[good_times]))
        mean_xi = np.mean(modern_xi[good_times])
        mean_fconv_Ls = np.mean(fconv_Ls[good_times])
        mean_fconv_cz = np.mean(fconv_cz[good_times])
    if 'stressfree' in d:
        sf = True
    else:
        sf = False



    with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
        print(f.keys())
        if 'F_KE_p' not in f.keys():
            pz_visc_bl = 0
            bot_visc_bl = 0
        else:
            enstrophy_profiles = f['enstrophy'][()]
            vel_rms_profiles = f['vel_rms'][()]
            fconv_profiles = f['F_conv'][()]
            f_ke_profiles = f['F_KE_p'][()]
            grad_profiles = -f['T_z'][()]
            grad_ad_profiles = -f['T_ad_z'][()]
            grad_rad_profiles = -f['T_rad_z'][()]
            z = f['z'][()]

            dissipation = enstrophy_profiles[-2,:]/Re
#            F_KE = fconv_profiles[-2,:] - dissipation #- np.gradient(f_ke_profiles[-2,:], z)
            F_visc = fconv_profiles[-2,:] - dissipation - np.gradient(f_ke_profiles[-2,:], z)

            if not sf:
                bot_visc_bl = z[(np.argmax(F_visc[z < 1]))]
                visc_bl_upper_bot = z[(z > 1.2)*(z < 1.8)][F_visc[(z > 1.2)*(z < 1.8)] < np.min(F_visc[(z > 1.2)*(z < 1.8)])*0.2][0]
                visc_bl_upper_top = z[(z > 1.2)*(z < 1.8)][F_visc[(z > 1.2)*(z < 1.8)] < np.min(F_visc[(z > 1.2)*(z < 1.8)])*0.2][-1]
            else:
                bot_visc_bl = z[(np.argmin(F_visc[z < 1]))]
                visc_bl_upper_bot = z[z > 1][F_visc[z > 1] < np.min(F_visc[z > 1])*0.25][0]
                visc_bl_upper_top = z[z > 1][F_visc[z > 1] < np.min(F_visc[z > 1])*0.25][-1]
#            plt.plot(z, enstrophy_profiles[-2, :])
#            plt.axvline(bot_visc_bl)
#            plt.ylabel(sf)
#            plt.xlabel(Re)
#            plt.show()

            pz_visc_bl = visc_bl_upper_top - visc_bl_upper_bot


    data.append((S, P, Re, Ls, Lz, sf, mean_L_d01, mean_L_d05, mean_L_d09, mean_f, mean_xi, mean_fconv_Ls, mean_fconv_cz, bot_visc_bl, pz_visc_bl, std_f))
data = np.array(data)
S = data[:,0]
P = data[:,1]
Re = data[:,2]
Ls = data[:,3]
Lz = data[:,4]
sf = data[:,5]

L_d01 = data[:,6]
L_d05 = data[:,7]
L_d09 = data[:,8]
theory_f  = data[:,9]
theory_xi = data[:,10]
fconv_Ls = data[:,11]
fconv_cz = data[:,12]
bot_bl = data[:,13]
pz_bl  = data[:,14]
std_f  = data[:,15]

print(bot_bl, Ls, theory_f, theory_f/(1 - 2*bot_bl/Ls))
