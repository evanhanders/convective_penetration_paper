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

dirs = glob.glob('noslip_erf_Re_cut/erf*/')
dirs += ['noslip_erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256/']
dirs += glob.glob('stressfree_erf_Re_cut/erf*/')
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
        time_window = np.min((500, tot_time-50))
        time_window = np.min((500, tot_time-50))
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
bot_bl = 2*data[:,13]
pz_bl  = data[:,14]
std_f  = data[:,15]

fig = plt.figure(figsize=(page_width, 2*page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.00, 0.60, 0.25, 0.40])
ax2 = fig.add_axes([0.35, 0.60, 0.25, 0.40])
ax3 = fig.add_axes([0.70, 0.60, 0.25, 0.40])
ax4 = fig.add_axes([0.00, 0.00, 0.25, 0.40])
ax5 = fig.add_axes([0.35, 0.00, 0.25, 0.40])
ax6 = fig.add_axes([0.70, 0.00, 0.25, 0.40])

good = sf == 1
ax1.scatter(Re[good], (L_d09 - Ls)[good], facecolor=(1,1,1,0.5), color='k', marker='v')
ax1.scatter(Re[good], (L_d05 - Ls)[good], facecolor=(1,1,1,0.5), color='k', marker='o', label='SF')
ax1.scatter(Re[good], (L_d01 - Ls)[good], facecolor=(1,1,1,0.5), color='k', marker='^')
good = sf == 0
ax1.scatter(Re[good], (L_d09 - Ls)[good], color='k', marker='v', alpha=0.8, s=15)
ax1.scatter(Re[good], (L_d05 - Ls)[good], color='k', marker='o', label='NS', alpha=0.8, s=15)
ax1.scatter(Re[good], (L_d01 - Ls)[good], color='k', marker='^', alpha=0.8, s=15)
ax1.set_xscale('log')
ax1.legend(frameon=True, fontsize=8, framealpha=0.6)
ax1.set_xlabel(r'$\mathcal{R}$')
ax1.set_ylabel(r'$\delta_{0.5}$')
ax1.set_xlim(2e1, 7e3)


good = sf == 1
ax2.scatter(Re[good], theory_f[good], facecolor=(1,1,1,0.5), color='k', marker='o')
good = sf == 0
ax2.scatter(Re[good], theory_f[good], color='k', marker='o', alpha=0.8, s=15)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\mathcal{R}$')
ax2.set_ylabel(r'$f$')
ax2.set_xlim(2e1, 7e3)

good = (sf == 1)
ax3.scatter(Re[good], theory_xi[good], facecolor=(1,1,1,0.5), color='k', marker='o')
good = sf == 0
ax3.scatter(Re[good], theory_xi[good], color='k', marker='o', alpha=0.8, s=15)
ax3.set_xscale('log')
ax3.set_ylabel(r'$\xi$')
ax3.set_xlabel(r'$\mathcal{R}$')
ax3.set_xlim(2e1, 7e3)

good = (sf == 1)*(Re >= 200)
ax4.scatter(theory_f[good], (L_d05 - Ls)[good], facecolor=(1,1,1,0.5), color='k', marker='o')
good = (sf == 0)*(Re >= 200)
ax4.scatter(theory_f[good], (L_d05 - Ls)[good], color='k', marker='o', alpha=0.8, s=15)
ax4.set_xlabel(r'$f$')
ax4.set_ylabel(r'$\delta_{0.5}$')
ax4.set_xlim(0.63, 0.75)
ax4.set_ylim(0.3, 0.55)



good = (sf == 1)*(Re >= 200)
ax5.errorbar(bot_bl[good], theory_f[good], yerr=std_f[good], lw=0, markerfacecolor=(1,1,1,1), color='k', marker='o')
good = (sf == 0)*(Re >= 200)
ax5.scatter(bot_bl[good], theory_f[good], color='k', marker='o', alpha=0.8, s=15)
#ax5.errorbar(bot_bl[good], theory_f[good], yerr=std_f[good], lw=0, color='k', marker='o', alpha=0.8, markersize=5)
ax5.set_xlim(0, 0.24)
ax5.set_xlabel(r'$\ell_\nu$')
ax5.set_ylabel(r'$f$')

bl = np.linspace(-0.05, 0.4, 100)
f_bl = 0.755*(1 - bl/Ls[good][0])
ax5.plot(bl, f_bl, c='orange', label=r'$0.755(1 - \ell_\nu/L_s)$')
ax5.set_ylim(0.6, 0.8)
ax5.legend(frameon=True, fontsize=8, framealpha=0.6)

xi = 0.6
delta_func = lambda f_val: 0.9 * 4 * (1 - f_val) / ( 1 + xi * f_val * 4)
delta = delta_func(f_bl)
ax4.plot(f_bl, delta, c='orange', label=r'$0.9 \cdot \rm{Eqn}.~17$')
ax4.legend(frameon=True, fontsize=8, framealpha=0.6)

ax1.axhline(delta_func(0.755), c='orange')


R = np.logspace(1, 4, 100)
ax6.plot(R, 8*R**(-2/3), c='orange', label=r'$ 8 \mathcal{R}^{-2/3}$')
good = (sf == 1)*(Re >= 200)
ax6.scatter(Re[good], bot_bl[good], facecolor=(1,1,1,0.5), color='k', marker='o')
good = (sf == 0)*(Re >= 200)
ax6.scatter(Re[good], bot_bl[good], color='k', marker='o', alpha=0.8, s=15)
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlabel(r'$\mathcal{R}$')
ax6.set_ylabel(r'$\ell_\nu$')
ax6.legend(frameon=True, fontsize=8, framealpha=0.6)
ax6.set_xlim(1.5e2, 7e3)
ax6.set_ylim(1.4e-2, 6e-1)

#for ax in [ax1, ax2, ax3]:
#    ax.xaxis.set_ticks_position('top')
#    ax.xaxis.set_label_position('top')

print(Ls, bot_bl[(sf == 1)*(Re >= 200)], Re[(sf == 1)*(Re >= 200)], delta_func(0.734), delta_func(0.755))


plt.savefig('parameters_vs_re.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/parameters_vs_re.pdf', dpi=300, bbox_inches='tight')
