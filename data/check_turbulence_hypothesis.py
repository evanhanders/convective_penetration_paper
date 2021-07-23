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

dirs = glob.glob('new_erf_Re*/erf*') 
dirs += ["new_erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.4/"]
dirs.sort(key = lambda x: float(x.split('_Re')[-1].split('_')[0]))

nz = 1024
Lz = 2
z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
domain = de.Domain([z_basis], grid_dtype=np.float64)
ke_rhs = domain.new_field()
ke_flux = domain.new_field()
domain_z = domain.grid(-1)

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

fig2 = plt.figure()
ax3 = fig2.add_subplot(1,1,1)

fig3 = plt.figure()
ax4 = fig3.add_subplot(1,1,1)

fig4 = plt.figure()
ax5 = fig4.add_subplot(1,1,1)


fig5 = plt.figure()
ax6 = fig5.add_subplot(1,1,1)

fig6 = plt.figure()
ax7 = fig6.add_subplot(1,1,1)


cmap = "viridis_r"
norm = mpl.colors.Normalize(vmin=np.log10(50), vmax=3.5)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)




data = []
for i, d in enumerate(dirs):
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        theory_f = f['f_theory_cz'][()]
        Ls = f['Ls'][()]
        tot_time = times[-1] - times[0]
        time_window = np.min((500, tot_time/2))
        good_times = times > (times[-1] - time_window)
        mean_L_d01 = np.mean(L_d01s[good_times])
        mean_L_d05 = np.mean(L_d05s[good_times])
        mean_L_d09 = np.mean(L_d09s[good_times])
        mean_f = np.mean(theory_f[good_times])
    S = float(d.split('_S')[-1].split('_')[0])
    P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
    Re = float(d.split('_Re')[-1].split('_')[0])
    Lz = float(d.split('_Lz')[-1].split('_')[0])
    with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
        enstrophy_profiles = f['enstrophy'][()]
        vel_rms_profiles = f['vel_rms'][()]
        fconv_profiles = f['F_conv'][()]
        f_ke_profiles = f['F_KE_p'][()]
        grad_profiles = -f['T_z'][()]
        grad_ad_profiles = -f['T_ad_z'][()]
        grad_rad_profiles = -f['T_rad_z'][()]
        z = f['z'][()]

        ke_rhs.set_scales(len(z)/nz, keep_data=False)
        ke_rhs['g'] = fconv_profiles[-2,:] - enstrophy_profiles[-2,:]/Re
        ke_rhs.antidifferentiate('z', ('left', 0), out=ke_flux)
        ke_flux.set_scales(1, keep_data=True)
        true_z2 = brentq(interp1d(domain_z, ke_flux['g']), 0.05, 0.9)
        guess_z2 = brentq(interp1d(z, f_ke_profiles[-2,:]), 0.05, 0.9)
        print(Re, true_z2, guess_z2)
#    if i != 0:
#        ax1.scatter(Re, np.mean(Ls) - guess_z2, color='orange')
#        ax1.scatter(Re, np.mean(Ls) - true_z2, color='black')
#    else:
#        ax1.scatter(Re, np.mean(Ls) - guess_z2, color='orange', label=r'neglect $F_{\rm{visc}}$')
#        ax1.scatter(Re, np.mean(Ls) - true_z2, color='black', label=r'use $F_{\rm{visc}}$')
#    theory_f = 0.65
#    Lcz_true = np.mean(Ls) - true_z2
#    Lcz_guess = np.mean(Ls) - guess_z2
#    theory_dp_true = Lcz_true * P * (1 - theory_f)/(1 + theory_f*P/2)
#    theory_dp_guess = Lcz_guess * P * (1 - theory_f)/(1 + theory_f*P/2)
#    ax2.scatter(Re, theory_dp_guess, color='orange')
#    ax2.scatter(Re, theory_dp_true, color='black')

    color = sm.to_rgba(np.log10(Re))
    z_bl = 0.1#z[z < 0.5][np.argmin(enstrophy_profiles[-2,z < 0.5])]
    if i != 0:
        ax1.scatter(Re, mean_L_d09 - Ls, c='k')
    else:
        ax1.scatter(Re, mean_L_d09 - Ls, c='k', label=r'$\delta_{0.9}$')

    N = -2
    dissipation_bl = np.sum(enstrophy_profiles[N,z <= z_bl]*np.gradient(z)[z <= z_bl])/Re
    dissipation_bulk = np.sum(enstrophy_profiles[N,(z <= Ls)*(z > z_bl)]*np.gradient(z)[(z <= Ls)*(z > z_bl)])/Re
    buoyancy_bl   = np.sum(fconv_profiles[N,z <= z_bl]*np.gradient(z)[z <= z_bl])
    buoyancy_bulk = np.sum(fconv_profiles[N,(z <= Ls)*(z > z_bl)]*np.gradient(z)[(z <= Ls)*(z > z_bl)])

    dissipation_full = np.sum(enstrophy_profiles[N,:]*np.gradient(z)/Re)
    buoyancy_full    = np.sum(fconv_profiles[N,:]*np.gradient(z))
    print('1 - dissipation / buoyancy: {:.3e}'.format(1 - dissipation_full/buoyancy_full))

    mean_u2 = np.sum((vel_rms_profiles[N,:]**2*np.gradient(z))[(z > 0.5)*(z < Ls)]) / (Ls - 0.5)
    ax7.scatter(Re, mean_u2, c='k')

    f = (dissipation_bulk + dissipation_bl)/(buoyancy_bulk + buoyancy_bl)
    theory_value = P*(1 - f)/(1 + P*f/2) * Ls
    if i != 0:
        ax1.scatter(Re, theory_value, c='orange')
    else:   
        ax1.scatter(Re, theory_value, c='orange', label='theory')
#    ax2.scatter(Re, dissipation_bl, c=color)
#    ax2.scatter(Re, dissipation_bulk, c='orange')
    ax2.scatter(Re, (dissipation_bulk + dissipation_bl)/(buoyancy_bulk + buoyancy_bl), c='k')
    if Re >= 100:
        ax3.axvline(np.mean(Ls), c='k')
#        ax3.plot(z, enstrophy_profiles[-2,:]/Re/0.2 / (vel_rms_profiles[-2,:]/vel_rms_profiles[-2,:].max())**2, c=color, label=Re)
        ax3.plot(z, enstrophy_profiles[-2,:]/Re/0.2, c=color, label=Re)
        ax3.plot(z, fconv_profiles[-2,:]/0.2, c=color)
#        ax3.axvline(z_bl, c=color)
#        ax3.axvline(mean_L_d09, c=color)
    if Re >= 100:
        ax4.axvline(np.mean(Ls), c='k')
#        ax4.plot(z, np.gradient(f_ke_profiles[-2,:], z), c=color, label=Re)
#        ax4.plot(z, fconv_profiles[-2,:] - enstrophy_profiles[-2,:]/Re, c=color, ls='--')
        F_visc = fconv_profiles[-2,:] - enstrophy_profiles[-2,:]/Re - np.gradient(f_ke_profiles[-2,:], z)
        sum_visc = np.sum((F_visc*np.gradient(z))[z < 0.5])
        print('sum visc: {:.3e}'.format(sum_visc))
        ax4.plot(z, F_visc, c=color)

        visc_bl = z[F_visc < 0][0]
        visc_bl_upper = z[z > 1][np.argmin(F_visc[z > 1])]
        visc_bl_upper1 = z[z > 1][np.argmax(F_visc[z > 1])]
        visc_bl_upper2 = z[np.abs(F_visc) > np.abs(F_visc[z > 1]).max()*0.1][-1]
        ax3.axvline(visc_bl, c=color)
        ax4.axvline(visc_bl, c=color)
        ax4.axvline(visc_bl_upper, c=color)
        ax4.axvline(visc_bl_upper2, c=color)
#        ax4.plot(z, vel_rms_profiles[-2,:]**2, c=color, label=Re)
        ax5.scatter(Re, visc_bl, c='k', marker='s')
        ax5.scatter(Re, visc_bl_upper2-visc_bl_upper, c='green')
        ax5.scatter(Re, mean_L_d09 - mean_L_d01, c='indigo', marker='d')
        ax5.set_yscale('log')
        ax5.set_xscale('log')

        ax6.plot(z, grad_profiles[N,:], c=color)
        ax6.axvline(visc_bl_upper, c=color, alpha=0.8)
        ax6.axvline(visc_bl_upper1, c=color, alpha=0.8)
        print(visc_bl_upper1, visc_bl_upper)
ax1.legend()
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_xlabel(r'$\mathcal{R}$')
ax2.set_xlabel(r'$\mathcal{R}$')
#ax1.set_ylabel(r'$L_{\rm{CZ}}$')
ax1.set_ylabel(r'$\delta_p$')
ax2.set_ylabel(r'$f$')
#ax2.set_ylabel(r'theoretical $\delta_p = L_{\rm{CZ}} \mathcal{P} \frac{1-f}{1 + f\mathcal{P}/2}$  with   $f = 0.6$')
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')


res = np.logspace(np.log10(50),4)
ax2.plot(res, 0.70*(Ls - 3*res**(-2/3)), c='orange', label=r'$\mathcal{R}^{-1/2}$')

fig.savefig('turbulence_check.png', dpi=300, bbox_inches='tight')

ax3.legend()
ax3.axhline(0, c='k', lw=0.5)
ax3.set_xlabel('z')
ax3.set_ylim(-1, 1.05)
ax3.set_xlim(0, 0.5)
#ax3.set_xlim(Ls, 2)
ax3.set_ylabel(r'$\omega^2/\mathcal{R}$')
fig2.savefig('dissipation_profiles.png', dpi=300, bbox_inches='tight')

ax4.legend()
ax4.set_xlabel('z')
#ax4.set_ylim(0, 0.5)
#ax4.set_xlim(0, 0.25)
#ax4.set_ylabel(r'$|u|^2$')
fig3.savefig('vel_profiles.png', dpi=300, bbox_inches='tight')


res = np.logspace(np.log10(50),4)
ax5.plot(res, 3*res**(-2/3), c='orange', label=r'$4\mathcal{R}^{-2/3}$')
ax5.set_ylabel('viscous bl depth')
ax5.set_xlabel('$\mathcal{R}$')
ax5.legend()
fig4.savefig('viscous_bl_depth.png', dpi=300, bbox_inches='tight')

ax6.set_xlim(1.4, 1.8)
ax6.set_xlabel('z')
ax6.set_ylabel(r'$\nabla$')
fig5.savefig('grad_profiles.png', dpi=300, bbox_inches='tight')

ax7.set_xlabel(r'$\mathcal{R}$')
ax7.set_ylabel(r'$u^2\,\,\rm{(CZ)}$')
ax7.set_xscale('log')
fig6.savefig('mid_cz_velocity.png', dpi=300, bbox_inches='tight')
