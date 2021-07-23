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
dirs += glob.glob('stressfree_erf/erf*/')
dirs.sort(key = lambda x: float(x.split('_Re')[-1].split('_')[0]))

nz = 1024
Lz = 2
z_basis = de.Chebyshev('z', nz, interval=[0, Lz], dealias=1)
domain = de.Domain([z_basis], grid_dtype=np.float64)
ke_rhs = domain.new_field()
ke_flux = domain.new_field()
domain_z = domain.grid(-1)

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

fig2 = plt.figure(figsize=(8, 3))

ax_f1 = fig2.add_subplot(1,2,1)
ax_f2 = fig2.add_subplot(1,2,2)

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
    if 'stressfree' in d:
        sf = True
    else:
        sf = False
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

    N = -2
    dissipation_cz = np.sum(enstrophy_profiles[N,z <= Ls]*np.gradient(z)[z <= Ls])/Re
    buoyancy_cz   = np.sum(fconv_profiles[N,z <= Ls]*np.gradient(z)[z <= Ls])
    dissipation_pz = np.sum(enstrophy_profiles[N,z > Ls]*np.gradient(z)[z > Ls])/Re

    color = sm.to_rgba(np.log10(Re))
    if Re >= 100:
        dissipation = enstrophy_profiles[-2,:]/Re
        F_visc = fconv_profiles[-2,:] - dissipation - np.gradient(f_ke_profiles[-2,:], z)


        if sf:
            ls = '--'
        else:
            ls = None
    
        ax1.plot(z, dissipation, c=color, label=Re, ls=ls)
        ax2.plot(z, dissipation, c=color, label=Re, ls=ls)

#        ax1.plot(z, F_visc, c=color, label=Re)
#        ax2.plot(z, F_visc, c=color, label=Re)

        if not sf:
            visc_bl_bottom = z[(np.argmax(F_visc[z < 1]))]
            visc_bl_upper_bot = z[(z > 1.2)*(z < 1.8)][F_visc[(z > 1.2)*(z < 1.8)] < np.min(F_visc[(z > 1.2)*(z < 1.8)])*0.2][0]
            visc_bl_upper_top = z[(z > 1.2)*(z < 1.8)][F_visc[(z > 1.2)*(z < 1.8)] < np.min(F_visc[(z > 1.2)*(z < 1.8)])*0.2][-1]
        else:
            visc_bl_bottom = z[(np.argmin(F_visc[z < 1]))]
            visc_bl_upper_bot = z[z > 1][F_visc[z > 1] < np.min(F_visc[z > 1])*0.25][0]
            visc_bl_upper_top = z[z > 1][F_visc[z > 1] < np.min(F_visc[z > 1])*0.25][-1]
#        ax1.axvline(visc_bl_bottom, c=color)
#        ax2.axvline(visc_bl_upper_bot, c=color)
#        ax2.axvline(visc_bl_upper_top, c=color)

        pz_viscous_bl = visc_bl_upper_top - visc_bl_upper_bot
#        visc_bl = z[F_visc < 0][0]
#        visc_bl_upper1 = z[z > 1][np.argmax(F_visc[z > 1])]
        if sf:
            facecolor=(1,1,1,0.5)
            label='stressfree'
        else:
            facecolor=color
            label='noslip'

        if Re == 800:
            ax3.scatter(Re, visc_bl_bottom, c=facecolor, edgecolor=color, label=label)
        else:
            ax3.scatter(Re, visc_bl_bottom, c=facecolor, edgecolor=color)
        ax4.scatter(Re, pz_viscous_bl, c=facecolor, edgecolor=color)
        ax4.scatter(Re, mean_L_d09 - mean_L_d01, c=facecolor, edgecolor=color, marker='d')
#        ax2.axvline(mean_L_d01, c=color)
#        ax2.axvline(mean_L_d09, c=color)


        #How good is the linear assumption?
        delta_p = visc_bl_upper_top - Ls
#        pz_dissipation = (1 - (z - Ls)/deltap) * avg_dissipation
#        truth_minus_theory = dissipation - pz_dissipation
#        integral = np.sum((truth_minus_theory*np.gradient(z))[(z > Ls)*(z <= visc_bl_upper_top)])
#        print(Re, integral)
#        ax2.plot(z, pz_dissipation, c=color, ls='--')

        f = dissipation_cz/buoyancy_cz
        xi = dissipation_pz/buoyancy_cz / (f * (delta_p/Ls))
        if sf:
            facecolor=(1,1,1,0.5)
            label='stressfree'
        else:
            facecolor=color
            label='noslip'
        if Re == 800:
            ax_f1.scatter(Re, dissipation_cz/buoyancy_cz, c=facecolor, edgecolor=color, label=label)
        else:
            ax_f1.scatter(Re, dissipation_cz/buoyancy_cz, c=facecolor, edgecolor=color)
        ax_f2.scatter(visc_bl_bottom/Ls, f,  c=facecolor, edgecolor=color)
#        ax_f2.scatter(Re, xi, c=facecolor, edgecolor=color)


ax1.set_xlabel('z')
ax2.set_xlabel('z')
#ax1.set_ylabel(r'$L_{\rm{CZ}}$')
ax1.set_ylabel(r'$\Phi$')
ax1.set_ylabel(r'$\Phi$')
#ax2.set_ylabel(r'theoretical $\delta_p = L_{\rm{CZ}} \mathcal{P} \frac{1-f}{1 + f\mathcal{P}/2}$  with   $f = 0.6$')
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax4.yaxis.set_ticks_position('right')
ax4.yaxis.set_label_position('right')

for ax  in [ax1, ax2]:
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_ylim(0, 0.2)



ax1.set_xlim(0, 0.5)
ax2.set_xlim(Ls, Ls + 0.8)

for ax in [ax3, ax4]:
    ax.set_yscale('log')
    ax.set_xscale('log')


res = np.logspace(np.log10(50),4)
ax3.plot(res, 5*res**(-2/3), c='k', label=r'$\mathcal{R}^{-2/3}$')
ax4.plot(res, 3*res**(-2/3), c='k', label=r'$\mathcal{R}^{-2/3}$')
ax3.legend()
ax4.legend()

ax3.set_xlabel(r'$\mathcal{R}$')
ax4.set_xlabel(r'$\mathcal{R}$')
ax3.set_ylabel('bot visc bl depth')
ax4.set_ylabel('top visc bl depth')

fig.savefig('viscous_boundary_layers.png', dpi=300, bbox_inches='tight')


f_inf = 0.735
delta_nu = 5*res**(-2/3)
f_theory = f_inf *(1 - delta_nu/Ls)
ax_f1.plot(res, f_theory, c='orange', label=r'$0.735(1 - 5\mathcal{R}^{-2/3}/L_s)$')
#########ax_f1.plot(res, f_theory, c='orange', label=r'$f_{\infty}(1 - \delta_\nu/L_s)$')
for ax in [ax_f1, ax_f2]:
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mathcal{R}$')
ax_f1.set_ylabel(r'$f$')
ax_f1.legend()
ax_f1.set_ylim(0.6, 0.8)

#ax_f2.axhline(1/2, c='k', lw=0.5, label='linear')
#ax_f2.axhline(2/3, c='r', lw=0.5, label='quadratic')
#ax_f2.legend()
#ax_f2.yaxis.set_ticks_position('right')
#ax_f2.yaxis.set_label_position('right')
#ax_f2.set_ylabel(r'$\xi$')

dnu = np.linspace(0, 1, 100)
ax_f2.plot(dnu, 0.735-dnu/Ls, label=r'0.735 - $\delta_\nu/L_s$', c='orange')
ax_f2.set_ylabel(r'$f$')
ax_f2.set_xlabel(r'$\delta_\nu/L_s$')
ax_f2.set_xscale('linear')
#ax_f2.set_yscale('log')
ax_f2.set_xlim(0, 0.15)
ax_f2.set_ylim(0.6, 0.8)
ax_f2.legend()

fig2.savefig('f_scaling.png', dpi=300, bbox_inches='tight')
