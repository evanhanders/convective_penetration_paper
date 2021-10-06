import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

col_width = 3.25
page_width = 6.5
golden_ratio = 1.61803398875

brewer_green  = np.array(( 27,158,119, 255))/255
brewer_orange = np.array((217, 95,  2, 255))/255
brewer_purple = np.array((117,112,179, 255))/255

calculate_BLs = False

if calculate_BLs:

    dirs = glob.glob('noslip_erf_Re_cut/erf*/')
    dirs += ['noslip_erf_P_cut/erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256/']
    dirs += glob.glob('stressfree_erf_Re_cut/erf*/')
    dirs.sort(key = lambda x: float(x.split('_Re')[-1].split('_')[0]))

    data = []
    for i, d in enumerate(dirs):
        S = float(d.split('_S')[-1].split('_')[0])
        P = float(d.split('_Pr')[0].split('_P')[-1].split('_')[0])
        Re = float(d.split('_Re')[-1].split('_')[0])
        Lz = float(d.split('_Lz')[-1].split('_')[0])
        if 'stressfree' in d:
            sf = 1
        else:
            sf = 0
        with h5py.File('{:s}/avg_profs/averaged_avg_profs.h5'.format(d), 'r') as f:
            enstrophy_profiles = f['enstrophy'][()]
            fconv_profiles = f['F_conv'][()]
            f_ke_profiles = f['F_KE_p'][()]
            z = f['z'][()]

            dissipation = enstrophy_profiles[-2,:]/Re
            F_visc = fconv_profiles[-2,:] - dissipation - np.gradient(f_ke_profiles[-2,:], z)
            if not sf:
                bot_visc_bl = 2*z[(np.argmax(F_visc[z < 1]))]
            else:
                bot_visc_bl = 2*z[(np.argmin(F_visc[z < 1]))]

        data.append((S, P, Re, sf, bot_visc_bl))

    outf = open('erf_viscous_boundaries.csv', 'w')
    title = '      S,       P,       R, SF,    ell_nu\n'
    outf.write(title)
    for d in data:
        string = '{:.1e}, {:.1e}, {:.1e}, {:2}, {:.3e}\n'.format(*tuple(d))
        outf.write(string)
    outf.close()
bl_data = np.genfromtxt('erf_viscous_boundaries.csv', skip_header=1, delimiter=',')
boundary_R   = bl_data[:,2]
boundary_sf  = bl_data[:,3]
boundary_ell = bl_data[:,4]

        

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
Ls = np.array(data[:,15], dtype=np.float64)

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

bot_bl = np.zeros_like(f)
for i in range(S.shape[0]):
    if S[i] == 1e3 and P[i] == 4:
        bot_bl[i] = boundary_ell[(boundary_R == Re[i])*(boundary_sf == sf[i])]


fig = plt.figure(figsize=(page_width, 2*page_width/(3*golden_ratio)))

ax1 = fig.add_axes([0.00, 0.60, 0.25, 0.40])
ax2 = fig.add_axes([0.35, 0.60, 0.25, 0.40])
ax3 = fig.add_axes([0.70, 0.60, 0.25, 0.40])
ax4 = fig.add_axes([0.00, 0.00, 0.25, 0.40])
ax5 = fig.add_axes([0.35, 0.00, 0.25, 0.40])
ax6 = fig.add_axes([0.70, 0.00, 0.25, 0.40])

good_sf = (erf == 1) * (ae == 1) * (sf == 1) * (S == 1e3) * (P == 4) 
good_ns = (erf == 1) * (ae == 1) * (sf == 0) * (S == 1e3) * (P == 4) 

ax1.scatter(Re[good_sf], d09[good_sf], facecolor=(1,1,1,0.5), color='k', marker='v')
ax1.scatter(Re[good_sf], d05[good_sf], facecolor=(1,1,1,0.5), color='k', marker='o', label=r'$\rm{SF}$')
ax1.scatter(Re[good_sf], d01[good_sf], facecolor=(1,1,1,0.5), color='k', marker='^')
ax1.scatter(Re[good_ns], d09[good_ns], color='k', marker='v', alpha=0.8, s=15)
ax1.scatter(Re[good_ns], d05[good_ns], color='k', marker='o', label=r'$\rm{NS}$', alpha=0.8, s=15)
ax1.scatter(Re[good_ns], d01[good_ns], color='k', marker='^', alpha=0.8, s=15)
ax1.set_xscale('log')
ax1.legend(frameon=True, fontsize=8, framealpha=0.9)
ax1.set_xlabel(r'$\mathcal{R}$')
ax1.set_ylabel(r'$\delta_{0.5}$')
ax1.set_xlim(2e1, 7e3)
ax1.scatter(Re[good_ns*(Re==400)], d05[good_ns*(Re==400)], marker='s',  s=350, edgecolor=brewer_purple, facecolor=(1,1,1,0))


ax2.scatter(Re[good_sf], f[good_sf], facecolor=(1,1,1,0.5), color='k', marker='o')
ax2.scatter(Re[good_ns], f[good_ns], color='k', marker='o', alpha=0.8, s=15)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\mathcal{R}$')
ax2.set_ylabel(r'$f$')
ax2.set_xlim(2e1, 7e3)
ax2.scatter(Re[good_ns*(Re==400)], f[good_ns*(Re==400)], marker='s',  s=150, edgecolor=brewer_purple, facecolor=(1,1,1,0))

ax3.scatter(Re[good_sf], xi[good_sf], facecolor=(1,1,1,0.5), color='k', marker='o')
ax3.scatter(Re[good_ns], xi[good_ns], color='k', marker='o', alpha=0.8, s=15)
ax3.set_xscale('log')
ax3.set_ylabel(r'$\xi$')
ax3.set_xlabel(r'$\mathcal{R}$')
ax3.set_xlim(2e1, 7e3)
ax3.scatter(Re[good_ns*(Re==400)], xi[good_ns*(Re==400)], marker='s',  s=150, edgecolor=brewer_purple, facecolor=(1,1,1,0))

good_sf *= (Re >= 200)
good_ns *= (Re >= 200)

ax4.scatter(f[good_sf], d05[good_sf], facecolor=(1,1,1,0.5), color='k', marker='o')
ax4.scatter(f[good_ns], d05[good_ns], color='k', marker='o', alpha=0.8, s=15)
ax4.set_xlabel(r'$f$')
ax4.set_ylabel(r'$\delta_{0.5}$')
ax4.set_xlim(0.63, 0.75)
ax4.set_ylim(0.3, 0.55)
ax4.scatter(f[good_ns*(Re==400)], d05[good_ns*(Re==400)], marker='s',  s=150, edgecolor=brewer_purple, facecolor=(1,1,1,0))



ax5.scatter(bot_bl[good_sf], f[good_sf], facecolor=(1,1,1,0.5), color='k', marker='o')
ax5.scatter(bot_bl[good_ns], f[good_ns], color='k', marker='o', alpha=0.8, s=15)
#ax5.errorbar(bot_bl[good], theory_f[good], yerr=std_f[good], lw=0, color='k', marker='o', alpha=0.8, markersize=5)
ax5.set_xlim(0, 0.24)
ax5.set_xlabel(r'$\ell_\nu$')
ax5.set_ylabel(r'$f$')
ax5.scatter(bot_bl[good_ns*(Re==400)], f[good_ns*(Re==400)], marker='s',  s=150, edgecolor=brewer_purple, facecolor=(1,1,1,0))

bl = np.linspace(-0.05, 0.4, 100)
f_bl = 0.755*(1 - bl/Ls[good_sf][0])
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
ax6.scatter(Re[good_sf], bot_bl[good_sf], facecolor=(1,1,1,0.5), color='k', marker='o')
ax6.scatter(Re[good_ns], bot_bl[good_ns], color='k', marker='o', alpha=0.8, s=15)
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlabel(r'$\mathcal{R}$')
ax6.set_ylabel(r'$\ell_\nu$')
ax6.legend(frameon=True, fontsize=8, framealpha=0.6)
ax6.set_xlim(1.5e2, 7e3)
ax6.set_ylim(1.4e-2, 6e-1)
ax6.scatter(Re[good_ns*(Re==400)], bot_bl[good_ns*(Re==400)], marker='s',  s=150, edgecolor=brewer_purple, facecolor=(1,1,1,0))

plt.savefig('parameters_vs_re.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/parameters_vs_re.pdf', dpi=300, bbox_inches='tight')
