from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj')
import h5py

import pandas as pd

col_width = 3.25
golden_ratio = 1.61803398875

S = 1e3
Re = 400
Lz = 2

p1      = ('p1_long', 'temporal_data/erf_re4e2_p1e0_s1e3_longevolution.h5')
p1_down = ('p1_down', 'temporal_data/erf_re4e2_p1e0_s1e3_pertDown.h5')
p1_up   = ('p1_up'  , 'temporal_data/erf_re4e2_p1e0_s1e3_pertUp.h5')
p2      = ('p2_long', 'temporal_data/erf_re4e2_p2e0_s1e3_longevolution.h5')
p2_down = ('p2_down', 'temporal_data/erf_re4e2_p2e0_s1e3_pertDown.h5')
p2_up   = ('p2_up'  , 'temporal_data/erf_re4e2_p2e0_s1e3_pertUp.h5')
p4      = ('p4_long', 'temporal_data/erf_re4e2_p4e0_s1e3_longevolution.h5')
p4_down = ('p4_down', 'temporal_data/erf_re4e2_p4e0_s1e3_pertDown.h5')
p4_up   = ('p4_up'  , 'temporal_data/erf_re4e2_p4e0_s1e3_pertUp.h5')

files = [p1, p1_down, p1_up, p2, p2_down, p2_up, p4, p4_down, p4_up]

fig = plt.figure(figsize=(2*col_width, col_width/golden_ratio))
ax1 = fig.add_axes([0.00,    0.5, 0.33, 0.5])
ax2 = fig.add_axes([0.00,    0.0, 0.33, 0.5])
ax3 = fig.add_axes([0.33,    0.5, 0.33, 0.5])
ax4 = fig.add_axes([0.33,    0.0, 0.33, 0.5])
ax5 = fig.add_axes([0.66,    0.5, 0.33, 0.5])
ax6 = fig.add_axes([0.66,    0.0, 0.33, 0.5])

data = dict()
for tup in files:
    data[tup[0]] = dict()

    with h5py.File(tup[1], 'r') as in_f:
        data[tup[0]]['d05s'] = in_f['scalars/d05'][()] 
        data[tup[0]]['times'] = in_f['scalars/times'][()]
        data[tup[0]]['max_time'] = in_f['scalars/times'][-1]
        data[tup[0]]['times'] /= data[tup[0]]['max_time']
        data[tup[0]]['f'] = in_f['scalars/f'][()]

        f_data = dict()
        f_data['sim_time'] = data[tup[0]]['times']
        f_data['f'] = data[tup[0]]['f']
        f_df = pd.DataFrame(data=f_data)
        rolledf = f_df.rolling(window=200, min_periods=25).mean()
        data[tup[0]]['f'] = rolledf['f']


ax1.axhline(np.mean(data['p1_long']['d05s'][-1000:]), c='k', lw=0.5)
ax3.axhline(np.mean(data['p2_long']['d05s'][-1000:]), c='k', lw=0.5)
ax5.axhline(np.mean(data['p4_long']['d05s'][-1000:]), c='k', lw=0.5)

ax2.axhline(np.mean(data['p1_long']['f'][-1000:]), c='k', lw=0.5)
ax4.axhline(np.mean(data['p2_long']['f'][-1000:]), c='k', lw=0.5)
ax6.axhline(np.mean(data['p4_long']['f'][-1000:]), c='k', lw=0.5)


ax1.axvline(data['p1_down']['max_time']/data['p1_long']['max_time'], c='green', lw=2)
ax3.axvline(data['p2_down']['max_time']/data['p2_long']['max_time'], c='green', lw=2)
ax5.axvline(data['p4_down']['max_time']/data['p4_long']['max_time'], c='green', lw=2)
ax1.axvline(data['p1_up']['max_time']/data['p1_long']['max_time'], c='orange')
ax3.axvline(data['p2_up']['max_time']/data['p2_long']['max_time'], c='orange')
ax5.axvline(data['p4_up']['max_time']/data['p4_long']['max_time'], c='orange')


ax1.plot(data['p1_long']['times'][110:], data['p1_long']['d05s'][110:], c='k', zorder=100)
ax3.plot(data['p2_long']['times'][80:], data['p2_long']['d05s'][80:], c='k', zorder=100)
ax5.plot(data['p4_long']['times'], data['p4_long']['d05s'], c='k', zorder=100)
ax1.plot(data['p1_up']['times'],   data['p1_up']['d05s'], c='orange', zorder=5)
ax3.plot(data['p2_up']['times'],   data['p2_up']['d05s'], c='orange', zorder=5)
ax5.plot(data['p4_up']['times'],   data['p4_up']['d05s'], c='orange', zorder=5)
ax1.plot(data['p1_down']['times'], data['p1_down']['d05s'], c='green', zorder=10)
ax3.plot(data['p2_down']['times'], data['p2_down']['d05s'], c='green', zorder=10)
ax5.plot(data['p4_down']['times'], data['p4_down']['d05s'], c='green', zorder=10)

ax2.plot(data['p1_long']['times'], data['p1_long']['f'], c='k', zorder=100)
ax4.plot(data['p2_long']['times'], data['p2_long']['f'], c='k', zorder=100)
ax6.plot(data['p4_long']['times'], data['p4_long']['f'], c='k', zorder=100)
ax2.plot(data['p1_up']['times'],   data['p1_up'][  'f'], c='orange', zorder=5)
ax4.plot(data['p2_up']['times'],   data['p2_up'][  'f'], c='orange', zorder=5)
ax6.plot(data['p4_up']['times'],   data['p4_up'][  'f'], c='orange', zorder=5)
ax2.plot(data['p1_down']['times'], data['p1_down']['f'], c='green', zorder=10)
ax4.plot(data['p2_down']['times'], data['p2_down']['f'], c='green', zorder=10)
ax6.plot(data['p4_down']['times'], data['p4_down']['f'], c='green', zorder=10)



for ax in [ax2, ax4, ax6]:
    ax.set_ylim(0.6, 0.9)
    ax.set_xlabel(r'$t/t_{\rm{sim}}$')
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlim(0, 1)
for ax in [ax3, ax4, ax6]:
    ax.set_yticks(())
for ax in [ax1, ax3, ax5]:
    ax.set_xticks(())
    ax.set_ylim(0, 0.6)
    ax.set_yticks((0.2, 0.4, 0.6))
for ax in [ax3, ax5]:
    ax.set_yticklabels(())

for ax in [ax4, ax6]:
    ax.set_xticklabels(('', 0.2, 0.4, 0.6, 0.8, 1.0))


ax5.yaxis.set_ticks_position('right')
ax1.set_ylabel(r'$\delta_{\rm{p}}$')
ax2.set_ylabel(r'$f$')

ax1.text(0.50, 0.85, r'$\mathcal{P}_D = 1$', transform=ax1.transAxes, ha='center')
ax3.text(0.50, 0.85, r'$\mathcal{P}_D = 2$', transform=ax3.transAxes, ha='center')
ax5.text(0.50, 0.15, r'$\mathcal{P}_D = 4$', transform=ax5.transAxes, ha='center')

fig.savefig('AE_time_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('../manuscript/AE_time_figure.pdf', dpi=300, bbox_inches='tight')
