"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    compare_top_cz_traces.py <dirs>... [options]

Options:
    --dpi=<dpi>                         Image pixel density [default: 200]

"""
from collections import OrderedDict
import h5py
from mpi4py import MPI
from docopt import docopt
args = docopt(__doc__)
from plotpal.file_reader import SingleFiletypePlotter as SFP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.optimize as sop
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)

dirs = args['<dirs>']

fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

colors = ['k', 'b', 'g', 'r']

for i, d in enumerate(dirs):
    with h5py.File('{:s}/data_top_cz.h5'.format(d), 'r') as f:
        times = f['times'][()]
        L_d01s = f['L_d01s'][()]
        L_d05s = f['L_d05s'][()]
        L_d09s = f['L_d09s'][()]
        f_theory_enstrophy = f['f_theory_enstrophy'][()]

    if 'predictive' in d:
        label = d.split('predictive')[-1].split('/')[0]
    else:
        label = "?"
    ax1.plot(times, L_d01s, c=colors[i], label=label)
    ax1.plot(times, L_d05s, c=colors[i])
    ax1.plot(times, L_d09s, c=colors[i])

    ax2.plot(times, f_theory_enstrophy, c=colors[i])
ax2.set_ylim(0.75, 0.95)
ax1.legend()

for d in dirs:
    plt.savefig('{:s}/comparison_top_cz_trace.png'.format(d), dpi=int(args['--dpi']), bbox_inches='tight')
