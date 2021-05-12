"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    compare_top_cz_traces.py <dirs>... [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: top_cz]
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]
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

fig_name   = args['--fig_name']

dirs = args['<dirs>']
data = []

zmax = 0

for d in dirs:
    with h5py.File('{:s}/{:s}/data_top_cz.h5'.format(d, fig_name), 'r') as f:
        times = f['times'][()]
        cross_z = f['cross_z'][()] - 1
        vel_cross_z = f['vel_cross_z'][()] - 1
        z_departure = f['z_departure'][()] - 1
    data.append((times, cross_z, vel_cross_z))

    if 'adiabatic' in d:
        label = 'adiabaticIC'
        color = 'green'
        zorder = 1
        plt.plot(times, z_departure, label=label, color=color, zorder=zorder)
    elif 'predictive' in d:
        label = d.split('predictive')[-1].split('/')[0]
        plt.plot(times, z_departure, label=label)
    else:
        label = 'schwarzschildIC'
        color = 'k'
        zorder = 10
        plt.plot(times, z_departure, label=label, color=color, zorder=zorder)
    if np.max(cross_z) > zmax:
        zmax = np.max(z_departure) 

plt.ylim(0, zmax*1.02)
plt.legend()
plt.xlabel('sim time')
plt.ylabel('top cz z')
for d in dirs:
    plt.savefig('{:s}/{:s}/comparison_top_cz_trace.png'.format(d, fig_name), dpi=int(args['--dpi']), bbox_inches='tight')
