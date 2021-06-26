"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    find_threeD_filling_factor.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: filling_factor]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
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

import dedalus.public as de

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])
start_fig = int(args['--start_fig']) - 1

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']
logger.info("reading data from {}".format(root_dir))

#therm_mach2 = float(root_dir.split("Ma2t")[-1].split("_")[0])
#Lz = float(root_dir.split('Lz')[-1].split('_')[0])
#nz = int(root_dir.split('Titer')[-1].split('x')[-1].split('_')[0])
#z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=1)
#domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
#dense_scales=20
#Tz_field = domain.new_field()
#grad_ad_field = domain.new_field()
#z_dense = domain.grid(0, scales=dense_scales)

plotter = SFP(root_dir, file_dir='volumes', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')
fig = plt.figure(figsize=(8,2))
ax1 = fig.add_subplot(1,1,1)
axs = [ax1,]
#write_nums = []
#times = []
#cross_z = []
#vel_cross_z = []
#z_departure = []


bases_names = ['x', 'y', 'z',]
fields = ['w']
if not plotter.idle:
    while plotter.files_remain(bases_names, fields):
        bases, tasks, write_num, sim_time = plotter.read_next_file()
        horiz_num = len(bases['x'])*len(bases['y'])
        z = bases['z'].flatten()

        for i in range(tasks['w'].shape[0]):
            if plotter.reader.comm.rank == 0:
                print('plotting {:06d}/{}'.format(i+1, tasks['w'].shape[0]))

            w = tasks['w'][i,:]
            up_mask = w > 0
            down_mask = w <= 0

            up_num = np.sum(up_mask, axis=(0,1))
            down_num = np.sum(up_mask, axis=(0,1))

            f_up = up_num / horiz_num
            f_down = down_num / horiz_num

            ax1.plot(z, f_up, c='k', label=r'$f_{\uparrow}$')
            ax1.plot(z, f_up, c='k', label=r'$f_{\uparrow}$')
            ax1.legend(loc='upper left')
            ax1.set_ylabel(r'$f$')

            for ax in axs:
                ax.set_xlabel('z')
                ax.set_xlim(z.min(), z.max())

            plt.suptitle('sim_time = {:.2f}'.format(sim_time[i]))

            fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, start_fig+write_num[i]), dpi=int(args['--dpi']), bbox_inches='tight')
            for ax in axs:
                ax.cla()
#    write_nums = np.array(write_nums)
#    times = np.array(times)
#    cross_z = np.array(cross_z)
#    vel_cross_z = np.array(vel_cross_z)
#    z_departure = np.array(z_departure)
#buffer = np.zeros(1, dtype=int)
#if plotter.idle:
#    buffer[0] = 0
#else:
#    buffer[0] = int(write_nums.max())
#plotter.reader.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
#global_max_write = buffer[0]
#if plotter.idle:
#    buffer[0] = int(1e6)
#else:
#    buffer[0] = int(write_nums.min())
#plotter.reader.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MIN)
#global_min_write = buffer[0]
#data = np.zeros((5, int(global_max_write - global_min_write + 1)))
#if not plotter.idle:
#    write_nums -= int(global_min_write)
#    data[0, write_nums] = write_nums
#    data[1, write_nums] = times
#    data[2, write_nums] = cross_z
#    data[3, write_nums] = vel_cross_z
#    data[4, write_nums] = z_departure
#plotter.reader.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
#write_nums = data[0,:]
#times   = data[1,:]
#cross_z = data[2,:]
#vel_cross_z = data[3,:]
#z_departure = data[4,:]
#
#if plotter.reader.comm.rank == 0:
#    fig = plt.figure()
##    plt.plot(times, cross_z, c='indigo', label=r'zero of $N^2 - \omega^2$')
##    plt.plot(times, vel_cross_z, c='green', label=r'zero of $N^2 - u^2$')
#    plt.plot(times, z_departure - 1, c='red', label=r'50% departure from grad_ad')
#    plt.legend(loc='best')
#    plt.xlabel('time')
#    plt.ylabel(r'$\delta_p$')
#    fig.savefig('{:s}/{:s}.png'.format(plotter.out_dir, 'trace_top_cz'), dpi=400, bbox_inches='tight')
#
#    with h5py.File('{:s}/data_top_cz.h5'.format(plotter.out_dir), 'w') as f:
#        f['times'] = times
#        f['cross_z'] = cross_z
#        f['vel_cross_z'] = vel_cross_z
#        f['z_departure'] = z_departure
