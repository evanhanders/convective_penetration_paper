"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    find_top_cz.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: theory_movie]
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
Re_in = float(root_dir.split('Re')[-1].split('_')[0])
Lz = float(root_dir.split('Lz')[-1].split('_')[0])
nz = int(root_dir.split('Titer')[-1].split('x')[-1].split('_')[0].split('/')[0])

plotter = SFP(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')
MPI.COMM_WORLD.barrier()
plotterOne = SFP(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='single', comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
axs = [ax1, ax2, ax3]

class RollingProfileAverager:
    
    def __init__(self, start_file, n_files):
        self.start_file = start_file
        if n_files is None:
            self.last_file = np.inf
        else:
            self.last_file = start_file + n_files
        self.before_tasks = None
        self.current_tasks = OrderedDict()
        self.rolled_averages = OrderedDict()


    def get_before_after(self, file_name, tasks):
        file_num = int(file_name.split('_s')[-1].split('.h5')[0])
        if file_num > self.start_file:
            before_file_name = file_name.replace('_s{}.h5'.format(file_num), '_s{}.h5'.format(file_num-1))
        else:
            before_file_name = None

        if before_file_name is not None:
            self.before_tasks = OrderedDict()
            with h5py.File(before_file_name, 'r') as f:
                for k in tasks:
                    self.before_tasks[k] = f['tasks'][k][()].squeeze()
        else:
            self.before_tasks = None

        with h5py.File(file_name, 'r') as f:
            for k in tasks:
                self.current_tasks[k] = f['tasks'][k][()].squeeze()
                if len(self.current_tasks[k].shape) == 1:
                    self.current_tasks[k] = np.expand_dims(self.current_tasks[k], axis=0)

    def calculate_rolling_average(self, index, tasks, avg_num=10):
        for k in tasks:
            if index+1 >= avg_num:
                self.rolled_averages[k] = np.mean(self.current_tasks[k][index+1-avg_num:index+1,:], axis=0)
            else:
                if self.before_tasks is None:
                    self.rolled_averages[k] = np.mean(self.current_tasks[k][:index+1,:], axis=0)
                else:
                    roll_arr = np.zeros((avg_num, self.current_tasks[k].shape[1]))
                    roll_arr[:avg_num-(index+1),:] = self.before_tasks[k][(index+1)-avg_num:,:]
                    roll_arr[avg_num-(index+1):,:] = self.current_tasks[k][:(index+1)]
                    self.rolled_averages[k] = np.mean(roll_arr, axis=0)
            
roller = RollingProfileAverager(start_file, n_files)            

bases_names = ['z',]
fields = ['T', 'T_z', 'bruntN2', 'bruntN2_structure', 'F_rad', 'F_conv', 'T_rad_z', 'T_rad_z_IH', 'advection', 'vel_rms', 'effective_heating', 'T1', 'T1_z', 'heat_fluc_rad', 'heat_fluc_conv', 'T1_fluc', 'enstrophy', 'F_KE', 'F_KE_vert', 'F_KE_p']
if not plotter.idle:
    init_fields = ['T_rad_z', 'T_rad_z_IH', 'F_rad', 'T', 'T_ad_z', 'flux_of_z', 'T_z']
    plotterOne.set_read_fields(bases_names, init_fields)
    first = plotterOne.read_next_file()
    first_tasks = first[1]
    for k in init_fields:
        first_tasks[k] = first_tasks[k].squeeze()
    while plotter.files_remain(bases_names, fields):
        roller.get_before_after(plotter.files[plotter.current_filenum], fields)
        bases, tasks, write_num, sim_time = plotter.read_next_file()
        z = bases['z'].squeeze()

        Tz_rad_IH0 = first_tasks['T_rad_z_IH'][0,:].squeeze()
        grad_ad = -first_tasks['T_ad_z'][0,:].squeeze()
        grad_rad = -Tz_rad_IH0

        for k in fields:
            tasks[k] = tasks[k].squeeze()
            if len(tasks[k].shape) == 1:
                tasks[k] = np.expand_dims(tasks[k], axis=0)

        for i in range(tasks['T'].shape[0]):
            roller.calculate_rolling_average(i, fields, avg_num=20)
            if plotter.reader.comm.rank == 0:
                print('plotting {:06d}/{}'.format(i+1, tasks['T'].shape[0]))

            F_conv = roller.rolled_averages['F_conv']
            enstrophy_div_R = roller.rolled_averages['enstrophy']/Re_in
            ke_eqn_rhs = F_conv - enstrophy_div_R

            F_KE = roller.rolled_averages['F_KE'] # w * u^2 / 2
            F_KE_w = roller.rolled_averages['F_KE_vert'] # w^3 / 2
            F_KE_p = roller.rolled_averages['F_KE_p'] # w * p 

            velocity = roller.rolled_averages['vel_rms']
            approx_Re_micro = Re_in * velocity**2 / roller.rolled_averages['enstrophy']**(1/2)

            ax1.plot(z, F_conv, c='red', label=r'$F_{\rm{conv}}$')
            ax1.plot(z, enstrophy_div_R, c='indigo', label=r'$\omega^2/\mathcal{R}$')
            ax1.plot(z, ke_eqn_rhs, c='k', label=r'$F_{\rm{conv}} - \omega^2/\mathcal{R}$')
            ax1.legend(loc='upper right')
            ax1.set_ylabel(r'KE forcings')
            ax1.set_ylim(-0.25, 0.25)

            ax2.plot(z, F_KE, c='k', label=r'$w |u^2| / 2$')
            ax2.plot(z, F_KE_p, c='green',  label=r'$w \varpi$')
            ax2.plot(z, F_KE_p+F_KE, c='red',  label=r'$w \varpi + w |u^2| / 2$')
            ax2.plot(z, F_KE_w * (F_KE_p+F_KE).max()/F_KE_w.max(), c='orange', label=r'$f w^3 / 2$')
            ax2.legend(loc='upper right')
            ax2.set_ylabel('KE fluxes')

            ax3.plot(z, approx_Re_micro, c='k')
            ax3.set_ylabel(r'$\mathcal{R} u^2 / |\omega|$')
            ax3.set_yscale('log')
            ax3.set_ylim(1e-1, 100)
            ax3.axhline(1, c='k')

            for ax in axs:
                ax.set_xlabel('z')
                ax.set_xlim(z.min(), z.max())
                ax.axhline(0, c='k', lw=0.5)

            plt.suptitle('sim_time = {:.2f}'.format(sim_time[i]))

            fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, start_fig+write_num[i]), dpi=int(args['--dpi']), bbox_inches='tight')
            for ax in axs:
                ax.cla()
        F_conv_avg_first = True

