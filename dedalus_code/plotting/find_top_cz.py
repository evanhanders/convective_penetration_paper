"""
Script for plotting movies of 1D profile movies showing the top of the CZ vs time.

Usage:
    find_top_cz.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: top_cz]
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

plotter = SFP(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')
MPI.COMM_WORLD.barrier()
plotterOne = SFP(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='single', comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(1,1,1)
axs = [ax1,]

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

write_nums = []
times = []
cross_z = []
vel_cross_z = []


bases_names = ['z',]
fields = ['T', 'T_z', 'bruntN2', 'bruntN2_structure', 'F_rad', 'F_conv', 'T_rad_z', 'T_rad_z_IH', 'advection', 'vel_rms', 'effective_heating', 'T1', 'T1_z', 'heat_fluc_rad', 'heat_fluc_conv', 'T1_fluc', 'enstrophy']
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

        Tz_rad0 = first_tasks['T_rad_z'][0,:].squeeze()
        Tz_rad_IH0 = first_tasks['T_rad_z_IH'][0,:].squeeze()
        grad_ad = first_tasks['T_ad_z'][0,:].squeeze()
        system_flux = first_tasks['flux_of_z'][0,:].squeeze()[-1] #get flux at top of domain
        system_flux_prof = first_tasks['flux_of_z'][0,:].squeeze() / system_flux
        fluxrange = np.abs(first_tasks['F_rad'][0,:].squeeze()/system_flux - system_flux_prof).max() * 1.5
        F_cond0 = first_tasks['F_rad'][0,:].squeeze()

        for k in fields:
            tasks[k] = tasks[k].squeeze()
            if len(tasks[k].shape) == 1:
                tasks[k] = np.expand_dims(tasks[k], axis=0)

        for i in range(tasks['T'].shape[0]):
            roller.calculate_rolling_average(i, fields, avg_num=20)
            if plotter.reader.comm.rank == 0:
                print('plotting {:06d}/{}'.format(i+1, tasks['T'].shape[0]))

            F_cond = tasks['F_rad'][i]
            F_conv = tasks['F_conv'][i]

            T = tasks['T'][i]
            T_z = tasks['T_z'][i]
            T1 = T - first_tasks['T'][0]

            brunt_sub_enstrophy = roller.rolled_averages['bruntN2'] - roller.rolled_averages['enstrophy']
            brunt_sub_vel2 = roller.rolled_averages['bruntN2'] - roller.rolled_averages['vel_rms']**2
            cz_sign = np.mean(brunt_sub_enstrophy[(z > 0.1)*(z < 0.9)])
            if np.sum((z > 0.5)*(brunt_sub_enstrophy/cz_sign < 0)) > 0:
                cross_guess = z[(z > 0.5)*(brunt_sub_enstrophy/cz_sign < 0)][0]
                x1 = 1
            else:
                cross_guess = 1
                x1 = 1.1

            func_bse = interp1d(z, brunt_sub_enstrophy, bounds_error=False, fill_value='extrapolate')
            func_bsv2 = interp1d(z, brunt_sub_vel2, bounds_error=False, fill_value='extrapolate')
            root = sop.root_scalar(func_bse, x0=cross_guess, x1=x1).root
            root_vel2 = sop.root_scalar(func_bsv2, x0=cross_guess, x1=x1).root
            logger.info('cross guess and root find: {:.3f}/{:.3f}'.format(cross_guess, root))
            
            times.append(sim_time[i])
            write_nums.append(write_num[i])
            cross_z.append(root)
            vel_cross_z.append(root_vel2)

            ax1.plot(z, roller.rolled_averages['bruntN2'],             c='k', label=r'$N^2$')
            ax1.plot(z, -roller.rolled_averages['bruntN2'],             c='k', ls='--')
            ax1.plot(z, roller.rolled_averages['vel_rms']**2, c='green', label=r'$u^2$')
            ax1.plot(z, roller.rolled_averages['enstrophy'], c='indigo', label=r'$\omega^2$')
            ax1.axvline(root, c='indigo')
            ax1.axvline(root_vel2, c='green')
            ax1.set_yscale('log')
            ax1.legend(loc='upper left')
            ax1.set_ylabel(r'$N^2$')
            ax1.set_ylim(1e-2, np.max(roller.rolled_averages['bruntN2'])*2)

            for ax in axs:
                ax.set_xlabel('z')
                ax.set_xlim(z.min(), z.max())

            plt.suptitle('sim_time = {:.2f}'.format(sim_time[i]))

            fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, start_fig+write_num[i]), dpi=int(args['--dpi']), bbox_inches='tight')
            for ax in axs:
                ax.cla()
        F_conv_avg_first = True
    write_nums = np.array(write_nums)
    times = np.array(times)
    cross_z = np.array(cross_z)
    vel_cross_z = np.array(vel_cross_z)
buffer = np.zeros(1, dtype=int)
if plotter.idle:
    buffer[0] = 0
else:
    buffer[0] = int(write_nums.max())
plotter.reader.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
global_max_write = buffer[0]
if plotter.idle:
    buffer[0] = int(1e6)
else:
    buffer[0] = int(write_nums.min())
plotter.reader.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MIN)
global_min_write = buffer[0]
data = np.zeros((4, int(global_max_write - global_min_write + 1)))
if not plotter.idle:
    write_nums -= int(global_min_write)
    data[0, write_nums] = write_nums
    data[1, write_nums] = times
    data[2, write_nums] = cross_z
    data[3, write_nums] = vel_cross_z
plotter.reader.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
write_nums = data[0,:]
times   = data[1,:]
cross_z = data[2,:]
vel_cross_z = data[3,:]

if plotter.reader.comm.rank == 0:
    fig = plt.figure()
    plt.plot(times, cross_z, c='indigo', label=r'zero of $N^2 - \omega^2$')
    plt.plot(times, vel_cross_z, c='green', label=r'zero of $N^2 - u^2$')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('z (top cz)')
    fig.savefig('{:s}/{:s}.png'.format(plotter.out_dir, 'trace_top_cz'), dpi=400, bbox_inches='tight')

    with h5py.File('{:s}/data_top_cz.h5'.format(plotter.out_dir), 'w') as f:
        f['times'] = times
        f['cross_z'] = cross_z
        f['vel_cross_z'] = vel_cross_z
