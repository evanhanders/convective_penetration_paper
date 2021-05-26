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
z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=1)
domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
dense_scales=20
z_dense = domain.grid(0, scales=dense_scales)

plotter = SFP(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')
MPI.COMM_WORLD.barrier()
plotterOne = SFP(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='single', comm=MPI.COMM_SELF)
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
axs = [ax1, ax2]

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

data_cube = []
#Needed data:
#write_nums, times, 
#delta_0.1, delta_0.5, delta_0.9, 
#enstrophy: full domain, below Ls, below delta0.1
#velocity: full domain, below Ls, below delta0.1
#Fconv: full domain, below Ls, below delta0.1
#|grad| above delta0.5, |gradrad| above delta0.5

grad_field = domain.new_field()
grad_integ_field = domain.new_field()
grad_ad_field = domain.new_field()
grad_rad_field = domain.new_field()
grad_rad_integ_field = domain.new_field()
delta_grad_field = domain.new_field()
vel_field = domain.new_field()
vel_integ_field = domain.new_field()
enstrophy_field = domain.new_field()
enstrophy_integ_field = domain.new_field()
fconv_field = domain.new_field()
fconv_integ_field = domain.new_field()

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

        Tz_rad_IH0 = first_tasks['T_rad_z_IH'][0,:].squeeze()
        grad_ad = -first_tasks['T_ad_z'][0,:].squeeze()
        grad_rad = -Tz_rad_IH0

        grad_ad_field.set_scales(1)
        grad_rad_field.set_scales(1)
        delta_grad_field.set_scales(1)
        grad_ad_field['g'] = grad_ad
        delta_grad_field['g'] = grad_ad - grad_rad
        grad_rad_field['g'] = grad_rad
        grad_rad_field.antidifferentiate('z', ('left', 0), out=grad_rad_integ_field)
        grad_ad_field.set_scales(dense_scales, keep_data=True)
        delta_grad_field.set_scales(dense_scales, keep_data=True)

        #Find Ls
        Ls = z_dense[delta_grad_field['g'] < 0][-1]

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

            grad_field.set_scales(1)
            grad_field['g'] = -T_z
            grad_field.antidifferentiate('z', ('left', 0), out=grad_integ_field)
            grad_field.set_scales(dense_scales, keep_data=True)
            vel_field['g'] = roller.rolled_averages['vel_rms']
            vel_field.antidifferentiate('z', ('left', 0), out=vel_integ_field)
            enstrophy_field['g'] = roller.rolled_averages['enstrophy']
            enstrophy_field.antidifferentiate('z', ('left', 0), out=enstrophy_integ_field)
            fconv_field['g'] = roller.rolled_averages['F_conv']
            fconv_field.antidifferentiate('z', ('left', 0), out=fconv_integ_field)

            #departure from grad_ad: 0.1, 0.5, 0.9
            departures = []
            for departure_factor in [0.1, 0.5, 0.9]:
                T_z_departure = (z_dense > 0.9)*(grad_field['g'] > grad_ad_field['g'] - departure_factor*delta_grad_field['g'])*(delta_grad_field['g'] > 0)
                if np.sum(T_z_departure) > 0:
                    z_T_departure = z_dense[T_z_departure].max()
                else:
                    z_T_departure = z.max()
                departures.append(z_T_departure)

            L_d01 = departures[0]
            L_d05 = departures[1]
            L_d09 = departures[2]

            enstrophy_full_domain = enstrophy_integ_field.interpolate(z=Lz)['g'].min() / Lz
            enstrophy_full_cz     = enstrophy_integ_field.interpolate(z=L_d05)['g'].min() / L_d05
            enstrophy_below_Ls    = enstrophy_integ_field.interpolate(z=Ls)['g'].min() / Ls
            vel_full_domain = vel_integ_field.interpolate(z=Lz)['g'].min() / Lz
            vel_full_cz     = vel_integ_field.interpolate(z=L_d05)['g'].min() / L_d05
            vel_below_Ls    = vel_integ_field.interpolate(z=Ls)['g'].min() / Ls
            fconv_full_domain = fconv_integ_field.interpolate(z=Lz)['g'].min() / Lz
            fconv_full_cz     = fconv_integ_field.interpolate(z=L_d05)['g'].min() / L_d05
            fconv_below_Ls    = fconv_integ_field.interpolate(z=Ls)['g'].min() / Ls

            grad_above05 = grad_integ_field.interpolate(z=L_d05)['g'].min() - grad_integ_field.interpolate(z=Lz)['g'].min()
            grad_rad_above05 = grad_rad_integ_field.interpolate(z=L_d05)['g'].min() - grad_rad_integ_field.interpolate(z=Lz)['g'].min()
            grad_above05 /= (Lz - L_d05)
            grad_rad_above05 /= (Lz - L_d05)

            data_list = [sim_time[i], write_num[i], L_d01, L_d05, L_d09]
            data_list += [enstrophy_full_domain, enstrophy_full_cz, enstrophy_below_Ls]
            data_list += [vel_full_domain, vel_full_cz, vel_below_Ls]
            data_list += [fconv_full_domain, fconv_full_cz, fconv_below_Ls]
            data_list += [grad_above05, grad_rad_above05]
            data_cube.append(data_list)

            ax1.plot(z, roller.rolled_averages['bruntN2'],             c='k', label=r'$N^2$')
            ax1.plot(z, -roller.rolled_averages['bruntN2'],             c='k', ls='--')
            ax1.plot(z, roller.rolled_averages['vel_rms']**2, c='green', label=r'$u^2$')
            ax1.plot(z, roller.rolled_averages['enstrophy'], c='indigo', label=r'$\omega^2$')
            ax1.set_yscale('log')
            ax1.legend(loc='upper left')
            ax1.set_ylabel(r'$N^2$')
            ax1.set_ylim(1e-2, np.max(roller.rolled_averages['bruntN2'])*2)

            ax2.plot(z, -roller.rolled_averages['T_z'], label='T_z', c='k')
            ax2.plot(z, grad_rad, label='T_rad_z', c='r')
            ax2.plot(z, grad_ad, lw=0.5, c='b', label='T_ad_z')
            y_min = np.abs(roller.rolled_averages['T_rad_z'][z > 1]).min()
            deltay = np.abs(grad_ad).max() - y_min
            ax2.set_ylim(np.abs(grad_ad).max() - deltay*1.25, np.abs(grad_ad).max() + deltay*1.25)
            ax2.legend(loc='upper right')
            ax2.set_ylabel('-dz(T)')

            for ax in axs:
                ax.set_xlabel('z')
                ax.set_xlim(z.min(), z.max())
                ax.axvline(L_d01, c='red')
                ax.axvline(L_d05, c='k')
                ax.axvline(L_d09, c='red')

            plt.suptitle('sim_time = {:.2f}'.format(sim_time[i]))

            fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter.out_dir, fig_name, start_fig+write_num[i]), dpi=int(args['--dpi']), bbox_inches='tight')
            for ax in axs:
                ax.cla()
        F_conv_avg_first = True
    data_cube = np.array(data_cube)
    write_nums = np.array(data_cube[:,1], dtype=int)
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
if plotter.idle:
    buffer[0] = 0
else:
    buffer[0] = data_cube.shape[1]
plotter.reader.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
num_fields = buffer[0]

global_data = np.zeros((int(global_max_write - global_min_write + 1), num_fields))
if not plotter.idle:
    write_nums -= int(global_min_write)
    global_data[write_nums,:] = data_cube
plotter.reader.comm.Allreduce(MPI.IN_PLACE, global_data, op=MPI.SUM)
times      = global_data[:,0]
write_nums = global_data[:,1]
L_d01s     = global_data[:,2]
L_d05s     = global_data[:,3]
L_d09s     = global_data[:,4]
enstrophy_full = global_data[:,5]
enstrophy_cz   = global_data[:,6]
enstrophy_Ls   = global_data[:,7]
vel_full = global_data[:,8]
vel_cz   = global_data[:,9]
vel_Ls   = global_data[:,10]
fconv_full = global_data[:,11]
fconv_cz   = global_data[:,12]
fconv_Ls   = global_data[:,13]
grad_above = global_data[:,14]
grad_rad_above = global_data[:,15]

if plotter.reader.comm.rank == 0:
    fig = plt.figure()
    plt.plot(times, L_d01s - Ls, c='k', label=r'10% departure from grad_ad')
    plt.plot(times, L_d05s - Ls, c='red', label=r'50% departure from grad_ad')
    plt.plot(times, L_d09s - Ls, c='k', label=r'90% departure from grad_ad')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel(r'$\delta_p$')
    fig.savefig('{:s}/{:s}.png'.format(plotter.out_dir, 'trace_top_cz'), dpi=400, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(times, vel_Ls, c='k')
    plt.xlabel('time')
    plt.ylabel(r'$|u_{\rm{cz}}|$ (below Ls)')
    fig.savefig('{:s}/{:s}.png'.format(plotter.out_dir, 'cz_velocities'), dpi=400, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(times, enstrophy_full/Re_in, c='k', label=r'$\langle \omega^2 \rangle / \mathcal{R}$')
    plt.plot(times, fconv_full,   c='orange', label=r'$\langle F_{\rm{conv}} \rangle$')
    plt.plot(times, enstrophy_cz/Re_in, c='blue', ls=':', label=r'$\langle \omega^2 \rangle / \mathcal{R}$ (cz)')
    plt.plot(times, fconv_cz,   c='red', ls=':', label=r'$\langle F_{\rm{conv}} \rangle$ (cz)')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel(r'KE balances')
    fig.savefig('{:s}/{:s}.png'.format(plotter.out_dir, 'ke_balances'), dpi=400, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(times, grad_above, c='blue', label='grad_above')
    plt.plot(times, grad_rad_above, c='red', label='grad_rad_above')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel(r'Avg grad above')
    fig.savefig('{:s}/{:s}.png'.format(plotter.out_dir, 'avg_grad_above'), dpi=400, bbox_inches='tight')


    with h5py.File('{:s}/data_top_cz.h5'.format(plotter.out_dir), 'w') as f:
        f['times'] = times     
        f['write_nums'] = write_nums
        f['L_d01s'] = L_d01s    
        f['L_d05s'] = L_d05s    
        f['L_d09s'] = L_d09s    
        f['enstrophy_full'] = enstrophy_full
        f['enstrophy_cz'] = enstrophy_cz  
        f['enstrophy_Ls'] = enstrophy_Ls  
        f['vel_full'] = vel_full 
        f['vel_cz'] = vel_cz   
        f['vel_Ls'] = vel_Ls   
        f['fconv_full'] = fconv_full 
        f['fconv_cz'] = fconv_cz   
        f['fconv_Ls'] = fconv_Ls   
        f['grad_above'] = grad_above 
        f['grad_rad_above'] = grad_rad_above 
        f['Ls'] = Ls
        f['Lz'] = Lz
        f['Re_in'] = Re_in

