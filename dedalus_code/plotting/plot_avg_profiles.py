"""
Script for creating averaged 1D profiles of a simulation.

Usage:
    plot_avg_profiles.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: avg_profs]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 20]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 1000]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import ProfilePlotter
import logging
logger = logging.getLogger(__name__)


n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])
avg_writes = int(args['--avg_writes'])

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']

plotter = ProfilePlotter(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files)

fields = ['T', 'T_z', 'bruntN2', 'F_rad', 'F_conv', 'T_rad_z', 'T_rad_z_IH', 'T_ad_z', 'vel_rms', 'effective_heating', 'T1', 'T1_z', 'heat_fluc_rad', 'heat_fluc_conv', 'T1_fluc', 'enstrophy']
for f in fields:
    plotter.add_profile(f, avg_writes)

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
plotter.plot_avg_profiles(dpi=int(args['--dpi']), **plotter_kwargs)
