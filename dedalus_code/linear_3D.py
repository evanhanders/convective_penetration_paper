"""
Dedalus script for a two-layer, Boussinesq simulation.
The bottom of the domain is at z = 0.
The lower part of the domain is stable; the domain is Schwarzschild stable above z >~ L_cz.

There are 6 control parameters:
    Re      - The approximate reynolds number = (u / diffusivity) of the evolved flows
    Pr      - The Prandtl number = (viscous diffusivity / thermal diffusivity)
    P       - The penetration parameter; When P >> 1 there is lots of convective penetration (of order P); when P -> 0 there is none.
    S       - The stiffness: the characteristic ratio of N^2 (above the penetration region) compared to the square convective frequency.
    zeta    - The fraction of the convective flux carried by the adiabatic gradient at z = 0 (below the heating layer)
    Lz      - The height of the box
    aspect  - The aspect ratio (Lx = aspect * Lz)

Usage:
    linear_3D.py [options] 
    linear_3D.py <config> [options] 

Options:
    --Re=<Reynolds>            Freefall reynolds number [default: 1e2]
    --Pr=<Prandtl>             Prandtl number = nu/kappa [default: 0.5]
    --P=<penetration>          ratio of CZ convective flux / RZ convective flux [default: 1]
    --S=<stiffness>            The stiffness [default: 1e2]
    --zeta=<frac>              Fbot = zeta * F_conv [default: 1e-3]
    --Lz=<L>                   Depth of domain [default: 2]
    --aspect=<aspect>          Aspect ratio of domain [default: 2]
    --L_cz=<L>                 Height of cz-rz erf step [default: 1]

    --nz=<nz>                  Vertical resolution   [default: 256]
    --nx=<nx>                  Horizontal (x) resolution [default: 32]
    --ny=<ny>                  Horizontal (y) resolution (sets to nx by default)
    --RK222                    Use RK222 timestepper (default: RK443)
    --SBDF2                    Use SBDF2 timestepper (default: RK443)
    --safety=<s>               CFL safety factor [default: 0.75]
    --mesh=<m>                 Processor distribution mesh (e.g., "4,4")

    --run_time_wall=<time>     Run time, in hours [default: 119.5]
    --run_time_ff=<time>       Run time, in freefall times [default: 1.6e3]

    --restart=<restart_file>   Restart from checkpoint
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --root_dir=<dir>           Root directory for output [default: ./]

    --adiabatic_IC             If flagged, set the background profile as a pure adiabat (not thermal equilibrium in RZ)
    --predictive=<delta>       A guess for delta_P the penetration depth. The initial state grad(T) will be an erf from grad(T_ad) to grad(T_rad) centered at L_cz + delta_P
    --plot_model               If flagged, create and plt.show() some plots of the 1D atmospheric structure.

    --T_iters=<N>              Number of times to iterate background profile before pure timestepping [default: 100]
    --completion_checks=<N>    If L_cz_change < min_L_cz change this many times, stop AE [default: 5]
    --transient_wait=<t>       Number of sim times to wait for AE procedure after transient starts, an integer [default: 10]
    --N_AE=<t>                 Number of sim times to calculate AE over, an integer. [default: 30]
    --max_L_cz_change=<L>       Maximum delta_p change allowed on AE step [default: 0.05]
    --min_L_cz_change=<L>       Minimum delta_p change allowed on AE step [default: 0.005] 
    --time_step_AE_size=<N>    Size of an AE timestep in simulation units (new L_cz = (timestep size) * |dL_cz/dt|) [default: 500]
"""
import logging
import os
import sys
import time
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from mpi4py import MPI
from scipy.special import erf

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

logger = logging.getLogger(__name__)
args = docopt(__doc__)

#Read config file
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def global_noise(domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    return noise_field

def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def set_equations(problem):
    kx_0  = "(nx == 0) and (ny == 0)"
    kx_n0 = "(nx != 0) or  (ny != 0)"
    equations = ( (True, "True", "T1_z - dz(T1) = 0"),
                  (True, "True", "ωx - dy(w) + dz(v) = 0"),
                  (True, "True", "ωy - dz(u) + dx(w) = 0"),
                  (True, "True", "ωz - dx(v) + dy(u) = 0"),
                  (True, kx_n0,  "dx(u) + dy(v) + dz(w) = 0"), #Incompressibility
                  (True, kx_0,   "p = 0"), #Incompressibility
                  (True, "True", "dt(u) + (dy(ωz) - dz(ωy))/Re0  + dx(p)        = v*ωz - w*ωy "), #momentum-x
                  (True, "True", "dt(v) + (dz(ωx) - dx(ωz))/Re0  + dy(p)        = w*ωx - u*ωz "), #momentum-x
                  (True, kx_n0,  "dt(w) + (dx(ωy) - dy(ωx))/Re0  + dz(p) - T1   = u*ωy - v*ωx "), #momentum-z
                  (True, kx_0,   "w = 0"), #momentum-z
                  (True, kx_n0, "dt(T1) - Lap(T1, T1_z)/Pe0  = -UdotGrad(T1, T1_z) - w*(T0_z - T_ad_z)"), #energy eqn
                  (True, kx_0,  "dt(T1) - dz(k0*T1_z)        = -UdotGrad(T1, T1_z) - w*(T0_z - T_ad_z) + (Q + dz(k0)*T0_z + k0*T0_zz)"), #energy eqn
                )
    for solve, cond, eqn in equations:
        if solve:
            logger.info('solving eqn {} under condition {}'.format(eqn, cond))
            problem.add_equation(eqn, condition=cond)

    boundaries = ( (True, " left(T1_z) = 0", "True"),
                   (True, "right(T1) = 0", "True"),
                   (True, " left(u) = 0", "True"),
                   (True, "right(u) = 0", "True"),
                   (True, " left(v) = 0", "True"),
                   (True, "right(v) = 0", "True"),
                   (True, " left(w) = 0", kx_n0),
                   (True, "right(w) = 0", kx_n0),
                 )
    for solve, bc, cond in boundaries:
        if solve: 
            logger.info('solving bc {} under condition {}'.format(bc, cond))
            problem.add_bc(bc, condition=cond)

    return problem

def set_subs(problem):
    # Set up useful algebra / output substitutions
    problem.substitutions['Lap(A, A_z)']                   = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    problem.substitutions['UdotGrad(A, A_z)']              = '(u*dx(A) + v*dy(A) + w*A_z)'
    problem.substitutions['GradAdotGradB(A, B, A_z, B_z)'] = '(dx(A)*dx(B) + dy(A)*dy(B) + A_z*B_z)'
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz/Ly'
    problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
    problem.substitutions['enstrophy'] = '(ωx**2 + ωy**2 + ωz**2)'
    problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
    problem.substitutions['Re']        = '(Re0*vel_rms)'
    problem.substitutions['Pe']        = '(Pe0*vel_rms)'
    problem.substitutions['T_z']       = '(T0_z + T1_z)'
    problem.substitutions['T']         = '(T0 + T1)'


    problem.substitutions['bruntN2_structure']   = 'T_z - T_ad_z'
    problem.substitutions['bruntN2']             = 'bruntN2_structure'

    #Fluxes
    problem.substitutions['F_rad']       = '-k0*T_z'
    problem.substitutions['T_rad_z']     = '-flux_of_z/k0'
    problem.substitutions['T_rad_z_IH']  = '-right(flux_of_z)/k0'
    problem.substitutions['F_conv']      = 'w*T'
    problem.substitutions['tot_flux']    = '(F_conv + F_rad)'
    return problem

def initialize_output(solver, data_dir, mode='overwrite', output_dt=2, iter=np.inf):
    Lx = solver.problem.parameters['Lx']
    Ly = solver.problem.parameters['Ly']
    analysis_tasks = OrderedDict()
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=40, mode=mode, iter=iter)
    slices.add_task("interp(T1, y={})".format(Ly/2), name="T1_y_mid")
    slices.add_task("interp(T1, x={})".format(Lx/2), name="T1_x_mid")
    slices.add_task("interp(T1, z=0.2)",  name="T1_z_0.2")
    slices.add_task("interp(T1, z=0.5)",  name="T1_z_0.5")
    slices.add_task("interp(T1, z=1)",    name="T1_z_1")
    slices.add_task("interp(T1, z=1.2)",  name="T1_z_1.2")
    slices.add_task("interp(T1, z=1.5)",  name="T1_z_1.5")
    slices.add_task("interp(T1, z=1.8)",  name="T1_z_1.8")
    slices.add_task("interp(w, y={})".format(Ly/2), name="w_y_mid")
    slices.add_task("interp(w, x={})".format(Lx/2), name="w_x_mid")
    slices.add_task("interp(w, z=0.2)",   name="w_z_0.2")
    slices.add_task("interp(w, z=0.5)",   name="w_z_0.5")
    slices.add_task("interp(w, z=1)",     name="w_z_1")
    slices.add_task("interp(w, z=1.2)",   name="w_z_1.2")
    slices.add_task("interp(w, z=1.5)",   name="w_z_1.5")
    slices.add_task("interp(w, z=1.8)",   name="w_z_1.8")
    analysis_tasks['slices'] = slices

    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=40, mode=mode)
    profiles.add_task("plane_avg(T)", name='T')
    profiles.add_task("plane_avg(T_z)", name='T_z')
    profiles.add_task("plane_avg(T1)", name='T1')
    profiles.add_task("plane_avg(sqrt((T1 - plane_avg(T1))**2))", name='T1_fluc')
    profiles.add_task("plane_avg(T1_z)", name='T1_z')
    profiles.add_task("plane_avg(u)", name='u')
    profiles.add_task("plane_avg(w)", name='w')
    profiles.add_task("plane_avg(vel_rms)", name='vel_rms')
    profiles.add_task("plane_avg(sqrt((v*ωz - w*ωy)**2 + (u*ωy - v*ωx)**2))", name='advection')
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(bruntN2)", name="bruntN2")
    profiles.add_task("plane_avg(bruntN2_structure)", name="bruntN2_structure")
    profiles.add_task("plane_avg(flux_of_z)", name="flux_of_z")
    profiles.add_task("plane_avg((Q + dz(k0)*T0_z + k0*T0_zz))", name="effective_heating")
    profiles.add_task("plane_avg(T_rad_z)", name="T_rad_z")
    profiles.add_task("plane_avg(T_rad_z)", name="T_rad_z_IH")
    profiles.add_task("plane_avg(T_ad_z)", name="T_ad_z")
    profiles.add_task("plane_avg(F_rad)", name="F_rad")
    profiles.add_task("plane_avg(F_conv)", name="F_conv")
    profiles.add_task("plane_avg(k0)", name="k0")
    profiles.add_task("plane_avg(dz(k0*T1_z))", name="heat_fluc_rad")
    profiles.add_task("plane_avg(-dz(F_conv))", name="heat_fluc_conv")
    analysis_tasks['profiles'] = profiles

    scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=output_dt*5, max_writes=np.inf, mode=mode)
    scalars.add_task("vol_avg(cz_mask*vel_rms**2)/vol_avg(cz_mask)", name="cz_vel_squared")
    scalars.add_task("vol_avg((1-cz_mask)*bruntN2)/vol_avg(1-cz_mask)", name="rz_brunt_squared")
    analysis_tasks['scalars'] = scalars

    checkpoint_min = 60
    checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoint', wall_dt=checkpoint_min*60, sim_dt=np.inf, iter=np.inf, max_writes=1, mode=mode)
    checkpoint.add_system(solver.state, layout = 'c')
    analysis_tasks['checkpoint'] = checkpoint

    volumes = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=100*output_dt, max_writes=5, mode=mode, iter=iter)
    volumes.add_task("w")
    volumes.add_task("T1")
    volumes.add_task("enstrophy")
    analysis_tasks['volumes'] = volumes

    return analysis_tasks

def run_cartesian_instability(args):
    #############################################################################################
    ### 1. Read in command-line args, set up data directory
    if args['--ny'] is None: args['--ny'] = args['--nx']
    data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]
    data_dir += "_Re{}_P{}_zeta{}_S{}_Lz{}_Lcz{}_Pr{}_a{}_Titer{}_{}x{}x{}".format(args['--Re'], args['--P'], args['--zeta'], args['--S'], args['--Lz'], args['--L_cz'], args['--Pr'], args['--aspect'], args['--T_iters'], args['--nx'], args['--ny'], args['--nz'])
    if args['--predictive'] is not None:
        data_dir += '_predictive{}'.format(args['--predictive'])
    if args['--adiabatic_IC']:
        data_dir += '_adiabaticIC'
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}'.format(data_dir)):
            os.makedirs('{:s}'.format(data_dir))
    logger.info("saving run in: {}".format(data_dir))

    mesh = args['--mesh']
    ncpu = MPI.COMM_WORLD.size
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]
    else:
        log2 = np.log2(ncpu)
        if log2 == int(log2):
            mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
        logger.info("running on processor mesh={}".format(mesh))

    ########################################################################################
    ### 2. Organize simulation parameters
    aspect   = float(args['--aspect'])
    nx = int(args['--nx'])
    ny = int(args['--ny'])
    nz = int(args['--nz'])
    Re0 = float(args['--Re'])
    S  = float(args['--S'])
    Pr = float(args['--Pr'])
    P = float(args['--P'])
    invP = 1/P

    Pe0   = Pr*Re0
    L_cz  = float(args['--L_cz'])
    Lz    = float(args['--Lz'])
    Lx    = aspect * Lz
    Ly    = Lx

    dH = 0.2
    Qmag = 1
    Fconv = dH * Qmag
    zeta = float(args['--zeta'])
    Fbot = zeta*Fconv

    #Model values
    xi = 1 + P * (1 + zeta)
    dz_k0 = dH / (L_cz * S * xi)
    k_bot = dH * zeta / (S * xi)
    k_ad  = dH * (1 + zeta) / (S * xi)
    grad_ad = Qmag * S * xi

    #Adjust to account for expected velocities. and larger m = 0 diffusivities.
    Pe0 /= (np.sqrt(Qmag))
    Re0 /= (np.sqrt(Qmag)) 

    logger.info("Running two-layer instability with the following parameters:")
    logger.info("   Re = {:.3e}, S = {:.3e}, resolution = {}x{}x{}, aspect = {}".format(Re0, S, nx, ny, nz, aspect))
    logger.info("   Pr = {:2g}".format(Pr))
    logger.info("   Re0 = {:.3e}, Pe0 = {:.3e}, Qmag ~ u^2 = {:.3e}".format(Re0, Pe0, Qmag))

    
    ###########################################################################################################3
    ### 3. Setup Dedalus domain, problem, and substitutions/parameters
    x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
    y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)
    bases = [x_basis, y_basis, z_basis]
    domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
    reducer = flow_tools.GlobalArrayReducer(domain.distributor.comm_cart)
    z = domain.grid(-1)
    z_de = domain.grid(-1, scales=domain.dealias)

    #Establish variables and setup problem
    variables = ['T1', 'T1_z', 'p', 'u', 'v', 'w', 'ωx', 'ωy', 'ωz']
    problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

    # Set up background / initial state vs z.
    T0   = domain.new_field()
    T0_z = domain.new_field()
    T0_zz = domain.new_field()
    T_ad_z = domain.new_field()
    T_rad_z0 = domain.new_field()
    k0     = domain.new_field()
    k0_z     = domain.new_field()
    Q = domain.new_field()
    flux_of_z = domain.new_field()
    cz_mask = domain.new_field()
    for f in [T0, T0_z, T_ad_z, k0, k0_z, Q, flux_of_z, T_rad_z0, cz_mask]:
        f.set_scales(domain.dealias)
    for f in [T_ad_z, k0]:
        f.meta['x', 'y']['constant'] = True

    cz_mask['g'] = zero_to_one(z_de, 0.2, width=0.05)*one_to_zero(z_de, L_cz, width=0.05)

    delta = 0.02
    dz_k_func = lambda z: dz_k0 * (1 + (1/P - 1)*zero_to_one(z, L_cz, width=delta))
    Q_func  = lambda z: Qmag*zero_to_one(z, 0.1, delta)*one_to_zero(z, 0.1+dH, delta)
    T_rad_func = lambda flux, k: -flux / k

    k0_z['g'] = dz_k_func(z_de)
    k0_z.antidifferentiate('z', ('left', k_bot), out=k0)
    Q['g'] = Q_func(z_de)
    Q.antidifferentiate('z', ('left', Fbot), out=flux_of_z)
    flux = flux_of_z.interpolate(z=L_cz)['g'].min()

    T_ad_z['g'] = -grad_ad
    T_rad_z0['g'] = T_rad_func(flux_of_z['g'], k0['g'])

    max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad_z['g'])
    logger.info("Max brunt T: {:.3e}".format(max_brunt))

    if args['--adiabatic_IC']:
        T0_zz['g'] = 0
        T0_zz.antidifferentiate('z', ('left', -grad_ad), out=T0_z)
        T0_z.antidifferentiate('z', ('right', 1), out=T0)
    else:
        #Construct T0_zz so that it gets around the discontinuity.
        width = 0.05
        T0_z['g'] = T_rad_z0['g']
        T0_z['g'] += (-grad_ad - T_rad_z0['g'])*one_to_zero(z_de, L_cz, width=width)
        T0_z.antidifferentiate('z', ('right', 1), out=T0)
        T0_z.differentiate('z', out=T0_zz)

    #Check that heating and cooling cancel each other out.
    fH = domain.new_field()
    fH2 = domain.new_field()
    fH.set_scales(domain.dealias)
    fH['g'] = Q['g'] +  k0_z['g']*T0_z['g'] + k0['g']*T0_zz['g']
    fH.antidifferentiate('z', ('left', 0), out=fH2)
    logger.info('right(integ(heating - cooling)): {:.3e}'.format(fH2.interpolate(z=Lz)['g'].max()))

    if args['--plot_model']:
        import matplotlib.pyplot as plt
        plt.plot(z_de.flatten(), -T_ad_z['g'][0,0,:], c='b', lw=0.5, label='-T_ad_z')
        plt.plot(z_de.flatten(), -T_rad_z0['g'][0,0,:], c='r', label='-T_rad_z')
        plt.plot(z_de.flatten(), -T0_z['g'][0,0,:], c='k', label='-T0_z')
        plt.xlabel('z')
        plt.ylabel('-T_z')
        T_rad_top = T_rad_z0.interpolate(z=Lz)['g'].min()
#        plt.ylim(-0.9*T_rad_top, grad_ad - 0.1*(-grad_ad - T_rad_top) )
        plt.legend()
        plt.savefig('{:s}/T0_z_structure.png'.format(data_dir), dpi=400)

        plt.figure()
        plt.plot(z_de.flatten(), k0['g'][0,0,:], c='k', label='k0')
        plt.plot(z_de.flatten(), k0_z['g'][0,0,:], c='b', label='k0_z')
        plt.axhline(k_ad, c='r', lw=0.5)
        plt.xlabel('z')
        plt.ylabel('k0')
        plt.legend()
        plt.savefig('{:s}/k0_structure.png'.format(data_dir), dpi=400)
        plt.show()

    #Plug in default parameters
    problem.parameters['Pe0']    = Pe0
    problem.parameters['Re0']    = Re0
    problem.parameters['Lx']     = Lx
    problem.parameters['Ly']     = Ly
    problem.parameters['Lz']     = Lz
    problem.parameters['k0']     = k0
    problem.parameters['T0']     = T0
    problem.parameters['T0_z']     = T0_z
    problem.parameters['T0_zz']    = T0_zz
    problem.parameters['T_ad_z'] = T_ad_z
    problem.parameters['Q'] = Q
    problem.parameters['flux_of_z'] = flux_of_z
    problem.parameters['cz_mask'] = cz_mask
    problem.parameters['max_brunt'] = max_brunt 

    problem = set_subs(problem)
    problem = set_equations(problem)

    if args['--RK222']:
        logger.info('using timestepper RK222')
        ts = de.timesteppers.RK222
    elif args['--SBDF2']:
        logger.info('using timestepper SBDF2')
        ts = de.timesteppers.SBDF2
    else:
        logger.info('using timestepper RK443')
        ts = de.timesteppers.RK443
    solver = problem.build_solver(ts)
    logger.info('Solver built')

    ###########################################################################
    ### 4. Set initial conditions or read from checkpoint.
    mode = 'overwrite'
    if args['--restart'] is None:
        T1 = solver.state['T1']
        T1_z = solver.state['T1_z']
        z_de = domain.grid(-1, scales=domain.dealias)
        for f in [T1, T1_z]:
            f.set_scales(domain.dealias, keep_data=True)

        if args['--predictive'] is not None:
            z_p = L_cz + float(args['--predictive'])
            logger.info('using predictive 1D ICs with zp: {:.2f}'.format(z_p))

            T_z = -grad_ad + (T_rad_z0['g'] + grad_ad)*zero_to_one(z_de, z_p, width=0.05)
            T1_z['g'] = T_z - T0_z['g']
            T1_z.antidifferentiate('z', ('right', 0), out=T1)


        noise = global_noise(domain, int(args['--seed']))
        T1['g'] += 1e-3*np.sin(np.pi*(z_de))*noise['g']
        T1.differentiate('z', out=T1_z)
        dt = None
    else:
#        write, dt = solver.load_state(args['--restart'], -1) 
        mode = 'append'
        #For some reason some of the state fields are missing from checkpoints (Tz); copy+paste and modify from coer/solvers.py
        import pathlib
        path = pathlib.Path(args['--restart'])
        index = -1
        logger.info("Loading solver state from: {}".format(path))
        with h5py.File(str(path), mode='r') as file:
            # Load solver attributes
            write = file['scales']['write_number'][index]
            try:
                dt = file['scales']['timestep'][index]
            except KeyError:
                dt = None
            solver.iteration = solver.initial_iteration = file['scales']['iteration'][index]
            solver.sim_time = solver.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(solver.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(solver.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            # Load fields
            for field in solver.state.fields:
                if field.name not in file['tasks'].keys():
                    logger.info("can't find {}".format(field))
                    continue
                dset = file['tasks'][field.name]
                # Find matching layout
                for layout in solver.domain.dist.layouts:
                    if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                        break
                else:
                    raise ValueError("No matching layout")
                # Set scales to match saved data
                scales = dset.shape[1:] / layout.global_shape(scales=1)
                scales[~layout.grid_space] = 1
                # Extract local data from global dset
                dset_slices = (index,) + layout.slices(tuple(scales))
                local_dset = dset[dset_slices]
                # Copy to field
                field_slices = tuple(slice(n) for n in local_dset.shape)
                field.set_scales(scales, keep_data=False)
                field[layout][field_slices] = local_dset
                field.set_scales(solver.domain.dealias, keep_data=True)
        solver.state['T1'].differentiate('z', out=solver.state['T1_z'])

    ###########################################################################
    ### 5. Set simulation stop parameters, output, and CFL
    t_ff    = 1/np.sqrt(Qmag)
    t_therm = Pe0
    t_brunt   = np.sqrt(1/max_brunt)
    max_dt    = np.min((0.5*t_ff, t_brunt))
    logger.info('buoyancy and brunt times are: {:.2e} / {:.2e}; max_dt: {:.2e}'.format(t_ff, t_brunt, max_dt))
    if dt is None:
        dt = max_dt

    cfl_safety = float(args['--safety'])
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.25, max_dt=max_dt, threshold=0.2)
    CFL.add_velocities(('u', 'v', 'w'))

    run_time_ff   = float(args['--run_time_ff'])
    run_time_wall = float(args['--run_time_wall'])
    solver.stop_sim_time  = run_time_ff*t_ff
    solver.stop_wall_time = run_time_wall*3600.
 
    ###########################################################################
    ### 6. Setup output tasks; run main loop.
    analysis_tasks = initialize_output(solver, data_dir, mode=mode, output_dt=t_ff)

    T_rad_z0.set_scales(domain.dealias)
    delta_grad_de = grad_ad + T_rad_z0['g'][0,0,:] #= grad_ad - grad_rad; positive in RZ, negative in CZ

    dense_scales = 20
    z_dense = domain.grid(-1, scales=dense_scales)
    T_rad_z0.set_scales((1,1,dense_scales))
    delta_grad = grad_ad + T_rad_z0['g'][0,0,:] #positive in RZ, negative in CZ
    dense_handler = solver.evaluator.add_dictionary_handler(sim_dt=1, iter=np.inf)
    dense_handler.add_task("plane_avg(-T_z)", name='grad', scales=(1,1,dense_scales), layout='g')

    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re", name='Re')
    flow.add_property("Pe", name='Pe')
    flow.properties.add_task("plane_avg(T1_z)", name='mean_T1_z', scales=domain.dealias, layout='g')
    flow.properties.add_task("vol_avg(cz_mask*vel_rms**2/max_brunt)**(-1)", name='stiffness')

    grad_departure_frac = 0.5

    Hermitian_cadence = 100

    def main_loop(dt):
        max_T_iters = int(args['--T_iters'])
        transient_wait = int(args['--transient_wait'])
        N = int(args['--N_AE'])
        max_L_cz_change = float(args['--max_L_cz_change'])
        min_L_cz_change = float(args['--min_L_cz_change'])
        time_step_AE_size = float(args['--time_step_AE_size']) 

        T1 = solver.state['T1']
        T1_z = solver.state['T1_z']
        Re_avg = 0
        done_T_iters = 0

        halfN = int(N/2)
        transient_start = None

        top_cz_times = np.zeros(N)
        delta_p01_vals = np.zeros(N)
        delta_p05_vals = np.zeros(N)
        delta_p09_vals = np.zeros(N)
        good_times = np.zeros(N, dtype=bool)
        last_height_t = 0
        completion_checks = np.zeros(int(args['--completion_checks']), dtype=bool)

        cz_bools = [None, None, None]

        delta_p01 = 0
        delta_p05 = 0
        delta_p09 = 0

        try:
            logger.info('Starting loop')
            start_iter = solver.iteration
            start_time = time.time()
            while solver.ok and np.isfinite(Re_avg):
                effective_iter = solver.iteration - start_iter
                solver.step(dt)

                if effective_iter % Hermitian_cadence == 0:
                    for f in solver.state.fields:
                        f.require_grid_space()

                if solver.sim_time > last_height_t + 1:
                    last_height_t = int(solver.sim_time)
                    #Get departure points from grad_ad
                    grad = dense_handler['grad']['g'][0,0,:]
                    for i, departure_frac in enumerate([0.1, 0.5, 0.9]):
                        cz_bools[i] = (grad > grad_ad - departure_frac*delta_grad)*(delta_grad > 0)
                    delta_p01 = 0
                    delta_p05 = 0
                    delta_p09 = 0
                    if np.sum(cz_bools[0]) > 0:
                        delta_p01 = z_dense.flatten()[cz_bools[0]].max()
                    if np.sum(cz_bools[1]) > 0:
                        delta_p05 = z_dense.flatten()[cz_bools[1]].max()
                    if np.sum(cz_bools[2]) > 0:
                        delta_p09 = z_dense.flatten()[cz_bools[2]].max()
                    delta_p01 = reducer.reduce_scalar(delta_p01, MPI.MAX)
                    delta_p05 = reducer.reduce_scalar(delta_p05, MPI.MAX)
                    delta_p09 = reducer.reduce_scalar(delta_p09, MPI.MAX)

                    #Store trajectory of grad_ad->grad_rad departure points over time
                    if Re_avg > 1:
                        if transient_start is None:
                            transient_start = int(solver.sim_time)
                        if solver.sim_time > transient_start + transient_wait:
                            #Shift old points
                            delta_p01_vals[:-1] = delta_p01_vals[1:]
                            delta_p05_vals[:-1] = delta_p05_vals[1:]
                            delta_p09_vals[:-1] = delta_p09_vals[1:]
                            top_cz_times[:-1] = top_cz_times[1:]
                            good_times[:-1] = good_times[1:]
                            #Store new points
                            delta_p01_vals[-1] = delta_p01
                            delta_p05_vals[-1] = delta_p05
                            delta_p09_vals[-1] = delta_p09
                            top_cz_times[-1] = solver.sim_time
                            good_times[-1] = True

                    #Adjust background thermal profile
                    if done_T_iters < max_T_iters and np.sum(good_times) == N:
                        #Linear fit to 0.1, 0.5, 0.9 penetration depths vs time
                        nice_times = top_cz_times - top_cz_times[0]
                        delta_p01_fit = np.polyfit(nice_times, delta_p01_vals, 1) #fits y = ax + b, returns [a,b]
                        ddelta_p01_dt_linfit = delta_p01_fit[0]
                        delta_p05_fit = np.polyfit(nice_times, delta_p05_vals, 1) #fits y = ax + b, returns [a,b]
                        ddelta_p05_dt_linfit = delta_p05_fit[0]
                        delta_p09_fit = np.polyfit(nice_times, delta_p09_vals, 1) #fits y = ax + b, returns [a,b]
                        ddelta_p09_dt_linfit = delta_p09_fit[0]

                        #Current L_cz = avg of delta_p0.5 over the AE window; 
                        avg_L_cz = np.mean(delta_p05_vals)
                        #transition width = mean distance between 0.9 and 0.5 or 0.5 and 0.1 (whichever is smaller). Adjust to erf units (thus the 0.906).
                        avg_width_up   = np.mean(delta_p09_vals - delta_p05_vals)/0.906 #Erf Heaviside drops to 0.1 or 0.9 at z = z_0 +/- 0.906 * w, for width w
                        avg_width_down = np.mean(delta_p05_vals - delta_p01_vals)/0.906 #Erf Heaviside drops to 0.1 or 0.9 at z = z_0 +/- 0.906 * w, for width w
                        avg_width = np.min((avg_width_down, avg_width_up))

                        #Propagation speed of CZ front is avg propagation speed of delta_p0.1 and delta_p0.5
                        avg_dL_cz_dt = (ddelta_p01_dt_linfit + ddelta_p05_dt_linfit)/2
                        delta_L_cz = time_step_AE_size*avg_dL_cz_dt
                        if np.abs(delta_L_cz) > max_L_cz_change:
                            delta_L_cz *= max_L_cz_change/np.abs(delta_L_cz)

                        logger.info('L_cz: {:.3f}, delta_L_cz: {:.3f}, avg width: {:.3f}'.format(avg_L_cz, delta_L_cz, avg_width))
                        logger.info("ddelta_p01_dt {:.3e}".format(ddelta_p01_dt_linfit))
                        logger.info("ddelta_p05_dt {:.3e}".format(ddelta_p05_dt_linfit))
                        logger.info("ddelta_p09_dt {:.3e}".format(ddelta_p09_dt_linfit))

                        if np.abs(delta_L_cz) < min_L_cz_change:
                            #Too small of a jump; don't adjust, count towards completion
                            completion_checks[int(np.sum(completion_checks))] = True
                            logger.info("{} completion checks done".format(np.sum(completion_checks)))
                            good_times[:halfN] = False
                            if np.sum(completion_checks) == completion_checks.shape[0]:
                                logger.info("{} completion checks done; finishing AE.".format(completion_checks.shape[0]))
                                done_T_iters = max_T_iters
                        else:
                            #AE -- adjust mean temperature profile.
                            L_cz_end   = avg_L_cz + delta_L_cz
                            mean_T_z = -(grad_ad - zero_to_one(z_de, L_cz_end, width=avg_width)*delta_grad_de)
                            mean_T1_z = mean_T_z - T0_z['g'][0,0,:]
                            T1_z['g'] -= flow.properties['mean_T1_z']['g']
                            T1_z['g'] *= one_to_zero(z_de, 1, width=0.05)
                            T1_z['g'] += mean_T1_z
                            T1_z.antidifferentiate('z', ('right', 0), out=T1)

                            for fname in ['u', 'v', 'w', 'ωx', 'ωy', 'ωz', 'p']:
                                solver.state[fname].set_scales(domain.dealias, keep_data=True)
                                solver.state[fname]['g'] *= one_to_zero(z_de, 1, width=0.05)

                            if args['--SBDF2']:
                                solver.timestepper._iteration = 0

                            good_times[:] = False
                            transient_start = None
                            done_T_iters += 1
                            logger.info('T_adjust {}/{}: Adjusting mean state to have L_cz = {:.4f}'.format(done_T_iters, max_T_iters, L_cz_end))


                if effective_iter % 10 == 0:
                    Re_avg = flow.grid_average('Re')

                    log_string =  'Iteration: {:7d}, '.format(solver.iteration)
                    log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time/t_ff, solver.sim_time/Pe0,  dt/t_ff)
                    log_string += 'Pe: {:8.3e}/{:8.3e}, '.format(flow.grid_average('Pe'), flow.max('Pe'))
                    log_string += 'deltap (0.1, 0.5, 0.9): {:.03f}/{:.03f}/{:.03f} '.format(delta_p01, delta_p05, delta_p09)
                    log_string += 'stiffness: {:.01e}'.format(flow.grid_average('stiffness'))
                    logger.info(log_string)

                dt = CFL.compute_dt()
                    
        except:
            raise
            logger.error('Exception raised, triggering end of main loop.')
        finally:
            end_time = time.time()
            main_loop_time = end_time-start_time
            n_iter_loop = solver.iteration-start_iter
            logger.info('Iterations: {:d}'.format(n_iter_loop))
            logger.info('Sim end time: {:f}'.format(solver.sim_time))
            logger.info('Run time: {:f} sec'.format(main_loop_time))
            logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
            logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
            try:
                final_checkpoint = solver.evaluator.add_file_handler(data_dir+'final_checkpoint', wall_dt=np.inf, sim_dt=np.inf, iter=1, max_writes=1)
                final_checkpoint.add_system(solver.state, layout = 'c')
                solver.step(1e-5*dt) #clean this up in the future...works for now.
                post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
            except:
                raise
                print('cannot save final checkpoint')
            finally:
                logger.info('beginning join operation')
                for key, task in analysis_tasks.items():
                    logger.info(task.base_path)
                    post.merge_analysis(task.base_path)
            domain.dist.comm_cart.Barrier()
        return Re_avg

    Re_avg = main_loop(dt)
    if np.isnan(Re_avg):
        return False, data_dir
    else:
        return True, data_dir

if __name__ == "__main__":
    ended_well, data_dir = run_cartesian_instability(args)
    if MPI.COMM_WORLD.rank == 0:
        print('ended with finite Re? : ', ended_well)
        print('data is in ', data_dir)
