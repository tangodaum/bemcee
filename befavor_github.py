# ==============================================================================
# -*- coding:utf-8 -*-
# ==============================================================================
# importing packages
from reading_routines import read_stars, read_models
from emcee.utils import MPIPool
from emcee_routines import run
import sys
# import matplotlib
# from sys import argv
# matplotlib.use('Agg')

# ==============================================================================
# General Options
a_parameter = 1.4  # Set internal steps of each walker
extension = '.png'  # Figure extension to be saved
include_rv = False  # If False: fix Rv = 3.1, else Rv will be inferead
af_filter = True  # Remove walkers outside the range 0.2 < af < 0.5
long_process = False  # Run with few walkers or many?
tag = 'hip+af_filter+fullsed+aara_xdr+Total'  # Suffix for the figures
list_of_stars = '1_test.txt'  # 'Total.txt', 'UV.txt', '0_test.txt', argv[1]
plot_fits = True  # Include fits in the corner plot
plot_in_log_scale = True  # yscale in log for fits
Nsigma_dis = 5.  # Set the range of values for the distance
phot = False  # If False: read photospheric grid; If True: read "BeAtlas grid"
acrux = False  # If True, it will run in Nproc processors in the cluster
Nproc = 24  # Number of processors to be used in the cluster

# ==============================================================================
# Acrux
if acrux is True:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
else:
    pool = False

# ==============================================================================
# Reading the list of stars
stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsin_obs,\
    list_pre_ebmv, incl0, bump0, lbd_range =\
    read_stars(list_of_stars)

# Reading Models
ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_models(phot)

# ==============================================================================
# Run code
input_params = stars, list_plx, list_sig_plx, list_vsini_obs,\
    list_sig_vsin_obs, list_pre_ebmv, lbd_range, listpar,\
    Nsigma_dis, include_rv, phot, ctrlarr, minfo, models,\
    lbdarr, listpar, dims, isig, a_parameter, af_filter,\
    tag, plot_fits, plot_in_log_scale, long_process,\
    extension, acrux, pool, Nproc

run(input_params)

# ==============================================================================
# The End
print(75 * '=')
print('\nSimulation Finished\n')
print(75 * '=')

if acrux is True:
    pool.close()

# ==============================================================================

