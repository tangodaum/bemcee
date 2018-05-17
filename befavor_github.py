# ==============================================================================
# -*- coding:utf-8 -*-
# ==============================================================================
# importing packages
from emcee_routines import emcee_inference
import numpy as np
from reading_routines import read_iue, read_stars, read_befavor_xdr,\
    read_star_info
# import matplotlib
# from sys import argv
# matplotlib.use('Agg')


# ==============================================================================
# General Options
a_parameter = 1.4
extension = '.png'  # Figure extension to be saved
include_rv = False
af_filter = True
long_process = False  # Run with few walkers or many
cut_iue_regions = False
tag = '_rv_false+hip+fullsed'
# list_of_stars = argv[1]  # Define the list of stars here
list_of_stars = '0_test.txt'
plot_fits = True
plot_in_log_scale = True

# ------------------------------------------------------------------------------
folder_fig = 'figures/'
folder_models = 'models/'
folder_tables = 'tables/'
folder_data = 'iue/'  # folder_data = 'data/'
Nsigma_dis = 5.  # ***

# ==============================================================================
# Reading the list of stars
stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsin_obs,\
    list_pre_ebmv, incl0, bump0, lbd_range =\
    read_stars(folder_tables, list_of_stars)

# Reading Models
ctrlarr, minfo, models, lbdarr, listpar,\
    dims, isig = read_befavor_xdr(folder_models)


# ==============================================================================
def main():

    # Define some tabulated values
    if np.size(stars) == 1:
        # Reading Stellar input
        ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
            Ndim, band =\
            read_star_info(stars, list_plx, list_sig_plx,
                           list_vsini_obs, list_sig_vsin_obs,
                           list_pre_ebmv, lbd_range, listpar,
                           Nsigma_dis, include_rv)

        # Reading IUE data
        logF, dlogF, logF_grid, wave =\
            read_iue(models, lbdarr, folder_data,
                     folder_fig, stars, cut_iue_regions)

        emcee_inference(stars, Ndim, ranges, lbdarr, wave, logF, dlogF,
                        minfo, listpar, logF_grid, vsin_obs, sig_vsin_obs,
                        dist_pc, sig_dist_pc, isig, dims, include_rv,
                        a_parameter, af_filter, tag, plot_fits, long_process,
                        plot_in_log_scale)
    else:
        for i in range(np.size(stars)):
            ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
                Ndim, band =\
                read_star_info(stars[i], list_plx[i], list_sig_plx[i],
                               list_vsini_obs[i], list_sig_vsin_obs[i],
                               list_pre_ebmv[i], lbd_range[i], listpar[i],
                               Nsigma_dis, include_rv)

            logF, dlogF, logF_grid, wave =\
                read_iue(models, lbdarr, folder_data,
                         folder_fig, stars[i], cut_iue_regions)

            emcee_inference(stars[i], Ndim, ranges, lbdarr, wave, logF,
                            dlogF, minfo, listpar, logF_grid, vsin_obs,
                            sig_vsin_obs, dist_pc, sig_dist_pc, isig,
                            dims, include_rv, a_parameter,
                            af_filter, tag, plot_fits, long_process,
                            plot_in_log_scale)


# ==============================================================================
if __name__ == '__main__':
    main()

print(75 * '=')
print('\nSimulation Finished\n')
print(75 * '=')

# ==============================================================================

