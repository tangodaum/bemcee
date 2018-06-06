import numpy as np
import time
import corner_bruno
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from constants import G, Msun, Rsun
from be_theory import oblat2w, t_tms_from_Xc, obl2W
import emcee
import matplotlib as mpl
from corner import corner
from matplotlib import ticker
from matplotlib import *
import matplotlib.font_manager as fm
from utils import geneva_interp_fast, find_nearest, griddataBA,\
    griddataBAtlas, kde_scipy
from be_theory import hfrac2tms
from plot_routines import print_output, par_errors,\
    plot_fit_last, plot_residuals
from reading_routines import read_iue, read_votable,\
    read_star_info
from convergence_routines import plot_convergence

font = fm.FontProperties(size=30)
sfmt = ticker.ScalarFormatter(useMathText=True)
# sfmt.set_powerlimits((0, 0))
sfmt.set_scientific(True)
sfmt.set_powerlimits((-2, 3))

font = fm.FontProperties(size=30)
sfmt = ticker.ScalarFormatter(useMathText=True)
# sfmt.set_powerlimits((0, 0))
sfmt.set_scientific(True)
sfmt.set_powerlimits((-2, 3))
# font = fm.FontProperties(size=16)
font = {'family': 'serif', 'size': 14}

pylab.tick_params(direction='in', size=3, which='both')
plt.tick_params(direction='in', size=3, which='both')
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('font', **font)

folder_data = 'data/'
folder_fig = 'figures/'


# ==============================================================================
def lnlike(params, lbd, logF, dlogF, logF_mod, ranges, include_rv, model):

    """
    Returns the likelihood probability function.

    # p0 = mass
    # p1 = oblat
    # p2 = age
    # p3 = inclination
    # p4 = ebmv
    # p5 = distance
    """

    if model == 'befavor':
        dist = params[4]
        ebmv = params[5]
    if model == 'aara' or model == 'acol' or model == 'bcmi':
        dist = params[7]
        ebmv = params[8]
    if model == 'beatlas':
        dist = params[5]
        ebmv = params[6]

    if include_rv is True:
        RV = params[6]
    else:
        RV = 3.1

    norma = (10 / dist)**2
    uplim = dlogF == 0
    keep = np.logical_not(uplim)

    logF_mod += np.log10(norma)

    tmp_flux = 10**logF_mod

    flux_mod = pyasl.unred(lbd * 1e4, tmp_flux, ebv=-1 * ebmv, R_V=RV)

    logF_mod = np.log10(flux_mod)

    # print(logF_mod)
    chi2 = np.sum(((logF[keep] - logF_mod[keep])**2 / (dlogF[keep])**2.))
    # chi2 = np.sum((logF - logF_mod)**2 / (dlogF)**2.)

    if chi2 is np.nan:
        chi2 = np.inf

    return -0.5 * chi2


# ==============================================================================
def lnprior(params, vsin_obs, sig_vsin_obs, dist_pc, sig_dist_pc,
            ranges, model, stellar_prior, npy_star, pdf_mas,
            pdf_obl, pdf_age, pdf_dis, pdf_ebv, grid_mas,
            grid_obl, grid_age, grid_dis, grid_ebv):

    if model == 'befavor':
        Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
            params[2], params[3], params[4], params[5]
    if model == 'aara' or model == 'acol' or model == 'bcmi':
        Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
            params[2], params[6], params[7], params[8]
    if model == 'beatlas':
        Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
            0.3, params[4], params[5], params[6]

    # Reading Stellar Priors
    if stellar_prior is True:
        temp, idx_mas = find_nearest(grid_mas, value=Mstar)
        temp, idx_obl = find_nearest(grid_obl, value=oblat)
        temp, idx_age = find_nearest(grid_age, value=Hfrac)
        temp, idx_dis = find_nearest(grid_dis, value=dist)
        temp, idx_ebv = find_nearest(grid_ebv, value=ebv)
        chi2_stellar_prior = Mstar * pdf_mas[idx_mas] +\
            oblat * pdf_obl[idx_obl] + \
            Hfrac * pdf_age[idx_age] + \
            dist * pdf_dis[idx_dis] + \
            ebv * pdf_ebv[idx_ebv]
    else:
        chi2_stellar_prior = 0.0

    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
    t = np.max(np.array([hfrac2tms(Hfrac), 0.]))

    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)

    wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)

    vsini = oblat2w(oblat) * wcrit * (Rpole * Rsun * oblat) *\
        np.sin(np.arccos(cosi)) * 1e-5

    chi2_vsi = (vsin_obs - vsini)**2 / sig_vsin_obs**2.

    chi2_dis = (dist_pc - dist)**2 / sig_dist_pc**2.

    chi2_prior = chi2_vsi + chi2_dis + chi2_stellar_prior

    return -0.5 * chi2_prior


# ==============================================================================
def lnprob(params, lbd, logF, dlogF, minfo, listpar, logF_grid,
           vsin_obs, sig_vsin_obs, dist_pc, sig_dist_pc, isig,
           ranges, dims, include_rv, model, stellar_prior, npy_star,
           pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv, grid_mas,
           grid_obl, grid_age, grid_dis, grid_ebv):

    count = 0
    inside_ranges = True
    while inside_ranges * (count < len(params)):
        inside_ranges = (params[count] >= ranges[count, 0]) *\
            (params[count] <= ranges[count, 1])
        count += 1
    if inside_ranges:

        if include_rv is True:
            lim = 3
        else:
            lim = 2
        # print(params[:-lim])
        # print(logF_grid)
        if model == 'beatlas':
            # [9.90066097 1.34002053 0.01568269 3.74540388 0.19852969]

            logF_mod = griddataBAtlas(minfo, logF_grid, params[:-lim],
                                      listpar, dims, isig)
        else:
            logF_mod = griddataBA(minfo, logF_grid, params[:-lim],
                                  listpar, dims)

        lp = lnprior(params, vsin_obs, sig_vsin_obs, dist_pc,
                     sig_dist_pc, ranges, model, stellar_prior,
                     npy_star, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv,
                     grid_mas, grid_obl, grid_age, grid_dis, grid_ebv)

        lk = lnlike(params, lbd, logF, dlogF, logF_mod, ranges,
                    include_rv, model)

        lpost = lp + lk

        # print('{0:.2f} , {1:.2f}, {2:.2f}'.format(lp, lk, lpost))

        if not np.isfinite(lpost):
            return -np.inf
        else:
            return lpost
    else:
        return -np.inf


# ==============================================================================
def run_emcee(p0, sampler, nib, nimc, Ndim, file_name):

    print('\n')
    print(75 * '=')
    print('\n')
    print("Burning-in ...")
    start_time = time.time()
    pos, prob, state = sampler.run_mcmc(p0, nib)

    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    af = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction (BI):", af)

    sampler.reset()

# ------------------------------------------------------------------------------
    print('\n')
    print(75 * '=')
    print("Running MCMC ...")
    pos, prob, state = sampler.run_mcmc(pos, nimc, rstate0=state)

    # Print out the mean acceptance fraction.
    af = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction:", af)

    # median and errors
    flatchain = sampler.flatchain
    par, errors = par_errors(flatchain)

    # best fit parameters
    maxprob_index = np.argmax(prob)
    minprob_index = np.argmax(prob)

    # Get the best parameters and their respective errors
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:, i].std() for i in range(Ndim)]

# ------------------------------------------------------------------------------
    params_fit = []
    errors_fit = []
    pa = []

    for j in range(Ndim):
        for i in range(len(pos)):
            pa.append(pos[i][j])

    pa = np.reshape(pa, (Ndim, len(pos)))

    for j in range(Ndim):
        p = corner_bruno.quantile(pa[j], [0.16, 0.5, 0.84])
        params_fit.append(p[1])
        errors_fit.append((p[0], p[2]))

    params_fit = np.array(params_fit)
    errors_fit = np.array(errors_fit)

# ------------------------------------------------------------------------------
    # Print the output
    print_output(params_fit, errors_fit)
    # Turn it True if you want to see the parameters' sample histograms

    return sampler, params_fit, errors_fit, maxprob_index, minprob_index, af


# ==============================================================================
def emcee_inference(star, Ndim, ranges, lbdarr, wave, logF, dlogF, minfo,
                    listpar, logF_grid, vsin_obs, sig_vsin_obs, dist_pc,
                    sig_dist_pc, isig, dims, include_rv, a_parameter,
                    af_filter, tag, plot_fits, long_process, log_scale,
                    model, acrux, pool, Nproc, stellar_prior, npy_star,
                    pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv,
                    grid_mas, grid_obl, grid_age, grid_dis, grid_ebv):

        if long_process is True:
            Nwalk = 500  # 200  # 500
            nint_burnin = 100  # 50
            nint_mcmc = 1000  # 500  # 1000
        else:
            Nwalk = 20
            nint_burnin = 5
            nint_mcmc = 10

        p0 = [np.random.rand(Ndim) * (ranges[:, 1] - ranges[:, 0]) +
              ranges[:, 0] for i in range(Nwalk)]
        start_time = time.time()

        if acrux is True:
            sampler = emcee.EnsembleSampler(Nwalk, Ndim, lnprob,
                                            args=[wave, logF, dlogF, minfo,
                                                  listpar, logF_grid, vsin_obs,
                                                  sig_vsin_obs, dist_pc,
                                                  sig_dist_pc,
                                                  isig, ranges, dims,
                                                  include_rv, model,
                                                  stellar_prior, npy_star,
                                                  pdf_mas, pdf_obl, pdf_age,
                                                  pdf_dis, pdf_ebv,
                                                  grid_mas, grid_obl,
                                                  grid_age, grid_dis,
                                                  grid_ebv],
                                            a=a_parameter, threads=Nproc,
                                            pool=pool)
        else:
            sampler = emcee.EnsembleSampler(Nwalk, Ndim, lnprob,
                                            args=[wave, logF, dlogF, minfo,
                                                  listpar, logF_grid, vsin_obs,
                                                  sig_vsin_obs, dist_pc,
                                                  sig_dist_pc,
                                                  isig, ranges, dims,
                                                  include_rv, model,
                                                  stellar_prior, npy_star,
                                                  pdf_mas, pdf_obl, pdf_age,
                                                  pdf_dis, pdf_ebv,
                                                  grid_mas, grid_obl,
                                                  grid_age, grid_dis,
                                                  grid_ebv],
                                            a=a_parameter, threads=1)

        sampler_tmp = run_emcee(p0, sampler, nint_burnin, nint_mcmc,
                                Ndim, file_name=star)
        print("--- %s minutes ---" % ((time.time() - start_time) / 60))

        sampler, params_fit, errors_fit, maxprob_index,\
            minprob_index, af = sampler_tmp

        if include_rv is True:
            mass_true, obt_true, xc_true,\
                cos_true, ebv_true, dist_true, rv_true = params_fit
        else:
            if model == 'befavor':
                mass_true, obt_true, xc_true,\
                    cos_true, ebv_true, dist_true = params_fit
            if model == 'aara' or model == 'acol' or model == 'bcmi':
                mass_true, obt_true, xc_true, n0, Rd, n_true,\
                    cos_true, ebv_true, dist_true = params_fit
            if model == 'beatlas':
                mass_true, obt_true, rh0_true, nix_true,\
                    inc_true, dis_true, ebv_true = params_fit
        # if model is False:
        #     angle_in_rad = np.arccos(params_fit[6])
        #     params_fit[6] = (np.arccos(params_fit[6])) * (180. / np.pi)
        #     errors_fit[6] = (errors_fit[6] / (np.sqrt(1. -
        #                      (np.cos(angle_in_rad)**2.)))) * (180. / np.pi)

        chain = sampler.chain

        if af_filter is True:
            acceptance_fractions = sampler.acceptance_fraction
            chain = chain[(acceptance_fractions >= 0.20) &
                          (acceptance_fractions <= 0.50)]
            af = acceptance_fractions[(acceptance_fractions >= 0.20) &
                                      (acceptance_fractions <= 0.50)]
            af = np.mean(af)

        af = str('{0:.2f}'.format(af))

        # Saving first sample
        file_npy = 'figures/' + str(star) + '/' + 'Walkers_' +\
            str(Nwalk) + '_Nmcmc_' + str(nint_mcmc) +\
            '_af_' + str(af) + '_a_' + str(a_parameter) +\
            tag + ".npy"
        np.save(file_npy, chain)

        # plot results
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.rcParams['font.size'] = 16

        # Loading first dataframe
        chain_1 = np.load(file_npy)
        Ndim_1 = np.shape(chain_1)[-1]
        flatchain_1 = chain_1.reshape((-1, Ndim_1))

        if model == 'befavor':
            mas_1 = flatchain_1[:, 0]
            obl_1 = flatchain_1[:, 1]
            age_1 = flatchain_1[:, 2]
            inc_1 = flatchain_1[:, 3]
            dis_1 = flatchain_1[:, 4]
            ebv_1 = flatchain_1[:, 5]
        if model == 'aara' or model == 'acol' or model == 'bcmi':
            mas_1 = flatchain_1[:, 0]
            obl_1 = flatchain_1[:, 1]
            age_1 = flatchain_1[:, 2]
            rh0_1 = flatchain_1[:, 3]
            rdk_1 = flatchain_1[:, 4]
            nix_1 = flatchain_1[:, 5]
            inc_1 = flatchain_1[:, 6]
            dis_1 = flatchain_1[:, 7]
            ebv_1 = flatchain_1[:, 8]
        if model == 'beatlas':
            mas_1 = flatchain_1[:, 0]
            obl_1 = flatchain_1[:, 1]
            rh0_1 = flatchain_1[:, 2]
            nix_1 = flatchain_1[:, 3]
            inc_1 = flatchain_1[:, 4]
            dis_1 = flatchain_1[:, 5]
            ebv_1 = flatchain_1[:, 6]

        if include_rv is True:
            rv_1 = flatchain_1[:, 6]
            par_list = np.zeros([len(mas_1), 7])
        else:
            if model == 'befavor':
                par_list = np.zeros([len(mas_1), 6])
            if model == 'aara' or model == 'acol' or model == 'bcmi':
                par_list = np.zeros([len(mas_1), 9])
            if model == 'beatlas':
                par_list = np.zeros([len(mas_1), 7])

        for i in range(len(mas_1)):
            if include_rv is True:
                par_list[i] = [mas_1[i], obl_1[i], age_1[i],
                               inc_1[i], dis_1[i], ebv_1[i],
                               rv_1[i]]
            else:
                if model == 'befavor':
                    par_list[i] = [mas_1[i], obl_1[i], age_1[i],
                                   inc_1[i], dis_1[i], ebv_1[i]]
                if model == 'aara' or model == 'acol' or model == 'bcmi':
                    par_list[i] = [mas_1[i], obl_1[i], age_1[i],
                                   rh0_1[i], rdk_1[i], nix_1[i],
                                   inc_1[i], dis_1[i], ebv_1[i]]
                if model == 'beatlas':
                    par_list[i] = [mas_1[i], obl_1[i], rh0_1[i],
                                   nix_1[i], inc_1[i], dis_1[i],
                                   ebv_1[i]]

        # plot corner
        if include_rv is True:
            samples = np.vstack((mas_1, obl_1, age_1,
                                inc_1, dis_1, ebv_1, rv_1)).T
        else:
            if model == 'befavor':
                samples = np.vstack((mas_1, obl_1, age_1,
                                    inc_1, dis_1, ebv_1)).T
            if model == 'aara' or model == 'acol' or model == 'bcmi':
                samples = np.vstack((mas_1, obl_1, age_1,
                                     rh0_1, rdk_1, nix_1,
                                    inc_1, dis_1, ebv_1)).T
            if model == 'beatlas':
                samples = np.vstack((mas_1, obl_1, rh0_1,
                                     nix_1, inc_1, dis_1, ebv_1)).T

        k, arr_t_tc, arr_Xc = t_tms_from_Xc(mass_true,
                                            savefig=False,
                                            plot_fig=False)
        ttms_ip = np.arange(0.001, 1., 0.001)
        Xc_ip = np.interp(ttms_ip, arr_t_tc[k], arr_Xc[k])

        for i in range(len(samples)):
            if model == 'befavor' or model == 'aara' or\
               model == 'acol' or model == 'bcmi':
                # Calculating logg for the non_rotating case
                Mstar, oblat, Hfrac = samples[i][0], samples[i][1],\
                    samples[i][2]
            else:
                Mstar, oblat, Hfrac = samples[i][0], samples[i][1], 0.3

            t = np.max(np.array([hfrac2tms(Hfrac), 0.]))

            Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                             neighbours_only=True,
                                             isRpole=False)

            # Converting oblat to W
            samples[i][1] = obl2W(samples[i][1])

            if model == 'befavor':
                # Converting angles to degrees
                samples[i][3] = (np.arccos(samples[i][3])) * (180. / np.pi)
                # Converting Xc to t(tms)
                samples[i][2] = ttms_ip[find_nearest(Xc_ip, samples[i][2])[1]]
            if model == 'aara':
                # Converting Xc to t(tms)
                samples[i][2] = ttms_ip[find_nearest(Xc_ip, samples[i][2])[1]]
                samples[i][5] = samples[i][5] + 1.5
                samples[i][6] = (np.arccos(samples[i][6])) * (180. / np.pi)
            if model == 'acol' or model == 'bcmi':
                # Converting Xc to t(tms)
                samples[i][2] = ttms_ip[find_nearest(Xc_ip, samples[i][2])[1]]
                samples[i][6] = (np.arccos(samples[i][6])) * (180. / np.pi)
            if model == 'beatlas':
                samples[i][4] = (np.arccos(samples[i][4])) * (180. / np.pi)

        # plot corner
        quantiles = [0.16, 0.5, 0.84]
        if include_rv is True:
            labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                      r'$i[\mathrm{^o}]$', r'$d\,[pc]$', r'E(B-V)',
                      r'$R_\mathrm{V}$']
        else:
            if model == 'befavor':
                labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                          r'$i[\mathrm{^o}]$', r'$d\,[pc]$', r'E(B-V)']
            if model == 'aara' or model == 'acol' or model == 'bcmi':
                labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                          r'$\log \, n_0 \, [cm^{-3}]$',
                          r'$R_\mathrm{D}\, [R_\star]$',
                          r'$n$', r'$i[\mathrm{^o}]$', r'$d\,[pc]$', r'E(B-V)']
            if model == 'beatlas':
                labels = [r'$M\,[M_\odot]$', r'$W$', r'$\Sigma_0$', r'$n$',
                          r'$i[\mathrm{^o}]$', r'$d\,[pc]$', r'E(B-V)']
        if model == 'befavor':
            ranges[1] = obl2W(ranges[1])
            ranges[2][0] = ttms_ip[find_nearest(Xc_ip, ranges[2][1])[1]]
            ranges[2][1] = ttms_ip[find_nearest(Xc_ip, ranges[2][0])[1]]
            ranges[3] = (np.arccos(ranges[3])) * (180. / np.pi)
            ranges[3] = np.array([ranges[3][1], ranges[3][0]])
        if model == 'aara':
            ranges[1] = obl2W(ranges[1])
            ranges[2][0] = ttms_ip[find_nearest(Xc_ip, ranges[2][1])[1]]
            ranges[2][1] = ttms_ip[find_nearest(Xc_ip, ranges[2][0])[1]]
            ranges[3] = np.array([ranges[3][1], ranges[3][0]])
            ranges[5] = ranges[5] + 1.5
            ranges[6] = (np.arccos(ranges[6])) * (180. / np.pi)
            ranges[6] = np.array([ranges[6][1], ranges[6][0]])
        if model == 'acol' or model == 'bcmi':
            ranges[1] = obl2W(ranges[1])
            ranges[2][0] = ttms_ip[find_nearest(Xc_ip, ranges[2][1])[1]]
            ranges[2][1] = ttms_ip[find_nearest(Xc_ip, ranges[2][0])[1]]
            ranges[3] = np.array([ranges[3][1], ranges[3][0]])
            ranges[6] = (np.arccos(ranges[6])) * (180. / np.pi)
            ranges[6] = np.array([ranges[6][1], ranges[6][0]])
        if model == 'beatlas':
            ranges[1] = obl2W(ranges[1])
            ranges[3] = np.array([ranges[3][-1], ranges[3][0]])
            ranges[4] = (np.arccos(ranges[4])) * (180. / np.pi)
            ranges[4] = np.array([ranges[4][1], ranges[4][0]])

        corner(samples, labels=labels, range=ranges, quantiles=quantiles,
               plot_contours=True, smooth=2., smooth1d=False,
               plot_datapoints=True, label_kwargs={'fontsize': 22},
               truths=None, show_titles=True, color_hist='black',
               plot_density=True, fill_contours=False,
               no_fill_contours=False, normed=True)

        if plot_fits is True:
            # plot_fit(params_fit, wave, logF, dlogF, minfo,
            #          listpar, lbdarr, logF_grid, isig, dims,
            #          Nwalk, nint_mcmc, par_list, ranges, include_rv,
            #          log_scale)

            plot_fit_last(params_fit, wave, logF, dlogF, minfo,
                          listpar, lbdarr, logF_grid, isig, dims,
                          Nwalk, nint_mcmc, ranges, include_rv,
                          file_npy, log_scale, model)

        current_folder = 'figures/' + str(star) + '/'
        fig_name = 'Walkers_' + np.str(Nwalk) + '_Nmcmc_' +\
            np.str(nint_mcmc) + '_af_' + str(af) + '_a_' +\
            str(a_parameter) + tag
        plt.savefig(current_folder + fig_name + '.png', dpi=100)
        # plt.close()

        # plt.clf()
        plot_residuals(params_fit, wave, logF, dlogF, minfo,
                       listpar, lbdarr, logF_grid, isig, dims,
                       Nwalk, nint_mcmc, ranges, include_rv,
                       file_npy, log_scale, model)
        plt.savefig(current_folder + fig_name + '_residuals' + '.png', dpi=100)

        plot_convergence(file_npy, file_npy[:-4] + '_convergence', model)

        return


# ==============================================================================
def run(input_params):
    stars, list_plx, list_sig_plx, list_vsini_obs,\
        list_sig_vsin_obs, list_pre_ebmv, lbd_range,\
        listpar, Nsigma_dis, include_rv, model, ctrlarr,\
        minfo, models, lbdarr, listpar, dims, isig,\
        a_parameter, af_filter, tag, plot_fits,\
        plot_in_log_scale, long_process, extension,\
        acrux, pool, Nproc, stellar_prior, npy_star = input_params

    if stellar_prior is True:
        chain = np.load('npys/' + npy_star)
        Ndim = np.shape(chain)[-1]
        flatchain = chain.reshape((-1, Ndim))

        mas = flatchain[:, 0]
        obl = flatchain[:, 1]
        age = flatchain[:, 2]
        dis = flatchain[:, -2]
        ebv = flatchain[:, -1]

        # grid_mas = np.linspace(np.min(mas), np.max(mas), 100)
        # grid_obl = np.linspace(np.min(obl), np.max(obl), 100)
        # grid_age = np.linspace(np.min(age), np.max(age), 100)
        # grid_ebv = np.linspace(np.min(ebv), np.max(ebv), 100)

        grid_mas = np.linspace(3.4, 14.6, 100)
        grid_obl = np.linspace(1.00, 1.45, 100)
        grid_age = np.linspace(0.08, 0.78, 100)
        grid_dis = np.linspace(0.00, 140, 100)
        grid_ebv = np.linspace(0.00, 0.10, 100)

        pdf_mas = kde_scipy(x=mas, x_grid=grid_mas, bandwidth=0.005)
        pdf_obl = kde_scipy(x=obl, x_grid=grid_obl, bandwidth=0.005)
        pdf_age = kde_scipy(x=age, x_grid=grid_age, bandwidth=0.01)
        pdf_dis = kde_scipy(x=dis, x_grid=grid_dis, bandwidth=0.01)
        pdf_ebv = kde_scipy(x=ebv, x_grid=grid_ebv, bandwidth=0.0005)

        # plt.plot(grid_mas, pdf_mas)
        # plt.hist(mas, normed=True)
        # plt.show()
        # plt.plot(grid_obl, pdf_obl)
        # plt.hist(obl, normed=True)
        # plt.show()
        # plt.plot(grid_age, pdf_age)
        # plt.hist(age, normed=True)
        # plt.show()
        # plt.plot(grid_dis, pdf_dis)
        # plt.hist(dis, normed=True)
        # plt.show()
        # plt.plot(grid_ebv, pdf_ebv)
        # plt.hist(ebv, normed=True)
        # plt.show()
    else:
        grid_mas = 0
        grid_obl = 0
        grid_age = 0
        grid_dis = 0
        grid_ebv = 0

        pdf_mas = 0
        pdf_obl = 0
        pdf_age = 0
        pdf_dis = 0
        pdf_ebv = 0

    # if model == 'befavor':
    #     cut_iue_regions = False
    # else:
    cut_iue_regions = False

    if np.size(stars) == 1:
        ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
            Ndim, band =\
            read_star_info(stars, list_plx, list_sig_plx,
                           list_vsini_obs, list_sig_vsin_obs,
                           list_pre_ebmv, lbd_range, listpar,
                           Nsigma_dis, include_rv, model)

        # Reading IUE data
        wave0, flux0, sigma0 = read_votable(folder_data, stars)

        logF, dlogF, logF_grid, wave =\
            read_iue(models, lbdarr, wave0, flux0, sigma0,
                     folder_data, folder_fig, stars,
                     cut_iue_regions, model)

        emcee_inference(stars, Ndim, ranges, lbdarr, wave, logF, dlogF,
                        minfo, listpar, logF_grid, vsin_obs, sig_vsin_obs,
                        dist_pc, sig_dist_pc, isig, dims, include_rv,
                        a_parameter, af_filter, tag, plot_fits, long_process,
                        plot_in_log_scale, model, acrux, pool, Nproc,
                        stellar_prior, npy_star, pdf_mas, pdf_obl, pdf_age,
                        pdf_dis, pdf_ebv, grid_mas, grid_obl,
                        grid_age, grid_dis, grid_ebv)
    else:
        for i in range(np.size(stars)):
            star = stars[i]
            star = star.astype('str')
            ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
                Ndim, band =\
                read_star_info(star, list_plx[i], list_sig_plx[i],
                               list_vsini_obs[i], list_sig_vsin_obs[i],
                               list_pre_ebmv[i], lbd_range[i], listpar,
                               Nsigma_dis, include_rv, model)

            wave0, flux0, sigma0 = read_votable(folder_data, star)

            logF, dlogF, logF_grid, wave =\
                read_iue(models, lbdarr, wave0, flux0, sigma0, folder_data,
                         folder_fig, star, cut_iue_regions, model)

            emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
                            dlogF, minfo, listpar, logF_grid, vsin_obs,
                            sig_vsin_obs, dist_pc, sig_dist_pc, isig,
                            dims, include_rv, a_parameter,
                            af_filter, tag, plot_fits, long_process,
                            plot_in_log_scale, model, acrux, pool, Nproc,
                            stellar_prior, npy_star, pdf_mas, pdf_obl, pdf_age,
                            pdf_dis, pdf_ebv, grid_mas, grid_obl,
                            grid_age, grid_dis, grid_ebv)
    return

