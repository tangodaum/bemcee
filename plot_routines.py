from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from be_theory import hfrac2tms
from utils import geneva_interp_fast, griddataBA


# ==============================================================================
def par_errors(flatchain):
    '''
    Most likely parameters and respective asymmetric errors

    Usage:
    par, errors = par_errors(flatchain)
    '''
    quantile = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                    zip(*np.percentile(flatchain,
                        [16, 50, 84], axis=0))))
    quantile = np.array(quantile)
    par, errors = quantile[:, 0], quantile[:, 1:].reshape((-1))

    return par, errors


# ==============================================================================
def print_output(params_fit, errors_fit):
    """ TBD """
    print(75 * '-')
    print('Output')

    for i in range(len(params_fit)):
        print(params_fit[i], ' +/- ', errors_fit[i])

    return


# ==============================================================================
def plot_fit(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
             isig, dims, Nwalk, Nmcmc, par_list, ranges, include_rv,
             npy, log_scale):
    '''
    Plots best model fit over data

    Usage:
    plot_fit(par, lbd, logF, dlogF, minfo, logF_grid, isig, Nwalk, Nmcmc)
    where
    par = np.array([Mstar, oblat, Sig0, Rd, n, cosi, dist])
    '''
    # model parameters
    if include_rv is True:
        Mstar, oblat, Hfrac, cosi, dist, ebv, rv = par
        lim = 3
        lim2 = 2
    else:
        Mstar, oblat, Hfrac, cosi, dist, ebv = par
        lim = 2
        lim2 = 1
        rv = 3.1

    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
    # t = np.max(np.array([hfrac2tms(hfrac), 0.]))
    t = np.max(np.array([hfrac2tms(Hfrac), 0.]))
    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)
    norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
    uplim = dlogF == 0
    keep = np.logical_not(uplim)
    # chain = np.load(npy)

    # interpolate models
    logF_mod = griddataBA(minfo, logF_grid, par[:-lim], listpar, dims)
    logF_list = np.zeros([len(par_list), len(logF_mod)])
    chi2 = np.zeros(len(logF_list))
    for i in range(len(par_list)):
        logF_list[i] = griddataBA(minfo, logF_grid, par_list[i, :-lim],
                                  listpar, dims)
    # convert to physical units
    logF_mod += np.log10(norma)
    logF_list += np.log10(norma)
    for j in range(len(logF_list)):
        chi2[j] = np.sum((logF[keep] -
                          logF_list[j][keep])**2 / (dlogF[keep])**2)

    # chib = chi2[np.argsort(chi2)[-30:]] / max(chi2)

    flux = 10.**logF
    dflux = dlogF * flux
    flux_mod = 10.**logF_mod

    flux_mod = pyasl.unred(lbd * 1e4, flux_mod, ebv=-1 * ebv, R_V=rv)

    # Plot definitions
    bottom, left = 0.75, 0.48
    width, height = 0.96 - left, 0.97 - bottom
    plt.axes([left, bottom, width, height])

    # plot fit
    if log_scale is True:
        for i in range(len(par_list)):
            if i % 80 == 0:
                ebv_temp = np.copy(par_list[i][-lim2])
                F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
                                     ebv=-1 * ebv_temp, R_V=rv)
                plt.plot(lbd, lbd * F_temp, color='gray', alpha=0.1)

        plt.errorbar(lbd, lbd * flux, yerr=lbd * dflux, ls='', marker='o',
                     alpha=0.5, ms=10, color='blue', linewidth=2)
        plt.plot(lbd, lbd * flux_mod, color='red', ls='-', lw=3.5, alpha=0.4,
                 label='$\mathrm{Best\, fit}$')

        plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
        plt.ylabel(r'$\lambda F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}]}$',
                   fontsize=20)
        plt.yscale('log')
    else:
        for i in range(len(par_list)):
            if i % 80 == 0:
                ebv_temp = np.copy(par_list[i][-lim2])
                F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
                                     ebv=-1 * ebv_temp, R_V=rv)
                plt.plot(lbd, F_temp, color='gray', alpha=0.1)

        plt.errorbar(lbd, flux, yerr=lbd * dflux, ls='', marker='o',
                     alpha=0.5, ms=10, color='blue', linewidth=2)
        plt.plot(lbd, flux_mod, color='red', ls='-', lw=3.5, alpha=0.4,
                 label='$\mathrm{Best\, fit}$')
        plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
        plt.ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2} \mu m]}$',
                   fontsize=20)

    plt.xlim(min(lbd), max(lbd))
    plt.tick_params(direction='in', length=6, width=2, colors='gray',
                    which='both')
    plt.legend(loc='upper right')

    return


# ==============================================================================
def plot_fit_last(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
                  isig, dims, Nwalk, Nmcmc, ranges, include_rv,
                  npy, log_scale, phot):
    '''
    Plots best model fit over data

    Usage:
    plot_fit(par, lbd, logF, dlogF, minfo, logF_grid, isig, Nwalk, Nmcmc)
    where
    par = np.array([Mstar, oblat, Sig0, Rd, n, cosi, dist])
    '''
    # model parameters
    if include_rv is True:
        Mstar, oblat, Hfrac, cosi, dist, ebv, rv = par
        lim = 3
        lim2 = 2
    else:
        if phot is True:
            Mstar, oblat, Hfrac, cosi, dist, ebv = par
        else:
            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv = par
        lim = 2
        lim2 = 1
        rv = 3.1

    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
    # t = np.max(np.array([hfrac2tms(hfrac), 0.]))
    t = np.max(np.array([hfrac2tms(Hfrac), 0.]))
    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)
    norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
    uplim = dlogF == 0
    keep = np.logical_not(uplim)

    # ***
    chain = np.load(npy)
    par_list = chain[:, -1, :]
    # interpolate models
    # print(minfo, logF_grid, par[:-lim], listpar, dims)
    logF_mod = griddataBA(minfo, logF_grid, par[:-lim], listpar, dims)
    logF_list = np.zeros([len(par_list), len(logF_mod)])
    chi2 = np.zeros(len(logF_list))
    for i in range(len(par_list)):
        logF_list[i] = griddataBA(minfo, logF_grid, par_list[i, :-lim],
                                  listpar, dims)
    # convert to physical units
    logF_mod += np.log10(norma)
    logF_list += np.log10(norma)
    for j in range(len(logF_list)):
        chi2[j] = np.sum((logF[keep] -
                          logF_list[j][keep])**2 / (dlogF[keep])**2)

    flux = 10.**logF
    dflux = dlogF * flux
    flux_mod = 10.**logF_mod

    flux_mod = pyasl.unred(lbd * 1e4, flux_mod, ebv=-1 * ebv, R_V=rv)

    # Plot definitions
    bottom, left = 0.80, 0.48  # 0.75, 0.48
    width, height = 0.96 - left, 0.97 - bottom
    plt.axes([left, bottom, width, height])

    # plot fit
    if log_scale is True:
        for i in range(len(par_list)):

            ebv_temp = np.copy(par_list[i][-lim2])
            F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
                                 ebv=-1 * ebv_temp, R_V=rv)
            plt.plot(lbd, lbd * F_temp, color='gray', alpha=0.1)

        plt.errorbar(lbd, lbd * flux, yerr=lbd * dflux, ls='', marker='o',
                     alpha=0.5, ms=10, color='blue', linewidth=2)
        plt.plot(lbd, lbd * flux_mod, color='red', ls='-', lw=3.5, alpha=0.4,
                 label='$\mathrm{Best\, fit}$')

        plt.ylabel(r'$\lambda F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}]}$',
                   fontsize=20)
        plt.yscale('log')
        plt.tick_params(labelbottom='off')
        # plt.setp(upperplot.get_xticklabels(), visible=False)

    else:
        for i in range(len(par_list)):
            ebv_temp = np.copy(par_list[i][-lim2])
            F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
                                 ebv=-1 * ebv_temp, R_V=rv)
            plt.plot(lbd, F_temp, color='gray', alpha=0.1)

        plt.errorbar(lbd, flux, yerr=lbd * dflux, ls='', marker='o',
                     alpha=0.5, ms=10, color='blue', linewidth=2)
        plt.plot(lbd, flux_mod, color='red', ls='-', lw=3.5, alpha=0.4,
                 label='$\mathrm{Best\, fit}$')
        # plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
        plt.ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2} \mu m]}$',
                   fontsize=20)
        plt.tick_params(labelbottom='off')
    plt.xlim(min(lbd), max(lbd))
    if phot is False:
        plt.xscale('log')
    plt.tick_params(direction='in', length=6, width=2, colors='gray',
                    which='both')
    plt.legend(loc='upper right')

    return


# ==============================================================================
def plot_residuals(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
                   isig, dims, Nwalk, Nmcmc, ranges, include_rv,
                   npy, log_scale, phot):
    '''
    Plots best model fit over data

    Usage:
    plot_fit(par, lbd, logF, dlogF, minfo, logF_grid, isig, Nwalk, Nmcmc)
    where
    par = np.array([Mstar, oblat, Sig0, Rd, n, cosi, dist])
    '''
    # model parameters
    if include_rv is True:
        Mstar, oblat, Hfrac, cosi, dist, ebv, rv = par
        lim = 3
        lim2 = 2
    else:
        if phot is True:
            Mstar, oblat, Hfrac, cosi, dist, ebv = par
        else:
            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv = par
        lim = 2
        lim2 = 1
        rv = 3.1

    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
    # t = np.max(np.array([hfrac2tms(hfrac), 0.]))
    t = np.max(np.array([hfrac2tms(Hfrac), 0.]))
    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)
    norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
    uplim = dlogF == 0
    keep = np.logical_not(uplim)

    # ***
    chain = np.load(npy)
    par_list = chain[:, -1, :]
    # interpolate models
    logF_mod = griddataBA(minfo, logF_grid, par[:-lim], listpar, dims)
    logF_list = np.zeros([len(par_list), len(logF_mod)])
    chi2 = np.zeros(len(logF_list))
    for i in range(len(par_list)):
        logF_list[i] = griddataBA(minfo, logF_grid, par_list[i, :-lim],
                                  listpar, dims)
    # convert to physical units
    logF_mod += np.log10(norma)
    logF_list += np.log10(norma)
    for j in range(len(logF_list)):
        chi2[j] = np.sum((logF[keep] -
                          logF_list[j][keep])**2 / (dlogF[keep])**2)

    # chib = chi2[np.argsort(chi2)[-30:]] / max(chi2)

    flux = 10.**logF
    dflux = dlogF * flux
    flux_mod = 10.**logF_mod

    flux_mod = pyasl.unred(lbd * 1e4, flux_mod,
                           ebv=-1 * ebv, R_V=rv)
    # alphas = (1. - chi2 / max(chi2)) / 50.

    # Plot definitions
    bottom, left = 0.71, 0.48
    width, height = 0.96 - left, 0.785 - bottom
    plt.axes([left, bottom, width, height])

    # plot fit
    for i in range(len(par_list)):

        ebv_temp = np.copy(par_list[i][-lim2])
        F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
                             ebv=-1 * ebv_temp, R_V=rv)
        plt.plot(lbd, (flux - F_temp) / dflux, 'bs', alpha=0.2)
    # plt.plot(lbd, (flux - flux_mod) / dflux, 'bs', markersize=5, alpha=0.1)
    # plt.ylabel('$(\mathrm{F}-F_\mathrm{model})/\sigma$', fontsize=20)
    plt.ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=20)
    plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
    plt.ylim(-100, 100)
    plt.hlines(y=0, xmin=min(lbd), xmax=max(lbd),
               linestyles='--', color='black')

    plt.xlim(min(lbd), max(lbd))
    if phot is False:
        plt.xscale('log')
    plt.tick_params(direction='in', length=6, width=2, colors='gray',
                    which='both')
    plt.legend(loc='upper right')

    return


