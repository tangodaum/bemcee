import numpy as np
import math
import pyhdust.beatlas as bat
from operator import is_not
from functools import partial
import os
import pyfits
from utils import bin_data, find_nearest
from scipy.interpolate import griddata
import atpy


# ==============================================================================
def read_stars(stars_table):

    folder_tables = 'tables/'

    typ = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    file_data = folder_tables + stars_table

    a = np.genfromtxt(file_data, usecols=typ, unpack=True,
                      delimiter='\t', comments='#',
                      dtype={'names': ('star', 'plx', 'sig_plx', 'vsini',
                                       'sig_vsini', 'pre_ebmv', 'incl',
                                       'bump', 'lbd_range'),
                             'formats': ('S9', 'f2', 'f2', 'f4',
                                         'f4', 'f4', 'f4', 'S5',
                                         'S24')})

    stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsin_obs,\
        list_pre_ebmv, incl0, bump0, lbd_range =\
        a['star'], a['plx'], a['sig_plx'], a['vsini'], a['sig_vsini'],\
        a['pre_ebmv'], a['incl'], a['bump'], a['lbd_range']

    if np.size(stars) == 1:
        stars = stars.astype('str')
    else:
        for i in range(len(stars)):
            stars[i] = stars[i].astype('str')

    return stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsin_obs,\
        list_pre_ebmv, incl0, bump0, lbd_range


# ==============================================================================
def read_befavor_xdr_complete():

    folder_models = 'models/'

    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL = folder_models + 'aara_sed.xdr'  # 'PL.xdr'
    # xdrPL = folder_models + 'aara_final.xdr'  # 'PL.xdr'
    # xdrPL = folder_models + 'aara_acs.xdr'  # 'PL.xdr'
    # xdrPL = folder_models + 'disk_flx.xdr'  # 'PL.xdr'

    listpar, lbdarr, minfo, models = bat.readBAsed(xdrPL, quiet=False)

    # F(lbd)] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0. or models[i][j] == 0.:
                models[i][j] = (models[i][j + 1] + models[i][j - 1]) / 2.

    # n0 to logn0
    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    minfo[:, 4] = np.log10(minfo[:, 4])

    if True:
        mask = []
        tmp, idx = find_nearest(lbdarr, 1000)
        for i in range(len(models)):
            if models[i][idx] > 2.21834e-10:
                mask.append(i)
                # print(i)
                # plt.plot(lbdarr, models[i], alpha=0.1)
        tmp, idx = find_nearest(lbdarr, 80)
        for i in range(len(models)):
            if models[i][idx] > 2e-8:
                mask.append(i)
                # print(i)
                # # plt.plot(lbdarr, models[i], alpha=0.1)
        tmp, idx = find_nearest(lbdarr, 850)
        for i in range(len(models)):
            if models[i][idx] > 7e-11:
                mask.append(i)
        #         print(i)
        #         plt.plot(lbdarr, models[i], alpha=0.1)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()

        new_models = np.delete(models, mask, axis=0)
        new_minfo = np.delete(minfo, mask, axis=0)

        models = np.copy(new_models)
        minfo = np.copy(new_minfo)

    # delete columns of fixed par
    cols2keep = [0, 1, 3, 4, 5, 7, 8]
    cols2delete = [2, 6]
    listpar = [listpar[i] for i in cols2keep]
    minfo = np.delete(minfo, cols2delete, axis=1)
    listpar[3].sort()

    # for i in range(len(models)):
    #     plt.plot(lbdarr, models[i], alpha=0.1)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_befavor_xdr():

    folder_models = 'models/'

    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, 0.014, np.NaN, 0.0, 50.0, 60.0, 3.5, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL = folder_models + 'BeFaVOr.xdr'
    listpar, lbdarr, minfo, models = bat.readBAsed(xdrPL, quiet=False)
    # [models] = [F(lbd)]] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.

    # delete columns of fixed par
    cols2keep = [0, 1, 3, 8]
    cols2delete = [2, 4, 5, 6, 7]
    listpar = [listpar[i] for i in cols2keep]
    minfo = np.delete(minfo, cols2delete, axis=1)
    listpar[3].sort()
    listpar[3][0] = 0.

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_beatlas_xdr():

    dims = ['M', 'ob', 'sig0', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]
    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    folder_models = 'models/'
    xdrPL = folder_models + 'disk_flx.xdr'  # 'PL.xdr'

    listpar, lbdarr, minfo, models = bat.readBAsed(xdrPL, quiet=False)

    # F(lbd)] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0. or models[i][j] == 0.:
                models[i][j] = (models[i][j + 1] + models[i][j - 1]) / 2.

    listpar[-1][0] = 0.

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_acol_xdr():

    # print(params_tmp)
    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    folder_models = 'models/'
    xdrPL = folder_models + 'acol.xdr'

    listpar, lbdarr, minfo, models = bat.readBAsed(xdrPL, quiet=False)

    # Filter (removing bad models)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(listpar)[0]):
        for j in range(len(listpar[i])):
            if listpar[i][j] < 0:
                listpar[i][j] = 0.

    mask = np.ones(len(minfo[0]), dtype=bool)
    mask[[2, 6]] = False
    result = []
    for i in range(len(minfo)):
        result.append(minfo[i][mask])
    minfo = np.copy(result)

    for i in range(np.shape(minfo)[0]):
        minfo[i][3] = np.log10(minfo[i][3])

    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    listpar = list([listpar[0], listpar[1], listpar[3], listpar[4], listpar[5],
                    listpar[7], listpar[8]])

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_star_info(stars, list_plx, list_sig_plx, list_vsini_obs,
                   list_sig_vsin_obs, list_pre_ebmv, lbd_range,
                   listpar, Nsigma_dis, include_rv, model):

        print(75 * '=')

        star_r = stars.item()
        # star_r = star_r.decode('UTF-8')
        print('\nRunning star: %s\n' % star_r)
        print(75 * '=')

        # star_params = {'parallax': list_plx,
        #                'sigma_parallax': list_sig_plx,
        #                'folder_ines': star_r + '/'}

        plx = np.copy(list_plx)
        dplx = np.copy(list_sig_plx)
        vsin_obs = np.copy(list_vsini_obs)

        band = np.copy(lbd_range)

# ------------------------------------------------------------------------------
        # Reading known stellar parameters
        dist_pc = 1e3 / plx  # pc
        sig_dist_pc = (1e3 * dplx / plx**2)
        sig_vsin_obs = np.copy(list_sig_vsin_obs)

# ------------------------------------------------------------------------------
        # Constrains additional parameters
        if include_rv is True:
            ebmv, rv = [[0.0, 0.1], [2.2, 5.8]]
        else:
            rv = 3.1
            ebmv, rv = [[0.0, 0.1], None]

# ------------------------------------------------------------------------------
        # To add new parameters
        dist_min = dist_pc - Nsigma_dis * sig_dist_pc
        dist_max = dist_pc + Nsigma_dis * sig_dist_pc

        if dist_min < 0:
            dist_min = 1

        addlistpar = [ebmv, [dist_min, dist_max], rv]

        addlistpar = list(filter(partial(is_not, None), addlistpar))

        if model == 'befavor':
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                               [listpar[1][0], listpar[1][-1]],
                               [listpar[2][0], listpar[2][-1]],
                               [listpar[3][0], listpar[3][-1]],
                               [dist_min, dist_max],
                               [ebmv[0], ebmv[-1]]])
        if model == 'aara':
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                               [listpar[1][0], listpar[1][-1]],
                               [listpar[2][0], listpar[2][-1]],
                               [listpar[3][0], listpar[3][-1]],
                               [listpar[4][0], listpar[4][-1]],
                               [listpar[5][0], listpar[5][-1]],
                               [listpar[6][0], listpar[6][-1]],
                               [dist_min, dist_max],
                               [ebmv[0], ebmv[-1]]])
        if model == 'beatlas':
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                               [listpar[1][0], listpar[1][-1]],
                               [listpar[2][0], listpar[2][-1]],
                               [listpar[3][0], listpar[3][-1]],
                               [listpar[4][0], listpar[4][-1]],
                               [dist_min, dist_max],
                               [ebmv[0], ebmv[-1]]])

        if model == 'acol' or model == 'bcmi':
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                               [listpar[1][0], listpar[1][-1]],
                               [listpar[2][0], listpar[2][-1]],
                               [listpar[3][0], listpar[3][-1]],
                               [listpar[4][0], listpar[4][-1]],
                               [listpar[5][0], listpar[5][-1]],
                               [listpar[6][0], listpar[6][-1]],
                               [dist_min, dist_max],
                               [ebmv[0], ebmv[-1]]])

        # print(ranges)
        # if include_rv is True:
        #     ranges = np.array([[listpar[0][0], listpar[0][-1]],
        #                        [listpar[1][0], listpar[1][-1]],
        #                        [listpar[2][0], listpar[2][-1]],
        #                        [listpar[3][0], listpar[3][-1]],
        #                        [dist_min, dist_max],
        #                        [ebmv[0], ebmv[-1]],
        #                        [rv[0], rv[-1]]])
        Ndim = len(ranges)

        return ranges, dist_pc, sig_dist_pc, vsin_obs,\
            sig_vsin_obs, Ndim, band


# ==============================================================================
def read_iue(models, lbdarr, wave0, flux0, sigma0, folder_data,
             folder_fig, star, cut_iue_regions, model):

    table = folder_data + str(star) + '/' + 'list_iue.txt'

    # os.chdir(folder_data + str(star) + '/')
    if os.path.isfile(table) is False or os.path.isfile(table) is True:
        os.system('ls ' + folder_data + str(star) +
                  '/*.FITS | xargs -n1 basename >' +
                  folder_data + str(star) + '/' + 'list_iue.txt')
        iue_list = np.genfromtxt(table, comments='#', dtype='str')
        file_name = np.copy(iue_list)

    fluxes, waves, errors = [], [], []

    for k in range(len(file_name)):
        file_iue = str(folder_data) + str(star) + '/' + str(file_name[k])
        hdulist = pyfits.open(file_iue)
        tbdata = hdulist[1].data
        wave = tbdata.field('WAVELENGTH') * 1e-4  # mum
        flux = tbdata.field('FLUX') * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum
        sigma = tbdata.field('SIGMA') * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum

        # Filter of bad data
        qualy = tbdata.field('QUALITY')
        idx = np.where((qualy == 0))
        wave = wave[idx]
        sigma = sigma[idx]
        flux = flux[idx]

        fluxes = np.concatenate((fluxes, flux), axis=0)
        waves = np.concatenate((waves, wave), axis=0)
        errors = np.concatenate((errors, sigma), axis=0)

    if os.path.isdir(folder_fig + str(star)) is False:
        os.mkdir(folder_fig + str(star))

# ------------------------------------------------------------------------------
    # Would you like to cut the spectrum?
    if cut_iue_regions is True:
        wave_lim_min_iue = 0.135
        wave_lim_max_iue = 0.180

        # Do you want to select a range to middle UV? (2200 bump region)
        wave_lim_min_bump_iue = 0.20  # 0.200 #0.195  #0.210 / 0.185
        wave_lim_max_bump_iue = 0.30  # 0.300 #0.230  #0.300 / 0.335

        indx = np.where(((waves >= wave_lim_min_iue) &
                         (waves <= wave_lim_max_iue)))
        indx2 = np.where(((waves >= wave_lim_min_bump_iue) &
                          (waves <= wave_lim_max_bump_iue)))
        indx3 = np.concatenate((indx, indx2), axis=1)[0]
        waves, fluxes, errors = waves[indx3], fluxes[indx3], errors[indx3]

    else:
        wave_lim_min_iue = min(waves)
        wave_lim_max_iue = 0.300
        indx = np.where(((waves >= wave_lim_min_iue) &
                         (waves <= wave_lim_max_iue)))
        waves, fluxes, errors = waves[indx], fluxes[indx], errors[indx]

    new_wave, new_flux, new_sigma = \
        zip(*sorted(zip(waves, fluxes, errors)))

    nbins = 200
    xbin, ybin, dybin = bin_data(new_wave, new_flux, nbins,
                                 exclude_empty=True)

    ordem = xbin.argsort()
    wave = xbin[ordem]
    flux = ybin[ordem]
    sigma = dybin[ordem]

    if model != 'befavor':
        wave = np.hstack([wave0, wave])
        flux = np.hstack([flux0, flux])
        sigma = np.hstack([sigma0, sigma])

        ordem = wave.argsort()
        wave = wave[ordem]
        flux = flux[ordem]
        sigma = sigma[ordem]

# ------------------------------------------------------------------------------
    # select lbdarr to coincide with lbd
    models_new = np.zeros([len(models), len(wave)])
    if model == 'beatlas' or model == 'aara':
        idx = np.where((wave >= np.min(lbdarr)) & (wave <= np.max(lbdarr)))
        wave = wave[idx]
        flux = flux[idx]
        sigma = sigma[idx]
        models_new = np.zeros([len(models), len(wave)])

    for i in range(len(models)):
        models_new[i, :] = 10.**griddata(np.log(lbdarr),
                                         np.log10(models[i]),
                                         np.log(wave), method='linear')
    # to log space
    logF = np.log10(flux)
    dlogF = sigma / flux
    logF_grid = np.log10(models_new)

    return logF, dlogF, logF_grid, wave


# ==============================================================================
def read_votable(folder_data, star):

    table = folder_data + str(star) + '/' + 'list.txt'

    # os.chdir(folder_data + str(star) + '/')
    if os.path.isfile(table) is False or os.path.isfile(table) is True:
        os.system('ls ' + folder_data + str(star) +
                  '/*.xml | xargs -n1 basename >' +
                  folder_data + str(star) + '/' + 'list.txt')
        vo_list = np.genfromtxt(table, comments='#', dtype='str')
        table_name = np.copy(vo_list)

    vo_file = folder_data + str(star) + '/' + str(table_name)
    try:
        t1 = atpy.Table(vo_file)
        wave = t1['Wavelength'][:]  # Angstrom
        flux = t1['Flux'][:]  # erg/cm2/s/A
        sigma = t1['Error'][:]  # erg/cm2/s/A
    except:
        t1 = atpy.Table(vo_file, tid=1)
        wave = t1['SpectralAxis0'][:]  # Angstrom
        flux = t1['Flux0'][:]  # erg/cm2/s/A
        sigma = [0.] * len(flux)  # erg/cm2/s/A

    new_wave, new_flux, new_sigma = zip(*sorted(zip(wave, flux, sigma)))

    new_wave = list(new_wave)
    new_flux = list(new_flux)
    new_sigma = list(new_sigma)

    # Filtering null sigmas
    for h in range(len(new_sigma)):
        if new_sigma[h] == 0.:
            new_sigma[h] = 0.002 * new_flux[h]

    wave = np.copy(new_wave) * 1e-4
    flux = np.copy(new_flux) * 1e4
    sigma = np.copy(new_sigma) * 1e4

    return wave, flux, sigma


# ==============================================================================
def read_models(model):
    if model == 'befavor':
        ctrlarr, minfo, models, lbdarr, listpar,\
            dims, isig = read_befavor_xdr()
    if model == 'aara':
        ctrlarr, minfo, models, lbdarr, listpar,\
            dims, isig = read_befavor_xdr_complete()
    if model == 'beatlas':
        ctrlarr, minfo, models, lbdarr, listpar,\
            dims, isig = read_beatlas_xdr()
    if model == 'acol' or model == 'bcmi':
        ctrlarr, minfo, models, lbdarr, listpar,\
            dims, isig = read_acol_xdr()

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig
