import numpy as np
from scipy.interpolate import griddata
import pyhdust.phc as phc
from scipy.stats import gaussian_kde


# ==============================================================================
def kde_scipy(x, x_grid, bandwidth=0.2):
    """Kernel Density Estimation with Scipy"""

    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    return kde.evaluate(x_grid)


# ==============================================================================
# BIN DATA
def bin_data(x, y, nbins, xran=None, exclude_empty=True):
    '''
    Bins data

    Usage:
    xbin, ybin, dybin = bin_data(x, y, nbins, xran=None, exclude_empty=True)

    where dybin is the standard deviation inside the bins.
    '''
    # make sure it is a numpy array
    x = np.array([x]).reshape((-1))
    y = np.array([y]).reshape((-1))
    # make sure it is in increasing order
    ordem = x.argsort()
    x = x[ordem]
    y = y[ordem]

    if xran is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = xran[0], xran[1]

    xborders = np.linspace(xmin, xmax, nbins + 1)
    xbin = 0.5 * (xborders[:-1] + xborders[1:])

    ybin = np.zeros(nbins)
    dybin = np.zeros(nbins)
    for i in range(nbins):
        aux = (x > xborders[i]) * (x < xborders[i + 1])
        if np.array([aux]).any():
            ybin[i] = np.mean(y[aux])
            dybin[i] = np.std(y[aux])
        else:
            ybin[i] = np.nan
            dybin[i] = np.nan

    if exclude_empty:
        keep = np.logical_not(np.isnan(ybin))
        xbin, ybin, dybin = xbin[keep], ybin[keep], dybin[keep]

    return xbin, ybin, dybin


# ==============================================================================
def find_nearest(array, value):
    '''
    Find the nearest value inside an array
    '''

    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# ==============================================================================
def find_neighbours(par, par_grid, ranges):
    '''
    Finds neighbours' positions of par in par_grid.

    Usage:
    keep, out, inside_ranges, par_new, par_grid_new = \
        find_neighbours(par, par_grid, ranges):

    where redundant columns in 'new' values are excluded,
    but length is preserved (i.e., par_grid[keep] in griddata call).
    '''
    # check if inside ranges

    if len(par) == 4:
        ranges = ranges[0: 4]
    if len(par) == 3:
        ranges = ranges[0: 3]
    # print(par, len(ranges))
    # print(par, len(ranges))
    count = 0
    inside_ranges = True
    while (inside_ranges is True) * (count < len(par)):
        inside_ranges = (par[count] >= ranges[count, 0]) *\
            (par[count] <= ranges[count, 1])
        count += 1

    # find neighbours
    keep = np.array(len(par_grid) * [True])
    out = []

    if inside_ranges:
        for i in range(len(par)):
            # coincidence
            if (par[i] == par_grid[:, i]).any():
                keep *= par[i] == par_grid[:, i]
                out.append(i)
            # is inside
            else:
                # list of values
                par_list = np.array(list(set(par_grid[:, i])))
                # nearest value at left
                par_left = par_list[par_list < par[i]]
                par_left = par_left[np.abs(par_left - par[i]).argmin()]
                # nearest value at right
                par_right = par_list[par_list > par[i]]
                par_right = par_right[np.abs(par_right - par[i]).argmin()]
                # select rows
                kl = par_grid[:, i] == par_left
                kr = par_grid[:, i] == par_right
                keep *= (kl + kr)
        # delete coincidences
        par_new = np.delete(par, out)
        par_grid_new = np.delete(par_grid, out, axis=1)
    else:
        print('Warning: parameter outside ranges.')
        par_new = par
        par_grid_new = par_grid

    return keep, out, inside_ranges, par_new, par_grid_new


# ==============================================================================
def geneva_interp_fast(Par, oblat, t, neighbours_only=True, isRpole=False):
    '''
    Interpolates Geneva stellar models, from grid of
    pre-computed interpolations.

    Usage:
    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)
    or
    Mstar, logL = geneva_interp_fast(Rpole, oblat, t,
                                     neighbours_only=True, isRpole=True)
    (in this case, the option 'neighbours_only' will be set to 'False')

    where t is given in tMS, and tar is the open tar file. For now, only
    Z=0.014 is available.
    '''
    # from my_routines import find_neighbours
    from scipy.interpolate import griddata

    # read grid
    dir0 = 'defs/geneve_models/'
    fname = 'geneva_interp_Z014.npz'
    data = np.load(dir0 + fname)
    Mstar_arr = data['Mstar_arr']
    oblat_arr = data['oblat_arr']
    t_arr = data['t_arr']
    Rpole_grid = data['Rpole_grid']
    logL_grid = data['logL_grid']

    # build grid of parameters
    par_grid = []
    for M in Mstar_arr:
        for ob in oblat_arr:
            for tt in t_arr:
                par_grid.append([M, ob, tt])
    par_grid = np.array(par_grid)

    # set input/output parameters
    if isRpole:
        Rpole = Par
        par = np.array([Rpole, oblat, t])
        Mstar_arr = par_grid[:, 0].copy()
        par_grid[:, 0] = Rpole_grid.flatten()
        neighbours_only = False
    else:
        Mstar = Par
        par = np.array([Mstar, oblat, t])
    # print(par)

    # set ranges
    ranges = np.array([[par_grid[:, i].min(),
                        par_grid[:, i].max()] for i in range(len(par))])

    # find neighbours
    if neighbours_only:
        keep, out, inside_ranges, par, par_grid = \
            find_neighbours(par, par_grid, ranges)
    else:
        keep = np.array(len(par_grid) * [True])
        # out = []
        # check if inside ranges
        count = 0
        inside_ranges = True
        while (inside_ranges is True) * (count < len(par)):
            inside_ranges = (par[count] >= ranges[count, 0]) *\
                (par[count] <= ranges[count, 1])
            count += 1

    # interpolation method
    if inside_ranges:
        interp_method = 'linear'
    else:
        print('Warning: parameters out of available range,' +
              ' taking closest model.')
        interp_method = 'nearest'

    if len(keep[keep]) == 1:
        # coincidence
        if isRpole:
            Mstar = Mstar_arr[keep][0]
            Par_out = Mstar
        else:
            Rpole = Rpole_grid.flatten()[keep][0]
            Par_out = Rpole
        logL = logL_grid.flatten()[keep][0]
    else:
        # interpolation
        if isRpole:
            Mstar = griddata(par_grid[keep], Mstar_arr[keep], par,
                             method=interp_method, rescale=True)[0]
            Par_out = Mstar
        else:
            Rpole = griddata(par_grid[keep], Rpole_grid.flatten()[keep],
                             par, method=interp_method, rescale=True)[0]
            Par_out = Rpole
        logL = griddata(par_grid[keep], logL_grid.flatten()[keep],
                        par, method=interp_method, rescale=True)[0]

    return Par_out, logL


# ==============================================================================
def griddataBA(minfo, models, params, listpar, dims):
    '''
    Moser's routine to interpolate BeAtlas models
    obs: last argument ('listpar') had to be included here
    '''

    idx = np.arange(len(minfo))
    lim_vals = len(params) * [[], ]
    for i in range(len(params)):
        # print(i, listpar[i], params[i], minfo[:, i])
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False),
            phc.find_nearest(listpar[i], params[i], bigger=True)]
        tmp = np.where((minfo[:, i] == lim_vals[i][0]) |
                       (minfo[:, i] == lim_vals[i][1]))
        idx = np.intersect1d(idx, tmp[0])

    out_interp = griddata(minfo[idx], models[idx], params)[0]

    if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0):

        mdist = np.zeros(np.shape(minfo))
        ichk = range(len(params))
        for i in ichk:
            mdist[:, i] = np.abs(minfo[:, i] - params[i]) /\
                (np.max(listpar[i]) - np.min(listpar[i]))
        idx = np.where(np.sum(mdist, axis=1) == np.min(np.sum(mdist, axis=1)))
        if len(idx[0]) != 1:
            out_interp = griddata(minfo[idx], models[idx], params)[0]
        else:
            out_interp = models[idx][0]

    # if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0) or\
    #    bool(np.isnan(np.sum(out_interp))) is True:
    #     print("# Houve um problema na grade e eu nao consegui arrumar...")

    return out_interp

