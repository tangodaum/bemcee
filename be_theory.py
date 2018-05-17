import numpy as np
import matplotlib.pylab as plt
import pyhdust.phc as phc
from utils import find_nearest


# ==============================================================================
def oblat2w(oblat):
    '''
    Author: Rodrigo Vieira
    Converts oblateness into wc=Omega/Omega_crit
    Ekstrom et al. 2008, Eq. 9

    Usage:
    w = oblat2w(oblat)
    '''
    if (np.min(oblat) < 1.) or (np.max(oblat) > 1.5):
        print('Warning: values out of allowed range')

    oblat = np.array([oblat]).reshape((-1))
    nw = len(oblat)
    w = np.zeros(nw)

    for iw in range(nw):
        if oblat[iw] <= 1.:
            w[iw] = 0.
        elif oblat[iw] >= 1.5:
            w[iw] = 1.
        else:
            w[iw] = (1.5**1.5) * np.sqrt(2. * (oblat[iw] - 1.) / oblat[iw]**3.)

    if nw == 1:
        w = w[0]

    return w


# ==============================================================================
def t_tms_from_Xc(M, savefig=None, plot_fig=None, ttms_true=None, Xc=None):
    '''
    Calculates the t(tms) for a given Xc and mass
    Xc: float
    M: float
    '''
# ------------------------------------------------------------------------------
    # Parameters from the models
    mass = np.array([14.6, 12.5, 10.8, 9.6, 8.6, 7.7, 6.4, 5.5, 4.8,
                    4.2, 3.8, 3.4])

    nm = len(mass)
    str_mass = ['M14p60', 'M12p50', 'M10p80', 'M9p600', 'M8p600', 'M7p700',
                'M6p400', 'M5p500', 'M4p800', 'M4p200', 'M3p800', 'M3p400']
    st = ['B0.5', 'B1', 'B1.5', 'B2', 'B2.5', 'B3', 'B4', 'B5', 'B6', 'B7',
          'B8', 'B9']
    zsun = 'Z01400'
    str_vel = ['V60000', 'V70000', 'V80000', 'V90000', 'V95000']
    Hfracf = 0.  # end of main sequence

    # ****
    folder_data = 'tables/models/models_bes/'

    if plot_fig is True:
        plt.xlabel(r'$t/t_{MS}$')
        plt.ylabel(r'$X_c$')
        plt.ylim([0.0, 0.8])
        plt.xlim([0.0, 1.0])

# ------------------------------------------------------------------------------
    # Loop (reading the models)
    typ = (1, 3, 16, 21)  # Age, Lum versus Teff versus Hfrac
    arr_age = []
    arr_Hfr = []
    arr_t_tc = []
    cor = phc.gradColor(np.arange(len(st)), cmapn='inferno')
    iv = 2  # O que eh isto?
    arr_Xc = []
    for i in range(nm):
        file_data = folder_data + str_mass[i] + zsun + str_vel[iv] + '.dat'
        age, lum, Teff, Hfrac = np.loadtxt(file_data, usecols=typ,
                                           unpack=True, skiprows=2)
        arr_age.append(age)
        arr_Hfr.append(Hfrac)

        iMS = np.where(abs(Hfrac - Hfracf) == min(abs(Hfrac - Hfracf)))
        X_c = Hfrac[0:iMS[0][0]]
        arr_Xc.append(X_c)

        t_tc = age[0:iMS[0][0]] / max(age[0:iMS[0][0]])
        arr_t_tc.append(t_tc)
    if plot_fig is True:
        plt.plot(t_tc, X_c, color=cor[i], label=('%s' % st[i]))

# ------------------------------------------------------------------------------
# Interpolation
    k = find_nearest(mass, M)[1]

    if plot_fig is True:
        plt.plot(ttms_true, Xc, 'o')
        plt.autoscale()
        plt.minorticks_on()
        plt.legend(fontsize=10, ncol=2, fancybox=False, frameon=False)

# ------------------------------------------------------------------------------

    if savefig is True:
        pdfname = 'Xc_vs_Tsp.png'
        plt.savefig(pdfname)

    return k, arr_t_tc, arr_Xc


# ==============================================================================
def hfrac2tms(Hfrac, inverse=False):
    '''
    Converts nuclear hydrogen fraction into fractional time in
    the main-sequence, (and vice-versa) based on the polynomial
    fit of the average of this relation for all B spectral types
    and rotational velocities.

    Usage:
    t = hfrac2tms(Hfrac, inverse=False)
    or
    Hfrac = hfrac2tms(t, inverse=True)
    '''
    if not inverse:
        coef = np.array([-0.57245754, -0.8041484,
                         -0.51897195, 1.00130795])
        tms = coef.dot(np.array([Hfrac**3, Hfrac**2, Hfrac**1, Hfrac**0]))
    else:
        # interchanged parameter names
        coef = np.array([-0.74740597, 0.98208541, -0.64318363,
                         -0.29771094, 0.71507214])
        tms = coef.dot(np.array([Hfrac**4, Hfrac**3, Hfrac**2,
                                 Hfrac**1, Hfrac**0]))

    # solving problem at lower extreme
    if tms < 0.:
        tms = 0.

    return tms


