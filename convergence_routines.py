import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
matplotlib.rcParams['font.family'] = "sans-serif"
font_color = "black"
tick_color = "black"


# ==============================================================================
def plot_convergence(npy, file_name, model):

    converged_idx = 0

    if model == 'befavor':
        linspace = [np.linspace(3.4, 14.6, 8),
                    np.linspace(1.0, 1.45, 6),
                    np.linspace(0.0, 1.00, 5),
                    np.linspace(0.0, 1.0, 5),
                    np.linspace(50, 130, 5),
                    np.linspace(0, 0.1, 5)]
    if model == 'aara' or model == 'acol' or model == 'bcmi':
        linspace = [np.linspace(3.4, 14.6, 8),
                    np.linspace(1.0, 1.45, 6),
                    np.linspace(0.0, 1.00, 5),
                    np.linspace(12., 14.0, 5),
                    np.linspace(0., 30.0, 5),
                    np.linspace(0.5, 2.0, 4),
                    np.linspace(0.0, 1.0, 5),
                    np.linspace(50, 130, 5),
                    np.linspace(0, 0.1, 5)]
    if model == 'beatlas':
        linspace = [np.linspace(3.8, 14.6, 5),
                    np.linspace(1.0, 1.45, 6),
                    np.linspace(0.0, 1.00, 5),
                    np.linspace(3.0, 4.5, 4),
                    np.linspace(0.0, 1.0, 5),
                    np.linspace(50, 130, 5),
                    np.linspace(0, 0.1, 5)]
    # Map the codified parameter names to their sexy latex equivalents
    if model == 'befavor':
        param_to_latex = dict(mass=r'$M\,[M_\odot]$',
                              oblat=r"$R_\mathrm{eq} / R_\mathrm{pole}$",
                              age=r"$H_\mathrm{frac}$", inc=r'$\cos i$',
                              dis=r'$d\,[pc]$', ebv=r'$E(B-V)$')
        params = ["mass", "oblat", "age", "inc", "dis", "ebv"]
        fig = plt.figure(figsize=(16, 20.6))
    if model == 'aara' or model == 'acol' or model == 'bcmi':
        param_to_latex = dict(mass=r'$M\,[M_\odot]$',
                              oblat=r"$R_\mathrm{eq} / R_\mathrm{pole}$",
                              age=r"$H_\mathrm{frac}$",
                              logn0=r'$\log \, n_0 \, [\mathrm{cm}^{-3}]$',
                              rdk=r'$R_\mathrm{D}\, [R_\star]$',
                              inc=r'$\cos i$', nix=r'$m$',
                              dis=r'$d\,[\mathrm{pc}]$', ebv=r'$E(B-V)$')
        params = ["mass", "oblat", "age", "logn0", "rdk", "nix",
                  "inc", "dis", "ebv"]
        fig = plt.figure(figsize=(16, 24.6))
    if model == 'beatlas':
        param_to_latex = dict(mass=r'$M\,[M_\odot]$',
                              oblat=r"$R_\mathrm{eq} / R_\mathrm{pole}$",
                              sig0=r'$\Sigma_0$',
                              nix=r'$n$',
                              inc=r'$\cos i$',
                              dis=r'$d\,[pc]$', ebv=r'$E(B-V)$')
        params = ["mass", "oblat", "sig0", "nix", "inc", "dis", "ebv"]
        fig = plt.figure(figsize=(16, 20.6))

    chain = np.load(npy)

    # chain = chain[(acceptance_fractions > 0.20) &
    #               (acceptance_fractions < 0.5)]

    gs = gridspec.GridSpec(len(params), 3)
    # gs.update(hspace=0.10, wspace=0.025, top=0.85, bottom=0.44)
    gs.update(hspace=0.25)

    for ii, param in enumerate(params):
        these_chains = chain[:, :, ii]

        max_var = max(np.var(these_chains[:, converged_idx:], axis=1))

        ax1 = plt.subplot(gs[ii, :2])

        ax1.axvline(0, color="#67A9CF", alpha=0.7, linewidth=2)

        for walker in these_chains:
            ax1.plot(np.arange(len(walker)) - converged_idx, walker,
                     drawstyle="steps",
                     color=cm.Blues_r(np.var(walker[converged_idx:]) /
                                      max_var),
                     alpha=0.5)

        ax1.set_ylabel(param_to_latex[param], fontsize=30, labelpad=18,
                       rotation="vertical", color=font_color)

        # Don't show ticks on the y-axis
        ax1.yaxis.set_ticks([])

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii == len(params) - 1:
            ax1.set_xlabel("step number", fontsize=24, labelpad=18,
                           color=font_color)
        else:
            ax1.xaxis.set_visible(False)

        ax2 = plt.subplot(gs[ii, 2])

        ax2.hist(np.ravel(these_chains[:, converged_idx:]),
                 bins=np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 20),
                 orientation='horizontal',
                 facecolor="#67A9CF",
                 edgecolor="none",
                 normed=True, histtype='barstacked')

        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()

        # print(ii)
        # print(param)
        ax2.set_yticks(linspace[ii])
        ax1.set_ylim(ax2.get_ylim())

        if ii == 0:
            t = ax1.set_title("Walkers", fontsize=30, color=font_color)
            t.set_y(1.01)
            t = ax2.set_title("Posterior", fontsize=30, color=font_color)
            t.set_y(1.01)

        ax1.tick_params(axis='x', pad=2, direction='out',
                        colors=tick_color, labelsize=22)
        ax2.tick_params(axis='y', pad=2, direction='out',
                        colors=tick_color, labelsize=22)

        ax1.get_xaxis().tick_bottom()

    fig.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.075, top=0.9,
                        left=0.12, right=0.88)
    plt.savefig(file_name + '.png')
