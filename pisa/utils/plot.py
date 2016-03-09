#
# plots.py
#
# Utility function for plotting maps
# ... and histograms
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27


import numpy as np
import colorsys

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import is_linear, is_logarithmic


def show_map(pmap, title=None, cbar=True,
             vmin=None, vmax=None,
             emin=None, emax=None,
             czmin=None, czmax=None,
             invalid=False, logE=None,
             log=False, fontsize=16,
             xlabel=r'cos $\vartheta_\mathrm{zenith}$',
             ylabel='Energy [GeV]',
             zlabel=None,
             zlabel_size='large',
             **kwargs):
    """Plot the given map with proper axis labels using matplotlib.

    The axis orientation follows the PINGU convention:
          - energy to the right
          - cos(zenith) to the top

    Parameters
    ----------
    pmap
    title
        Show this title above the map

    cbar
        show a colorbar for the z-axis (True or False)

    vmin/vmax
        set the minimum/maximum color (z-axis) values. If no value is provided,
        vmin and vmax are choosen symmetrically to comprehend the full range.

    emin, emax
        set the minimum maximum energy range to show. If not value is provided
        use the full range covered by the axis.

    czmin, czmax
        same as above for cos(zenith)

    log
        use a logarithmic (log10) colour (z-axis) scale

    logE
        show the x-axis on a logarithmic scale (True or False) Default is
        "guessed" from the bins size.

    invalid
        if True show color values for NaN, None, -inf and +inf, otherwise
        nothing (white) is plotted for these values.

    **kwargs
        Any additional keyword arguments are passed on pyplot.pcolormesh, which
        is used to do the plotting. E.g., `cmap` defines the colormap.
    """
    # Extract the map to plot, take the log if called for
    cmap = np.log10(pmap['map']) if log else pmap['map']

    # Mask invalid values
    cmap = np.ma.masked_invalid(cmap) if not invalid else cmap

    # Get the vertical range
    if not log and vmax is None:
        vmax = np.max(np.abs(np.array(cmap)[np.isfinite(cmap)]))
    if not log and vmin is None:
        vmin = -vmax if (cmap.min() < 0) else 0.

    # Get the energy range
    if emin is None:
        emin = pmap['ebins'][0]
    if emax is None:
        emax = pmap['ebins'][-1]

    # ... and for zenith range
    if czmin is None:
        czmin = pmap['czbins'][0]
    if emax is None:
        czmax = pmap['czbins'][-1]

    # Use pcolormesh to be able to show nonlinear spaces
    x, y = np.meshgrid(pmap['czbins'], pmap['ebins'])
    plt.pcolormesh(x, y, cmap, vmin=vmin, vmax=vmax, **kwargs)

    # Add nice labels
    #if xlabel == None:
    #    plt.xlabel(r'cos(zenith)', fontsize=16)
    #else:
    plt.xlabel(xlabel, fontsize=fontsize)
    #if yabel == None:
    #    plt.ylabel('Energy [GeV]', fontsize=16)
    #else:
    plt.ylabel(ylabel, fontsize=fontsize)

    # And a title
    if title is not None:
        plt.suptitle(title, fontsize=fontsize)

    axis = plt.gca()
    # Check wether energy axis is linear or log-scale
    if logE is None:
        logE = is_logarithmic(pmap['ebins'])

    if logE:
        axis.semilogy()
    else:
        if not is_linear(pmap['ebins']):
           raise NotImplementedError(
               'Bin edges appear to be neither logarithmically nor linearly'
               ' distributed!'
           )

    # Make sure that the visible range does not extend beyond the provided
    # range
    axis.set_ylim(emin, emax)
    axis.set_xlim(czmin, czmax)

    # Show the colorbar
    if cbar:
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if log else plt.colorbar()
        if zlabel:
            col_bar.set_label(zlabel, fontsize=fontsize)
        col_bar.ax.tick_params(labelsize=zlabel_size)

    # Return axes for further modifications
    return axis


def delta_map(amap, bmap):
    """Calculate the differerence between the two maps (amap - bmap), and
    return as a map dictionary.
    """
    if not np.allclose(amap['ebins'], bmap['ebins']) or \
       not np.allclose(amap['czbins'], bmap['czbins']):
       raise ValueError('Map range does not match!')

    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map': amap['map'] - bmap['map']}


def sum_map(amap, bmap):
    """Calculate the sum of two maps (amap + bmap), and return as a map
    dictionary.
    """
    if not np.allclose(amap['ebins'], bmap['ebins']) or \
       not np.allclose(amap['czbins'], bmap['czbins']):
       raise ValueError('Map range does not match!')

    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map' : amap['map'] + bmap['map']}


def ratio_map(amap, bmap):
    """Get the ratio of two maps (amap/bmap) and return as a map dictionary."""
    if (not np.allclose(amap['ebins'], bmap['ebins']) or
        not np.allclose(amap['czbins'], bmap['czbins'])):
        raise ValueError('Map range does not match!')

    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map' : amap['map']/bmap['map']}


def distinguishability_map(amap, bmap):
    """Calculate the Akhmedov-Style distinguishability map from two maps;
    defined as (amap-bmap)/sqrt(amap).
    """
    sqrt_map = {'ebins': amap['ebins'],
                'czbins': amap['czbins'],
                'map':np.sqrt(amap['map'])}
    return ratio_map(delta_map(amap, bmap), sqrt_map)


def stepHist(bin_edges, y, yerr=None,
             plt_lr_edges=False, lr_edge_val=0,
             ax=None, eband_kwargs={}, **kwargs):
    x = np.array(zip(bin_edges, bin_edges)).flatten()
    y = np.array(
        [lr_edge_val] + list(np.array(zip(y, y)).flatten()) + [lr_edge_val],
        dtype=np.float64
    )

    # Populate the y-errors
    if yerr is not None:
        yerr = np.squeeze(yerr)
        if np.isscalar(yerr[0]):
            yerr_low = [-e for e in yerr]
            yerr_high = yerr
        else:
            yerr_low = yerr[0]
            yerr_high = yerr[1]
        yerr_low = np.array(
            [lr_edge_val] +
            list(np.array(zip(yerr_low, yerr_low)).flatten()) +
            [lr_edge_val],
            dtype=np.float64
        )
        yerr_high = np.array(
            [lr_edge_val] +
            list(np.array(zip(yerr_high, yerr_high)).flatten()) +
            [lr_edge_val],
            dtype=np.float64
        )

    # Remove extra values at edges if not plotting the extreme edges
    if not plt_lr_edges:
        x = x[1:-1]
        y = y[1:-1]
        if yerr is not None:
            yerr_low = yerr_low[1:-1]
            yerr_high = yerr_high[1:-1]

    # Create an axis if one isn't supplied
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    # Plot the y-error bands
    err_plt = None
    if yerr is not None:
        custom_hatch = False
        if 'hatch' in eband_kwargs:
            hatch_kwargs = deepcopy(eband_kwargs)
            hatch_kwargs['facecolor'] = (1,1,1,0)
            eband_kwargs['hatch'] = None
            custom_hatch = True
        err_plt = ax.fill_between(x, y1=y+yerr_low, y2=y+yerr_high,
                                  **eband_kwargs)
        if custom_hatch:
            hatch_plt = ax.fill_between(x, y1=y+yerr_low, y2=y+yerr_high,
                                        **hatch_kwargs)

    # Plot the nominal values
    nom_lines = ax.plot(x, y, **kwargs)[0]

    # Match error bands' color to nominal lines
    if yerr is not None and not (('fc' in eband_kwargs) or
                                 ('facecolor' in eband_kwargs)):
        nl_color = nom_lines.get_color()
        nl_lw = nom_lines.get_linewidth()

        ep_facecolor = hsvaFact(nl_color, sf=0.8, vf=1, af=0.5)
        ep_edgecolor = 'none'
        err_plt.set_color(ep_facecolor)
        err_plt.set_facecolor(ep_facecolor)
        err_plt.set_edgecolor(ep_edgecolor)
        err_plt.set_linewidth(nl_lw*0.5)

        if custom_hatch:
            hatch_plt.set_color(eband_kwargs['edgecolor'])
            hatch_plt.set_facecolor((1,1,1,0))
            hatch_plt.set_edgecolor(eband_kwargs['edgecolor'])
            hatch_plt.set_linewidth(nl_lw*0.5)

    return ax, nom_lines, err_plt


def hsvaFact(c, hf=1.0, sf=1.0, vf=1.0, af=1.0, clip=True):
    r, g, b, a = mpl.colors.colorConverter.to_rgba(c)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    ri, gi, bi = colorsys.hsv_to_rgb(h*hf, s*sf, v*vf)
    if clip:
        # Clip all values to range [0,1]
        result = (np.clip(ri,0,1), np.clip(gi,0,1), np.clip(bi,0,1),
                  np.clip(a*af,0,1))
    else:
        # Rescale to fit largest within [0,1]; if all of r,g,b fit in this
        # range, do nothing
        maxval = max(ri, gi, bi)
        # Scale colors if one exceeds range
        if maxval > 1:
            ri /= maxval
            gi /= maxval
            bi /= maxval
        # Clip alpha to range [0,1]
        alpha = np.clip(a*af)
        result = (ri, gi, bi, alpha)
    return result
