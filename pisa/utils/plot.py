#
# plots.py
#
# Utility function for plotting maps
# 
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

from utils import is_linear, is_logarithmic

import numpy as np
import matplotlib.pyplot as plt

def show_map(pmap, title=None, cbar = True,
             vmin=None, vmax=None,
             emin=None, emax=None,
             czmin=None, czmax=None,
             invalid=False, logE=None,
             log=False, fontsize=16,
             xlabel=r'cos $\vartheta_\mathrm{zenith}$',
             ylabel='Energy [GeV]',
             **kwargs):
    '''Plot the given map with proper axis labels using matplotlib.
       The axis orientation follows the PINGU convention: 
          - energy to the right
          - cos(zenith) to the top

    Keyword arguments:

      * title: Show this title above the map

      * cbar: show a colorbar for the z-axis (True or False)

      * vmin/vmax: set the minimum/maximum color (z-axis) values.
                   If no value is provided, vmin and vmax are choosen
                   symmetrically to comprehend the full range.

      * emin/emax: set the minimum maximum energy range to show.
                   If not value is provided use the full range covered
                   by the axis.
     
      * czmin/czmax: same as above for cos(zenith) 
      
      * log: use a logarithmic (log10) colour (z-axis) scale

      * logE: show the x-axis on a logarithmic scale (True or False)
              Default is "guessed" from the bins size.

      * invalid: if True show color values for NaN, None, -inf and +inf,
                 otherwise nothing (white) is plotted for these values.


    Uses pyplot.pcolormesh to do the plot. Any additional keyword arguments, in
    particular

      * cmap: defines the colormap
    
    are just passed on to this function.
    '''
    
    #Extract the map to plot, take the log if called for
    cmap = np.log10(pmap['map']) if log else pmap['map']

    #Mask invalid values
    cmap = np.ma.masked_invalid(cmap) if not invalid else cmap
    
    #Get the vertical range
    if not log and vmax is None:
        vmax = np.max(np.abs(np.array(cmap)[np.isfinite(cmap)]))
    if not log and vmin is None:
        vmin = -vmax if (cmap.min() < 0) else 0.

    #Get the energy range
    if emin is None:
        emin = pmap['ebins'][0]
    if emax is None:
        emax = pmap['ebins'][-1]

    #... and for zenith range
    if czmin is None:
        czmin = pmap['czbins'][0]
    if emax is None:
        czmax = pmap['czbins'][-1]

    #Use pcolormesh to be able to show nonlinear spaces
    x,y = np.meshgrid(pmap['czbins'],pmap['ebins'])
    plt.pcolormesh(x,y,cmap,vmin=vmin, vmax=vmax, **kwargs)

    #Add nice labels
    #if xlabel == None:
    #    plt.xlabel(r'cos(zenith)',fontsize=16)
    #else:
    plt.xlabel(xlabel,fontsize=fontsize)
    #if yabel == None:
    #    plt.ylabel('Energy [GeV]',fontsize=16)
    #else:
    plt.ylabel(ylabel,fontsize=fontsize)

    #And a title
    if title is not None:
        plt.suptitle(title,fontsize='x-large')

    axis = plt.gca()
    #Check wether energy axis is linear or log-scale
    if logE is None:
        logE = is_logarithmic(pmap['ebins'])

    if logE:
        axis.semilogy()
    else:
        if not is_linear(pmap['ebins']):
           raise NotImplementedError('Bin edges appear to be neither logarithmically '
                         'nor linearly distributed!')


    #Make sure that the visible range does not extend beyond the provided range
    axis.set_ylim(emin,emax)
    axis.set_xlim(czmin,czmax)

    #Show the colorbar
    if cbar:
        plt.colorbar(format=r'$10^{%.1f}$') if log else plt.colorbar()
    
    #Return axes for further modifications
    return axis
