from kde.cudakde import gaussian_kde
import numpy as np
from uncertainties import unumpy as unp
from pisa.utils.profiler import profile, line_profile
from pisa.core.binning import OneDimBinning, MultiDimBinning

#@line_profile
def kde_histogramdd(sample, binning, weights=[],bw_method='scott',adaptive=True,
                        alpha=0.3,use_cuda=False,coszen_reflection=0.25,coszen_name = 'coszen',oversample=1):
    '''
    function to run kernel density estimation (KDE) for an array of data points, and
    then evaluate the on a histogram like grid, to effectively produce a histogram-like output

    this is based on sebastian schoenen's KDE implementation:
    http://code.icecube.wisc.edu/svn/sandbox/schoenen/kde/

    sample : nd-array of shape (N_evts, vars), with vars in the
                right order corresponding to the binning order
    binning : pisa MultiDimBinning
    weights : array
    bw_method: scott or silverman
    adaptive : bool
    alpha : some parameter for the KDEs
    use_cuda : run on GPU (only allowing <= 2d)
    coszen_reflection : part (number between 0 and 1) of binning that is refelct at the coszen -1 and 1 egdes
    coszen_name : binning name to identigy the coszen bin that needs to undergo special treatement for reflection
    oversample : int, evaluate KDE at more points per bin, takes longer, but is more accurate
    '''
    # compute raw histo for norm
    #bins = [unp.nominal_values(b.bin_edges) for b in binning]
    #if len(weights) == 0:
    #    raw_hist,e = np.histogramdd(sample,bins=bins)
    #else:
    #    raw_hist,e = np.histogramdd(sample,bins=bins,weights=weights)
    #norm = np.sum(raw_hist)
    if len(weights) == 0:
        norm = sample.shape[0]
    else:
        norm = np.sum(weights)
    #oversample
    if not oversample==1:
        binning = binning.oversample(oversample)
    # flip around to satisfy the kde implementation
    x = sample.T
    # must have same amount of dimensions as binning dimensions
    assert x.shape[0] == len(binning)
    cz_bin = binning.names.index(coszen_name)
    # normal hist

    #swap out cz bin to first place (index 0)
    if not cz_bin == 0:
        #also swap binning:
        new_binning = [binning[coszen_name]]
        for b in binning:
            if not b.name == coszen_name:
                new_binning.append(b)
        binning = MultiDimBinning(new_binning)
        x[[0,cz_bin]] = x[[cz_bin,0]]
    # check if edge needs to be reflected
    reflect_lower = binning[coszen_name].bin_edges[0] == -1
    reflect_upper = binning[coszen_name].bin_edges[-1] == 1
    # get the kernel weights
    kernel_weights_adaptive = gaussian_kde(x,weights=weights,bw_method=bw_method,adaptive=adaptive, alpha=alpha,use_cuda=True)
    # get the bin centers, where we're going to evaluate the kdes at, en extend the bin range for reflection
    bin_points = []
    for b in binning:
        c = unp.nominal_values(b.weighted_centers)
        if b.name == coszen_name:
            # how many bins to add for reflection
            l = int(len(c)*coszen_reflection)
            if reflect_lower:
                c0 = 2*c[0] - c[1:l+1][::-1]
            else:
                c0 = []
            if reflect_upper:
                c1 = 2*c[-1] -c[-l-1:-1][::-1]
            else:
                c1 = []
            c = np.concatenate([c0,c,c1])
        bin_points.append(c)
    # shape including reflection edges
    megashape = (binning.shape[0] + (int(reflect_upper)+int(reflect_lower))*l, binning.shape[1])
    # shape of the reflection edges alone
    minishape = (binning.shape[0] - l, binning.shape[1])
    # create a set of points
    grid = np.meshgrid(*bin_points,indexing='ij')
    points = np.array([g.ravel() for g in grid])
    # evaluate KDEs at given points
    hist = kernel_weights_adaptive(points)
    # reshape 1d array into nd
    hist = hist.reshape(megashape)
    # cut off the reflection edges, mirror then, fill up remaining space with zeros and add to histo
    if reflect_lower:
        hist0 = hist[0:l,:]
        hist0_0 = np.zeros(minishape)
        hist0 = np.flipud(np.concatenate([hist0_0, hist0]))
        hist = hist[l:,:]
    else:
        hist0 = 0
    if reflect_upper:
        hist1 = hist[-l:,:]
        hist1_0 = np.zeros(minishape)
        hist1 = np.flipud(np.concatenate([hist1, hist1_0]))
        hist = hist[:-l,:]
    else:
        hist1 = 0
    hist = hist + hist1 + hist0
    # bin volumes
    volume = binning.bin_volumes(attach_units=False)
    hist = np.multiply(hist,volume)
    #downsample
    if not oversample==1:
        for i,b in enumerate(binning):
            hist = np.add.reduceat(hist, np.arange(0,len(b.bin_edges)-1,oversample), axis=i)
    # swap back the axes
    if not cz_bin == 0:
        hist = np.swapaxes(hist,0, cz_bin)
    #hist = hist/np.sum(hist)*norm
    return hist*norm

if __name__ == '__main__':
    from pisa.core.map import Map, MapSet
    from pisa.utils.plotter import Plotter
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from pisa.utils.log import logging, set_verbosity

    parser = ArgumentParser()
    parser.add_argument('-v', action='count', default=None,
                         help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    my_plotter = Plotter(stamp='', outdir='.', fmt='pdf', log=False, annotate=False, symmetric=False, ratio=True)

    b1 = OneDimBinning(name='coszen', num_bins=100, is_lin=True, domain=[-1, 1], tex= r'$\cos(\theta)$')
    b2 = OneDimBinning(name='energy', num_bins=10, is_log=True, domain=[0.1, 10], tex=r'$E$')
    binning = MultiDimBinning([b2, b1])
    x = np.random.normal(1,1,(2,100000))
    x = np.array([np.abs(x[0])-1,x[1]])
    # cut away outside csozen
    x = x.T[(x[0]<=1) & (x[0] >= -1),:].T
    #swap
    x[[0,1]] = x[[1,0]]
    bins = [unp.nominal_values(b.bin_edges) for b in binning]
    raw_hist,e = np.histogramdd(x.T,bins=bins)

    hist = kde_histogramdd(x.T,binning,bw_method='silverman',coszen_name='coszen',oversample=10)
    #hist = hist/np.sum(hist)*np.sum(raw_hist)
    m1 = Map(name='KDE', hist=hist, binning=binning)
    m2 = Map(name='raw', hist=raw_hist, binning=binning)
    m = MapSet([m1,m2,m1/m2,m1-m2])
    my_plotter.plot_2d_array(m, fname='test_kde', cmap='summer')
