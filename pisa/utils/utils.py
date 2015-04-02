#
# utils.py
#
# A set of utility function to deal with maps, etc...
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   2014-01-27

import numpy as np
from scipy.stats import binned_statistic_2d
from pisa.utils.log import logging
import inspect,time


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs


def get_bin_centers(edges):
    '''Get the bin centers for a given set of bin edges.
       This works even if bins don't have equal width.'''
    if is_logarithmic(edges):
        return np.sqrt(np.array(edges[:-1]*np.array(edges[1:])))
    else:
        return (np.array(edges[:-1])+np.array(edges[1:]))/2.

def get_bin_edges_from_cen(centers):
    '''Get the bin edges from a given set of bin centers. This only works
    for log10 or linear binning'''
    if is_logarithmic(centers):


def get_bin_sizes(edges):

    '''Get the bin sizes for a given set of bin edges.
       This works even if bins don't have equal width.'''
    return np.array(edges[1:]) - np.array(edges[:-1])


def is_linear(edges, maxdev = 1e-5):
    '''Check whether the bin edges correspond to a linear axis'''
    linedges = np.linspace(edges[0],edges[-1],len(edges))
    return np.abs(edges-linedges).max() < maxdev


def is_logarithmic(edges, maxdev = 1e-5):
    '''Check whether the bin edges correspond to a logarithmic axis'''
    if np.any(np.array(edges) < 0): return False
    logedges = np.logspace(np.log10(edges[0]),np.log10(edges[-1]),len(edges))
    return np.abs(edges-logedges).max() < maxdev


def is_equal_binning(edges1,edges2,maxdev=1e-8):
    '''Check whether the bin edges are equal.'''
    if (np.shape(edges1)[0]) != (np.shape(edges2)[0]): return False
    return np.abs(edges1 - edges2).max() < maxdev


def is_coarser_binning(coarse_bins, fine_bins):
    '''Check whether coarse_bins lie inside of and are coarser than fine_bins'''
    #contained?
    if ((coarse_bins[0]<fine_bins[0]) or (coarse_bins[-1]>fine_bins[-1])):
        return False
    #actually coarser?
    if (len(fine_bins[np.all([fine_bins>=min(coarse_bins),
                              fine_bins<=max(coarse_bins)], axis=0)]) \
                              < len(coarse_bins)):
        return False
    return True


def subbinning(coarse_bins, fine_bins, maxdev=1e-8):
    '''Check whether coarse_bins can be retrieved from fine_bins
       via integer rebinning.
       * coarse_bins = [coarse_ax1, coarse_ax2, ...]
       * fine_bins = [fine_ax1, fine_ax2, ...]
       where the axes should be 1d numpy arrays'''
    rebin_info = []

    for crs_ax, fn_ax in zip(coarse_bins, fine_bins):
        #Test all possible positions...
        for start in range(len(fn_ax)-len(crs_ax)):
            #...and rebin factors
            for rebin in range(1, (len(fn_ax)-start)/len(crs_ax)+1):
                stop = start+len(crs_ax)*rebin
                if is_equal_binning(crs_ax,
                                    fn_ax[start:stop:rebin],
                                    maxdev=maxdev):
                    rebin_info.append((start, stop-rebin, rebin))
                    break
            else: continue # if no matching binning was found (no break)
            break # executed if 'continue' was skipped (break)
        else: break # don't search on if no binning found for first axis

    if (len(rebin_info) == len(coarse_bins)):
        #Matching binning was found for all axes
        return rebin_info
    else:
        return False


def get_binning(d, iterate=False, eset=[], czset=[]):
    '''Iterate over all maps in the dict, and return the ebins and czbins.
       If iterate is False, will return the first set of ebins, czbins it finds,
       otherwise will return a list of all ebins and czbins arrays'''
    #Only work on dicts
    if not type(d) == dict: return

    #Check if we are on map level
    if (sorted(d.keys()) == ['czbins','ebins','map']):
        #Immediately return if we found one
        if not iterate:
            return np.array(d['ebins']),np.array(d['czbins'])
        else:
            eset += [np.array(d['ebins'])]
            czset += [np.array(d['czbins'])]
    #Otherwise iterate through dict
    else:
        for v in d.values():
            bins = get_binning(v,iterate,eset,czset)
            if bins and not iterate: return bins

    #In iterate mode, return sets
    return eset, czset


def check_binning(data):
    '''
    Check whether all maps in data have the same binning, and return it.
    '''
    eset, czset = get_binning(data,iterate=True)

    for binset, label in zip([eset,czset],['energy','coszen']):
      if not np.alltrue([is_equal_binning(binset[0],bins)
                         for bins in binset[1:]]):
          raise Exception('Maps have different %s binning!'%label)

    return eset[0],czset[0]



def check_fine_binning(fine_bins, coarse_bins):
    """
    This function checks whether the specified fine binning exists and
    is actually finer than the coarse one.
    """

    if fine_bins is not None:
        if is_coarser_binning(coarse_bins, fine_bins):
            logging.info('Using requested binning for oversampling.')
            #everything is fine
            return True
        else:
            logging.error('Requested oversampled binning is coarser '
                          'than output binning. Aborting.')
            sys.exit(1)

    return False


def oversample_binning(coarse_bins, factor):
    """
    Oversample bin edges (coarse_bins) by the given factor
    """

    if is_linear(coarse_bins):
        logging.info('Oversampling linear output binning by factor %i.'
                %factor)
        fine_bins = np.linspace(coarse_bins[0], coarse_bins[-1],
                                factor*(len(coarse_bins)-1)+1)
    elif is_logarithmic(coarse_bins):
        logging.info('Oversampling logarithmic output binning by factor %i.'
                %factor)
        fine_bins = np.logspace(np.log10(coarse_bins[0]),
                                np.log10(coarse_bins[-1]),
                                factor*(len(coarse_bins)-1)+1)
    else:
        logging.warn('Irregular binning detected! Evenly oversampling '
                     'by factor %i'%factor)
        fine_bins = np.array([])
        for i, upper_edge in enumerate(coarse_bins[1:]):
            fine_bins = np.append(fine_bins,
                                  np.linspace(coarse_bins[i], upper_edge,
                                              factor, endpoint=False))

    return fine_bins


def get_smoothed_map(pvals,evals,czvals,e_coarse_bins,cz_coarse_bins):
    '''
    Creates a 'smoothed' oscillation probability map with binning
    given by e_coarse_bins, cz_coarse_bins through the use of the
    scipy.binned_statistic_2d function.

    \params:
      * pvals - array-like object of probability values
      * evals - array-like object of energy (GeV) values
      * czvals - array-like object of coszen values
      * e_coarse_bins - energy bins of final smoothed histogram (probability map)
      * cz_coarse_bins - coszen bins of final smoothed histogram

    '''

    smooth_map = binned_statistic_2d(evals,czvals,pvals,statistic='mean',
                                     bins=[e_coarse_bins,cz_coarse_bins])[0]
    return smooth_map


def integer_rebin_map(prob_map, rebin_info):
    '''
    Rebins a map (or a part of it) by an integer factor in every dimension.
    Merged bins will be averaged.
    '''
    #Make a copy of initial map
    rmap = np.array(prob_map)
    dim = len(rebin_info)

    for start, stop, rebin in np.array(rebin_info)[::-1]:
        #Roll last axis to front
        rmap = np.rollaxis(rmap, dim-1)
        #Select correct part and average
        rmap = np.average([rmap[start+i:stop:rebin] for i in range(rebin)],
                          axis=0)

    return rmap

def inspect_cur_frame():
    '''
    Very useful for showing exactly where the code is executing, in
    tracing down an error or in debugging.
    '''

    (frame,filename,line_num,fn_name,lines,index) = inspect.getouterframes(inspect.currentframe())[1]
    return "%s:%s at %s"%(filename,line_num,fn_name)
