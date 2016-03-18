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
"""General utility functions"""

import os
import re
import itertools
import inspect
import time
import numbers
import hashlib
import collections

import numpy as np
from scipy.stats import binned_statistic_2d

from pisa.utils import jsons
from pisa.utils.log import logging


class DictWithHash(dict):
    """A dictionary that can carry properties and hashes (and stores) the hash
    of its items.

    Attributes
    ----------
    hash : immutable object
        Value of the hash.
    is_new : bool
        Simple flag that can be polled to see if the object's hash has been
        updated. Resetting this flag after handling must be done by the user.

    Notes
    -----
    The contents are not automatically hashed, as it is beyond the scope of
    this object to figure out if a sub- or sub-sub (etc.) item has changed,
    and therefore the hash has been invalidated.

    Due to `hash` and `is_new` being attributes, these are transparent to
    PISA's to/from_json/hdf methods and will neither be written to nor read
    from files.
    """
    def __init__(self, *args, **kwargs):
        super(DictWithHash, self).__init__(*args, **kwargs)
        # Initialize with np.nan, a value that by default returns False when
        # compared against other objects -- including another np.nan
        self.hash = np.nan

        # The is_new flag is a simple mechanism for keeping track if the data
        # has been updated and, e.g., so a subsequent process must be
        # triggered. I.e., this is a passive polling-based system, vs. e.g.
        # callbacks.
        self.is_new = True

    def update_hash(self, obj_or_hash=None):
        """Update the object's hash.

        obj_or_hash : None, object, or hash value
            Used to update the hash value.
            - If an immutable object (i.e., it implements a `__hash__` method),
              then the hash is derived from the object via hash(obj_or_hash).
              In the case a valid hash value is passed in via `obj_or_hash`,
              hash(obj_or_hash) will simply return the hash value.
            - If a mutable object, the hash_obj() function is called on the
              object.
            - If None, hash_obj() is called on self, so hashing the entire
              contents of the instantiated object.

        Notes
        -----
        The hash_obj() function can be slow for large objects, so it is
        recommended that a simple object be used to update the hash (i.e.,
        avoid `obj_or_hash=None`).
        """
        if obj_or_hash is None and hash_val is None:
            self.hash = hash_obj(self.items())
        elif (hasattr(obj_or_hash, '__hash__') and
              obj_or_hash.__hash__ is not None):
            self.hash = hash(obj_or_hash)
        else:
            self.hash = hash_obj(obj_or_hash)
        self.is_new = True
        return self.hash

    def get_hash(self):
        return self.hash


class LRUCache:
    """Simple implementation of a least-recently-used (LRU) memory cache.
    Specify `depth` to set a limit on the number of entries.

    From: https://www.kunxi.org/blog/2014/05/lru-cache-in-python/"""
    GLOBAL_CACHE_DEPTH_OVERRIDE = None
    def __init__(self, capacity):
        self.capacity = capacity
        if self.GLOBAL_CACHE_DEPTH_OVERRIDE is not None:
            self.capacity = self.GLOBAL_CACHE_DEPTH_OVERRIDE
        self.cache = collections.OrderedDict()

    def get(self, key):
        value = self.cache.pop(key)
        self.cache[key] = value
        if hasattr(value, 'is_new'):
            value.is_new = False
        return value

    def set(self, key, value):
        if self.capacity > 0:
            try:
                self.cache.pop(key)
            except KeyError:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = value


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


def engfmt(n, units, sigfigs=3, decimals=None, sign_always=False):
    """Format number as string in engineering format (10^(multiples-of-three)),
    including the most common metric prefixes (from atto to Exa).

    Parameters
    ----------
    n : numeric
        Number to be formatted
    units : str
        A string that suffixes the output (separated by a space)
    sigfigs : int
        Number of significant figures to limit the result to
    decimals : int or None
        Number of decimals to display (zeros filled out as necessary)
    sign_always : bool
        Prefix the number with "+" sign if number is positive; otherwise,
        only negative numbers are prefixed with a sign ("-")
    """
    prefixes = {-18:'a', -15:'f', -12:'p', -9:'n', -6:'u', -3:'m', 0:'',
                3:'k', 6:'M', 9:'G', 12:'T', 15:'P', 18:'E'}
    # Logs don't like negative numbers...
    sign = np.sign(n)
    n *= sign

    mag = int(np.floor(np.log10(n)))
    pfx_mag = int(np.floor(np.log10(n)/3.0)*3)

    if decimals is None:
        decimals = sigfigs-1 - (mag-pfx_mag)

    round_to = decimals
    if sigfigs is not None:
        round_to = sigfigs-1 - (mag-pfx_mag)

    scaled_rounded = np.round(n/10.0**pfx_mag, round_to)

    sign_str = ''
    if sign_always and sign > 0:
        sign_str = '+'
    num_str = sign_str + format(sign*scaled_rounded, '.'+str(decimals)+'f')

    if pfx_mag not in prefixes:
        return num_str + 'e'+str(mag) + ' ' + units
    return  num_str + ' ' + prefixes[pfx_mag] + units


def timediffstamp(dt_sec, hms_always=False, sigfigs=3):
    """Smart string formatting for a time difference (in seconds)

    Parameters
    ----------
    dt_sec : numeric
        Time difference, in seconds
    hms_always : bool
        * True
            Always display hours, minuts, and seconds regardless of the order-
            of-magnitude of dt_sec
        * False
            Display a minimal-length string that is meaningful, by omitting
            units that are more significant than those necessary to display
            dt_sec; if...
            * dt_sec < 1 s
                Use engineering formatting for the number.
            * dt_sec is an integer in the range 0-59 (inclusive)
                `sigfigs` is ignored and the number is formatted as an integer
            See Notes below for handling of units.
        (Default: False)
    sigfigs : int
        Number of significant figures to display for seconds; zeros are filled
        out as necessary

    Notes
    -----
    If colon notation (e.g. HH:MM:SS.xxx, MM:SS.xxx, etc.) is not used, the
    number is only seconds, and is appended by a space ' ' followed by units
    of 's' (possibly with a metric prefix).
    """
    sign_str = ''
    sgn = 1
    if dt_sec < 0:
        sgn = -1
        sign_str = '-'
    dt_sec = sgn*dt_sec

    r = dt_sec % 3600
    h = int((dt_sec - r)/3600)
    s = r % 60
    m = int((r - s)/60)
    strdt = ''
    if hms_always or h != 0:
        strdt += format(h, '02d') + ':'
    if hms_always or h != 0 or m != 0:
        strdt += format(m, '02d') + ':'

    if float(s) == int(s):
        s = int(s)
        s_fmt = 'd' if len(strdt) == 0 else '02d'
    else:
        # If no hours or minutes, use engineering fmt
        if (h == 0) and (m == 0) and not hms_always:
            return engfmt(dt_sec*sgn, sigfigs=sigfigs, units='s')
        # Otherwise, round seconds to sigfigs-1 decimal digits (so sigfigs
        # isn't really acting as significant figures in this case)
        s = np.round(s, sigfigs-1)
        s_fmt = '.'+str(sigfigs-1)+'f' if len(strdt) == 0 \
            else '06.'+str(sigfigs-1)+'f'
    if len(strdt) > 0:
        strdt += format(s, s_fmt)
    else:
        strdt += format(s, s_fmt) + ' s'

    return sign_str + strdt


def timestamp(d=True, t=True, tz=True, utc=False, winsafe=False):
    """Simple utility to print out a time, date, or time+date stamp for the
    time at which the function is called.

    Parameters
    ----------:
    d : bool
        Include date (default: True)
    t : bool
        Include time (default: True)
    tz : bool
        Include timezone offset from UTC (default: True)
    utc : bool
        Include UTC time/date (as opposed to local time/date) (default: False)
    winsafe : bool
        Omit colons between hours/minutes (default: False)
    """
    if utc:
        time_tuple = time.gmtime()
    else:
        time_tuple = time.localtime()

    dts = ''
    if d:
        dts += time.strftime('%Y-%m-%d', time_tuple)
        if t:
            dts += 'T'
    if t:
        if winsafe:
            dts += time.strftime('%H%M%S', time_tuple)
        else:
            dts += time.strftime('%H:%M:%S', time_tuple)

        if tz:
            if utc:
                if winsafe:
                    dts += time.strftime('+0000')
                else:
                    dts += time.strftime('+0000')
            else:
                offset = time.strftime('%z')
                if not winsafe:
                    offset = offset[:-2:] + '' + offset[-2::]
                dts += offset
    return dts


def hrlist_formatter(start, end, step):
    """Format a range (sequence) in a simple and human-readable format.

    Parameters
    ----------
    start, end, step : numeric

    Notes
    -----
    If `start` and `end` are integers and `step` is 1, step size is omitted.

    The format does NOT follow Python's slicing syntax, in part because the
    interpretation is meant to differ; e.g.,
        '0-10:2' includes both 0 and 10 with step size of 2
    whereas
        0:10:2 (slicing syntax) excludes 10

    Numbers are converted to integers if they are equivalent for more compact
    display.

    Examples
    --------
    >>> hrlist_formatter(start=0, end=10, step=1)
    '0-10'
    >>>> hrlist_formatter(start=0, end=10, step=2)
    '0-10:2'
    >>>> hrlist_formatter(start=0, end=3, step=8)
    '0-3:8'
    >>>> hrlist_formatter(start=0.1, end=3.1, step=1.0)
    '0.1-3.1:1'
    """
    if int(start) == start:
        start = int(start)
    if int(end) == end:
        end = int(end)
    if int(step) == step:
        step = int(step)
    if int(start) == start and int(end) == end and step == 1:
        return '{}-{}'.format(start, end)
    return '{}-{}:{}'.format(start, end, step)


def list2hrlist(lst):
    """Convert a list of numbers to a compact and human-readable string.

    Adapted to make scientific notation work correctly from Scott B's
    adaptation to Python 2 of Rik Poggi's answer to his question
    stackoverflow.com/questions/9847601/convert-list-of-numbers-to-string-ranges

    Examples
    --------
    >>>> list2hrlist([0, 1])
    '0,1'
    >>>> list2hrlist([0, 1, 2])
    '0-2'
    >>>> utils.list2hrlist([0.1, 1.1, 2.1, 3.1])
    '0.1-3.1:1'
    """
    if isinstance(lst, numbers.Number):
        lst = [lst]
    lst = sorted(lst)
    rtol = np.finfo(float).resolution
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if not np.isclose(lst[scan + 2] - lst[scan + 1], step, rtol=rtol):
            result.append(str(lst[scan]))
            scan += 1
            continue
        for j in xrange(scan+2, n-1):
            if not np.isclose(lst[j+1] - lst[j], step, rtol=rtol):
                result.append(hrlist_formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(hrlist_formatter(lst[scan], lst[-1], step))
            return ','.join(result)
    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(itertools.imap(str, lst[scan:])))

    return ','.join(result)


def recursiveEquality(x, y):
    """Recursively verify equality between two objects x and y."""
    # None
    if x is None:
        if y is not None:
            return False
    # Scalar
    elif np.isscalar(x):
        if not np.isscalar(y):
            return False
        if x != y:
            return False
    # Dict
    elif isinstance(x, dict):
        if not isinstance(y, dict):
            return False
        xkeys = sorted(x.keys())
        if not xkeys == sorted(y.keys()):
            return False
        for k in xkeys:
            if not recursiveEquality(x[k], y[k]):
                return False
    # Sequence
    elif hasattr(x, '__len__'):
        if not len(x) == len(y):
            return False
        if isinstance(x, list) or isinstance(x, tuple):
            if not isinstance(y, list) or isinstance(y, tuple):
                return False
            for xs, ys in itertools.izip(x, y):
                if not recursiveEquality(xs, ys):
                    return False
        elif isinstance(x, np.ndarray):
            if not isinstance(y, np.ndarray):
                return False
            if not np.alltrue(x == y):
                return False
        else:
            raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                            (type(x), str(x), str(y)))
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))
    # If you make it to here, must be equal
    return True


def recursiveAllclose(x, y, *args, **kwargs):
    """Recursively verify close-equality between two objects x and y. If
    structure is different between the two objects, returns False

    args and kwargs are passed into numpy.allclose() function
    """
    # None
    if x is None:
        if y is not None:
            return False
    # Scalar
    elif np.isscalar(x):
        if not np.isscalar(y):
            return False
        # np.allclose doesn't handle some dtypes
        try:
            eq = np.allclose(x, y, *args, **kwargs)
        except TypeError:
            eq = x == y
        if not eq:
            return False
    # Dict
    elif isinstance(x, dict):
        if not isinstance(y, dict):
            return False
        xkeys = sorted(x.keys())
        if not xkeys == sorted(y.keys()):
            return False
        for k in xkeys:
            if not recursiveAllclose(x[k], y[k], *args, **kwargs):
                return False
    # Sequence
    elif hasattr(x, '__len__'):
        if not len(x) == len(y):
            return False
        if isinstance(x, list) or isinstance(x, tuple):
            # NOTE: A list is allowed to be allclose to a tuple so long
            # as the contents are allclose
            if not isinstance(y, list) or isinstance(y, tuple):
                return False
            for xs, ys in itertools.izip(x, y):
                if not recursiveAllclose(xs, ys, *args, **kwargs):
                    return False
        elif isinstance(x, np.ndarray):
            # NOTE: A numpy array only evalutes to allclose if compared to
            # another numpy array
            if not isinstance(y, np.ndarray):
                return False
            # np.allclose doesn't handle arrays of some dtypes
            # TODO: this can be rolled into the above clause, I think
            try:
                eq = np.allclose(x, y, *args, **kwargs)
            except TypeError:
                eq = np.all(x == y)
            if not eq:
                return False
        else:
            raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                            (type(x), str(x), str(y)))
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))
    # If you make it to here, must be close
    return True


def test_recursiveEquality():
    d1 = {'one':1, 'two':2, 'three': None}
    d2 = {'one':1.0, 'two':2.0, 'three': None}
    d3 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 2)}},
                 np.arange(3, 4)]}
    d4 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 2)}},
                 np.arange(3, 4)]}
    d5 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 3)}},
                 np.arange(3, 4)]}
    d6 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1.1, 2.1)}},
                 np.arange(3, 4)]}
    assert recursiveEquality(d1, d2)
    assert not recursiveEquality(d1, d3)
    assert recursiveEquality(d3, d4)
    assert not recursiveEquality(d3, d5)
    assert not recursiveEquality(d4, d5)
    assert not recursiveEquality(d3, d6)
    assert not recursiveEquality(d4, d6)

    logging.info('<< PASSED >> recursiveEquality')


def expandPath(path, exp_user=True, exp_vars=True, absolute=False):
    """Convenience function for expanding a path"""
    if exp_user:
        path = os.path.expanduser(path)
    if exp_vars:
        path = os.path.expandvars(path)
    if absolute:
        path = os.path.abspath(path)
    return path


def mkdir(d, mode=0750):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists"""
    try:
        os.makedirs(d, mode=mode)
    except OSError as err:
        if err[0] == 17:
            logging.warn('Directory "' + str(d) + '" already exists')
        else:
            raise err
    else:
        logging.info('Created directory "' + d + '"')


NSORT_RE = re.compile("(\\d+)")
def nsort(l):
    """Numbers sorted by value, not by alpha order.

    Code from
    nedbatchelder.com/blog/200712/human_sorting.html#comments
    """
    return sorted(l, key=lambda a: zip(NSORT_RE.split(a)[0::2],
                                       map(int, NSORT_RE.split(a)[1::2])))


def findFiles(root, regex=None, fname=None, recurse=True, dir_sorter=nsort,
              file_sorter=nsort):
    """Find files by re or name recursively w/ ordering.

    Code adapted from
    stackoverflow.com/questions/18282370/python-os-walk-what-order

    Parameters
    ----------
    root : str
        Root directory at which to start searching for files
    regex : str or re.SRE_Pattern
        Only yield files matching `regex`.
    fname : str
        Only yield files matching `fname`
    recurse : bool
        Whether to search recursively down from the root directory
    dir_sorter
        Function that takes a list and returns a sorted version of it, for
        purposes of sorting directories
    file_sorter
        Function as specified for `dir_sorter` but used for sorting file names

    Yields
    ------
    fullfilepath : str
    basename : str
    match : re.SRE_Match or None
    """
    root = os.path.expandvars(os.path.expanduser(root))
    if isinstance(regex, basestring):
        regex = re.compile(regex)

    # Define a function for accepting a filename as a match
    if regex is None:
        if fname is None:
            def validfilefunc(fn):
                return True, None
        else:
            def validfilefunc(fn):
                if fn == fname:
                    return True, None
                return False, None
    else:
        def validfilefunc(fn):
            match = regex.match(fn)
            if match and (len(match.groups()) == regex.groups):
                return True, match
            return False, None

    if recurse:
        for rootdir, dirs, files in os.walk(root):
            for basename in file_sorter(files):
                fullfilepath = os.path.join(root, basename)
                is_valid, match = validfilefunc(basename)
                if is_valid:
                    yield fullfilepath, basename, match
            for dirname in dir_sorter(dirs):
                fulldirpath = os.path.join(rootdir, dirname)
                for basename in file_sorter(os.listdir(fulldirpath)):
                    fullfilepath = os.path.join(fulldirpath, basename)
                    if os.path.isfile(fullfilepath):
                        is_valid, match = validfilefunc(basename)
                        if is_valid:
                            yield fullfilepath, basename, match
    else:
        for basename in file_sorter(os.listdir(root)):
            fullfilepath = os.path.join(root, basename)
            #if os.path.isfile(fullfilepath):
            is_valid, match = validfilefunc(basename)
            if is_valid:
                yield fullfilepath, basename, match


def get_bin_centers(edges):
    """Get the bin centers for a given set of bin edges.
       This works even if bins don't have equal width."""
    edges = np.array(edges, dtype=np.float)
    if is_logarithmic(edges):
        return np.sqrt(edges[:-1]*edges[1:])
    else:
        return (edges[:-1] + edges[1:])/2.


def get_edges_from_cen(bincen):
    """Get the bin edges from a given set of bin centers. This only works
    for log or linear binning"""
    if is_logarithmic(bincen):
        hwidth = 0.5*(np.log10(bincen[-1]) - np.log10(bincen[0])) \
            / (len(bincen)-1)
        return np.append([10**(np.log10(bincen[0])-hwidth)],
                         10**(np.log10(bincen[:])+hwidth))
    elif is_linear(bincen):
        hwidth = 0.5*(bincen[1] - bincen[0])
        return np.append([bincen[0]-hwidth], bincen[:]+hwidth)
    else:
        raise NotImplementedError('Only bin centers evenly spaced in '
                                  'log or linear space can be computed')


def get_bin_sizes(edges):
    """Get the bin sizes for a given set of bin edges.
    This works even if bins don't have equal width."""
    return np.diff(edges)


def is_linear(edges, maxdev=1e-5):
    """Check whether the bin edges are evenly spaced on a linear scale"""
    # Only 1 bin: might as well be linear
    if len(edges) < 3:
        return True
    bin_widths = np.diff(edges)
    return np.allclose(bin_widths, bin_widths[0], rtol=maxdev)


def is_logarithmic(edges, maxdev=1e-5):
    """Check whether the bin edges are evenly spaced on a log scale"""
    edges = np.array(edges, dtype=np.float)
    # Only 1 bin or <= 0: not log
    if len(edges) < 3 or np.any(edges <= 0):
        return False
    bin_mult_widths = edges[1:] / edges[:-1]
    return np.allclose(bin_mult_widths, bin_mult_widths[0], rtol=maxdev)


def is_equal_binning(edges1, edges2, maxdev=1e-8):
    """Check whether the bin edges are equal."""
    return (np.shape(edges1) == np.shape(edges2)) and np.allclose(edges1,
                                                                  edges2,
                                                                  rtol=maxdev)


def is_coarser_binning(coarse_bins, fine_bins):
    """Check whether coarse_bins lie inside of and are coarser than fine_bins"""
    # contained?
    if (coarse_bins[0] < fine_bins[0]) or (coarse_bins[-1] > fine_bins[-1]):
        return False
    # actually coarser?
    if len(fine_bins[np.all([fine_bins >= min(coarse_bins),
                             fine_bins <= max(coarse_bins)], axis=0)]) \
            < len(coarse_bins):
        return False
    return True


def subbinning(coarse_bins, fine_bins, maxdev=1e-8):
    """Check whether coarse_bins can be retrieved from fine_bins
       via integer rebinning.
       * coarse_bins = [coarse_ax1, coarse_ax2, ...]
       * fine_bins = [fine_ax1, fine_ax2, ...]
       where the axes should be 1d numpy arrays"""
    rebin_info = []

    for crs_ax, fn_ax in zip(coarse_bins, fine_bins):
        # Test all possible positions...
        for start in range(len(fn_ax)-len(crs_ax)):
            # ...and rebin factors
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

    if len(rebin_info) == len(coarse_bins):
        # Matching binning was found for all axes
        return rebin_info

    return False


def get_binning(d, iterate=False, eset=None, czset=None):
    """Iterate over all maps in the dict, and return the ebins and czbins.
       If iterate is False, will return the first set of ebins, czbins it finds,
       otherwise will return a list of all ebins and czbins arrays"""
    # Only work on dicts
    if not isinstance(d, dict):
        return

    eset = [] if eset is None else eset
    czset = [] if czset is None else czset

    # Check if we are on map level
    if (sorted(d.keys()) == ['czbins', 'ebins', 'map']):
        # Immediately return if we found one
        if not iterate:
            return np.array(d['ebins']), np.array(d['czbins'])
        else:
            eset.append(np.array(d['ebins']))
            czset.append(np.array(d['czbins']))
    # Otherwise iterate through dict
    else:
        for v in d.values():
            bins = get_binning(v, iterate, eset, czset)
            if bins and not iterate:
                return bins

    # In iterate mode, return sets
    return eset, czset


def check_binning(data):
    """Check whether all maps in data have the same binning, and return it."""
    eset, czset = get_binning(data, iterate=True)

    for binset, label in zip([eset, czset], ['energy', 'coszen']):
        if not np.alltrue([is_equal_binning(binset[0], bins)
                           for bins in binset[1:]]):
            raise Exception('Maps have different %s binning!'%label)

    return eset[0], czset[0]


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
            errmsg = 'Requested oversampled binning is coarser ' + \
                    'than output binning. Aborting.'
            logging.error(errmsg)
            raise ValueError(errmsg)
    return False


def oversample_binning(coarse_bins, factor):
    """Oversample bin edges (coarse_bins) by the given factor"""
    if factor == 1:
        return coarse_bins

    if is_linear(coarse_bins):
        logging.info('Oversampling linear output binning by factor %i.'%factor)
        fine_bins = np.linspace(coarse_bins[0], coarse_bins[-1],
                                factor*(len(coarse_bins)-1)+1)

    elif is_logarithmic(coarse_bins):
        logging.info('Oversampling logarithmic output binning by factor %i.'
                     % factor)
        fine_bins = np.logspace(np.log10(coarse_bins[0]),
                                np.log10(coarse_bins[-1]),
                                factor*(len(coarse_bins)-1)+1)

    else:
        logging.warn('Irregular binning detected! Evenly oversampling '
                     'by factor %i' % factor)
        fine_bins = np.array([])
        for i, upper_edge in enumerate(coarse_bins[1:]):
            fine_bins = np.append(fine_bins,
                                  np.linspace(coarse_bins[i], upper_edge,
                                              factor, endpoint=False))

    return fine_bins


def get_smoothed_map(pvals, evals, czvals, e_coarse_bins, cz_coarse_bins):
    """Creates a 'smoothed' oscillation probability map with binning
    given by e_coarse_bins, cz_coarse_bins through the use of the
    scipy.binned_statistic_2d function.

    See scipy.stats.binned_statistic_2d for details about arguments; it is
    called with statistic='mean', and so this argument is not user-settable.

    Parameters
    ----------
    pvals, evals, czvals : array-like
        probability, energy (GeV), and coszen values
    e_coarse_bins, cz_coarse_bins
        energy and coszen bins of final smoothed histogram (probability map)

    Returns
    -------
    smooth_map
    """
    smooth_map = binned_statistic_2d(evals, czvals, pvals, statistic='mean',
                                     bins=[e_coarse_bins, cz_coarse_bins])[0]
    return smooth_map


def integer_rebin_map(prob_map, rebin_info):
    """Rebins a map (or a part of it) by an integer factor in every dimension.
    Merged bins will be averaged."""
    # Make a copy of initial map
    rmap = np.array(prob_map)
    dim = len(rebin_info)

    for start, stop, rebin in np.array(rebin_info)[::-1]:
        # Roll last axis to front
        rmap = np.rollaxis(rmap, dim-1)
        # Select correct part and average
        rmap = np.average([rmap[start+i:stop:rebin] for i in range(rebin)],
                          axis=0)

    return rmap


def inspect_cur_frame():
    """Very useful for showing exactly where the code is executing, in tracing
    down an error or in debugging."""

    frame, filename, line_num, fn_name, lines, index = \
        inspect.getouterframes(inspect.currentframe())[1]
    return "%s:%s at %s" % (filename, line_num, fn_name)


def prefilled_map(ebins, czbins, val, dtype=float):
    """Generate a PISA "map" pre-filled with `val` and use datatype `dtype` for
    the data storage part of the map (and *not* for the ebins/czbins storage in
    the map).

    Note that a "map" here is acutally dictionary with structure
        {'ebins': (n_ebins-len array of float),
         'czbins': (n_czbins-len array of float),
         'map': (n_ebins x n_czbins array of dtype)}

    Note also that, elsewhere in PISA, "map" refers to a dictionary that has
    one of the above for each of several neutrino flavors, or
    {'<flavor>': {'<interaction type>': <map> ... }-structured dictionaries, or
    {'<flavor>': {'<interaction type>': {'<signature>': <map> ...}-
    structured dictionaries. Someday maybe the language will be cleaned up so
    as to not be so confusing to mere mortals.
    """
    n_ebins = len(ebins) - 1
    n_czbins = len(czbins) - 1
    newmap = {
        'ebins': ebins,
        'czbins': czbins,
        'map': np.full(shape=(n_ebins, n_czbins), fill_value=val, dtype=dtype)
    }
    return newmap

#import xxhash, dill
def hash_obj(obj):
    """Return hash for an object by serializing the object to a JSON string"""
    #return xxhash.xxh32(dill.dumps(obj)).intdigest()
    if isinstance(obj, np.ndarray) or isinstance(obj, np.matrix):
        return hash(obj.tostring())
    return hash(jsons.json.dumps(obj, sort_keys=True, cls=jsons.NumpyEncoder,
                                 indent=None, ensure_ascii=False,
                                 check_circular=True, allow_nan=True,
                                 separators=(',', ':')))


def hash_file(fname):
    """Return a hash for a file

    Currently, uses md5 sum as hash algorithm.
    """
    md5 = hashlib.md5()
    md5.update(file(fname, 'rb').read())
    return md5.hexdigest()


if __name__ == "__main__":
    test_recursiveEquality()
