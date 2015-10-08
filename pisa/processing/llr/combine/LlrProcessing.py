#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   18 September 2015
#
# A set of utility functions for dealing with the post processing of
# the LLR Analysis files
#

from __future__ import division, print_function

import h5py
import numpy as np

from pisa.utils.log import logging

def processTrial(combined, partial_run):
    """
    Works with appendTrials (see documentation good)
    """

    good_keys = ['seed','hypo_NMH','hypo_IMH']

    for key,value in partial_run.items():

        if key not in good_keys:
            raise KeyError("ERROR: key: {0:%s} must be one of {1:%s}"
                           .format(key, good_keys))

        if key == 'seed':
            combined[key].append(value)
        else:
            for key1 in partial_run[key].keys():
                if key1 not in combined[key].keys(): combined[key][key1] = []
                combined[key][key1].append(partial_run[key][key1][-1])


    return

def appendTrials(combined, partial_run):
    """
    Appends the data from partial_run (one of the independent
    processing runs on the cluster) into the combined data dictionary.

    Data is actually appended to combined in processTrial() fn.

    \Params:
      * combined - combined hierarchical data dictionary for the llh and log
        info that will be the final output
      * partial_run - data dictionary for the llh data that is aggregated
        into 'combined' data and saved for later processing.

    \Modifies:
      * combined - modified by adding partial_run info to it.

    \Returns:
      * None
    """

    for key in partial_run.keys():

        # Check if combined has been defined this far:
        if key not in combined.keys():
            combined[key] = {nkey: {} for nkey in partial_run[key][0].keys()}
            # Seed needs to be an array:
            combined[key]['seed'] = []

        ntrials =  len(partial_run[key])
        new_keys = partial_run[key][0].keys()

        # Loop over each trial, adding seed and data in
        # true_NMH/true_IMH to
        for ii in xrange(ntrials):
            processTrial(combined[key], partial_run[key][ii])


    return


def saveDict(data,fh):
    """
    Saves a data dictionary to filehandle (or group), fh, as long as the
    group to be written to the hdf5 file is one of:
      [np.ndarray, list, float, int, bool, str, None]
    """

    for k,v in data.items():
        if type(v) == dict:
            group = fh.create_group(k)
            logging.trace("  creating group %s"%k)
            saveDict(v,group)
        else:
            saveNonDict(k, v, fh)

    return


def saveNonDict(key, data, fh):

    logging.trace("    key: %s is of type: %s"%(key,type(data)))
    if type(data) in [np.ndarray,list]:
        logging.trace("    >>saving to dataset: %s"%key)
        fh.create_dataset(key,data=data)
    elif type(data) in [float,int,bool]:
        logging.trace("   >>saving '%s' to dataset: %s"%(str(data),key))
        fh.create_dataset(key,data=data)
    elif type(data) == str:
        logging.trace("   >>saving '%s' to dataset: %s"%(data,key))
        dtype = h5py.special_dtype(vlen=str)
        fh.create_dataset(key,data=data,dtype=dtype)
    elif data is None:
        # NOTE: we convert it to the boolean false here, since
        # I can't find a good way to store the 'None' type in h5py
        data = False
        logging.trace("   >>saving '%s' to dataset: %s"%(data,key))
        fh.create_dataset(key,data=data)
    else:
        raise ValueError("Key: '%s' is unrecognized type: '%s'"%(key,type(data)))

    return
