#
# sim_utils.py - Utilities for accessing IceCube/PINGU/DeepCore
# simulation data, for making cuts and the 1D aeff vs. energy files.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   15 March 2015
#

import tables
import numpy as np
import sys, logging
from pisa.utils.utils import get_bin_sizes

def get_arb_cuts(data, cut_list, mcnu='MCNeutrino', nuIDList=None,
                 cut_sim_down=False):
    '''
    Make arbitrary set of cuts, defined from cut_list and data.

    \params:
      * data - PyTables filehandle object, to obtain the cut information
      * cut_list - list of cuts, expects each element in the list to be a
        tuple of (attribute, column, value) where attribute.col(column) == value
      * mcnu - Monte Carlo neutrino field/key in hdf5 file to use for MC
        true information.
      * nuIDList - if defined, will be a list of flavor IDs to match i.e. [12,-12]
      * cut_sim_down - removes the simulated downgoing neutrinos using MCNeutrino.
    '''

    conditions = []
    try:
        conditions = [data.root.__getattr__(cut[0]).col(cut[1]) == cut[2] for cut in cut_list]
    except:
        conditions = [data.root.__getattribute__(cut[0]).col(cut[1]) == cut[2] for cut in cut_list]

    if nuIDList is not None:
        conditions.append([True if val in nuIDList else False for val in data.root.MCNeutrino.col('type')])
    if cut_sim_down:
        logging.debug("  >>Removing simulated downgoing events!")
        conditions.append(np.cos(data.root.MCNeutrino.col('zenith'))<0.)

    return np.alltrue(np.array(conditions),axis=0)


def get_aeff1D(data,cuts_list,ebins,files_per_run,mcnu='MCNeutrino',
               nc=False,solid_angle=None):
    '''
    Return 1D Aeff directly from simulations (using OneWeight) as a
    histogram, including the error bars.

    \params:
      * data - data table from a hdf5 PyTables file.
      * cuts_list - np array of indices for events passing selection cuts
      * ebins - energy bin edges in GeV that the Aeff histogram
        will be formed from
      * files_per_run - file number normalization to correctly calculate
        simulation weight for Aeff. Should be number of simulation files
        per run for flavour.
      * mcnu - Monte Carlo neutrino field/key in hdf5 file to use for MC
        true information.
      * solid_angle - total solid angle to integrate over. Most of the
        time, we only calculate Aeff for upgoing events, so we do half
        the 4pi solid angle of the whole sky, which would be 2.0*np.pi
      * nc - set this flag to true if we are using NC files.
    '''

    #if not nc:
    nfiles = len(set(data.root.I3EventHeader.col('Run')))*files_per_run
    logging.info("runs: %s"%str(set(data.root.I3EventHeader.col('Run'))))
    logging.info("num runs: %d"%len(set(data.root.I3EventHeader.col('Run'))))
    total_events = data.root.I3MCWeightDict.col('NEvents')[cuts_list]*nfiles/2.0

    # NOTE: solid_angle should be coordinated with get_arb_cuts(cut_sim_down=bool)
    sim_wt_array = (data.root.I3MCWeightDict.col('OneWeight')[cuts_list]/
                    total_events/solid_angle)
    egy_array = data.root.__getattribute__(mcnu).col('energy')[cuts_list]
    aeff,xedges = np.histogram(egy_array,weights=sim_wt_array,bins=ebins)

    egy_bin_widths = get_bin_sizes(ebins)

    aeff /= egy_bin_widths
    aeff*=1.0e-4 # [cm^2] -> [m^2]

    unweighted_events,xedges = np.histogram(egy_array,bins=ebins)

    # So that the error calculation comes out right without divide by zeros...
    unweighted_events = [1. if val < 1.0 else val for val in unweighted_events]

    aeff_error = np.divide(aeff,np.sqrt(unweighted_events))

    logging.debug("percent error on aeff: %s"%str(np.nan_to_num(aeff/aeff_error)))

    return aeff,aeff_error,xedges
