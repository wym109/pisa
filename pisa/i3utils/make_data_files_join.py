#! /usr/bin/env python
#
# make_data_files_join.py
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   July 7, 2014
#
# Makes data files, used in MC-based reco step, but uses approximation
# of NO DISTINCTION between nu/nu_bar OR ANY NC. However, it maintains
# their distinction at the final level of writing to the output file, to
# conform to the way the pisa code is written which will allow for the
# possibility later on to actually use separate distributions for nu/nubar
# and cc/nc.
#

import logging, tables, h5py
import numpy as np
from pisa.utils.log import set_verbosity
from hdfchain import HDFChain
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

CMSQ_TO_MSQ = 1.0e-4

def get_arb_cuts(data,cut_list):
    '''
    Make arbitrary set of cuts, defined from cut_list and data, which is a
    PyTables file object.
    '''
    conditions = []
    try:
        conditions = [data.__getattr__(cut[0]).col(cut[1]) == cut[2] for cut in cut_list]
    except:
        conditions = [data.__getattribute__(cut[0]).col(cut[1]) == cut[2] for cut in cut_list]

    return np.alltrue(np.array(conditions),axis=0)

def get_arrays(data,cuts,files_per_run):
    '''
    Forms arrays of sim_wt and true/reco energy/coszen from the data_files
    '''

    nfiles = len(set(data.I3EventHeader.col('Run')))*files_per_run
    sim_weight = ((2.0*data.I3MCWeightDict.col('OneWeight')[cuts]*CMSQ_TO_MSQ)/
                  (data.I3MCWeightDict.col('NEvents')[cuts]*nfiles))
    true_egy = data.MCNeutrino.col('energy')[cuts]
    true_cz = np.cos(data.MCNeutrino.col('zenith'))[cuts]
    reco_egy = data.MultiNest_Neutrino.col('energy')[cuts]
    try:
        reco_cz = np.cos(data.MultiNest_Neutrino.col('zenith'))[cuts]
    except:
        reco_cz = np.cos(data.MultiNest_BestFitParticle.col('zenith'))[cuts]

    arrays = [sim_weight,true_egy,true_cz,reco_egy,reco_cz]

    return arrays

def write_group(nu_group,intType,data_nu):
    '''
    Helper function to write all sim_wt arrays to file
    '''
    if intType not in ['cc','nc']:
        raise Exception('intType: %s unexpected. Expects cc or nc...'%intType)

    sub_group = nu_group.create_group(intType)
    sub_group.create_dataset('weighted_aeff',data=data_nu[0],dtype=np.float32)
    sub_group.create_dataset('true_energy',data=data_nu[1],dtype=np.float32)
    sub_group.create_dataset('true_coszen',data=data_nu[2],dtype=np.float32)
    sub_group.create_dataset('reco_energy',data=data_nu[3],dtype=np.float32)
    sub_group.create_dataset('reco_coszen',data=data_nu[4],dtype=np.float32)

    return

def write_to_hdf5(outfilename,flavor,data_nu_cc,data_nu_nc):
    '''
    Writes the sim wt arrays to outfilename, for single flavour's cc/nc fields.
    '''
    fh = h5py.File(outfilename,'a')
    nu_group = fh.create_group(flavor)

    write_group(nu_group,'cc',data_nu_cc)
    write_group(nu_group,'nc',data_nu_nc)
    fh.close()
    return


set_verbosity(0)
parser = ArgumentParser(description='''Takes the simulated data files as input and writes
    out the sim_wt arrays for use in the EventCountsOsc stage of the map generation''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('nue', metavar='<nue .hd5 file>',type=str,
                    help='''nue_data file, which should a <geometry>_nue.hd5 file.''')
parser.add_argument('numu', metavar='<numu .hd5 file>',type=str,
                    help='''numu_data file, which should a <geometry>_numu.hd5 file.''')
parser.add_argument('nutau', metavar='<nutau .hd5 file>',type=str,
                    help='''nutau_data file, which should a <geometry>_nutau.hd5 file.''')
parser.add_argument('outfile',metavar='<outfile.hd5>',type=str,
                    help='''output filename''')
parser.add_argument('--files_per_run',metavar="<NUM_FILES>",type=int,default=500,
                    help='''Files per simulation run for each of the input files.''')
select_cuts = parser.add_mutually_exclusive_group(required=True)
select_cuts.add_argument('--cutsV3', default=False, action='store_true',
                         help="Use V3 selection cuts.")
select_cuts.add_argument('--cutsV4', default = False, action='store_false',
                         help="Use V4 selection cuts.")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

data_files = {'nue':args.nue,'numu':args.numu,'nutau':args.nutau}
logging.info("input files: %s"%data_files)

# Ensure overwrite of existing filename...
outfilename = args.outfile
fh = h5py.File(outfilename,'w')
fh.close()
logging.info("Writing to file: %s",outfilename)

# Define V3 or V4 cuts:
cut_list = []
if args.cutsV3:
    logging.warn("Using cuts V3...")
    cut_list.append(('NewestBgRejCutsStep1','value',True))
    cut_list.append(('NewestBgRejCutsStep2','value',True))
if args.cutsV4:
    logging.warn("Using cuts V4...")
    cut_list.append(('Cuts_V4_Step1','value',True))
    cut_list.append(('Cuts_V4_Step2','value',True))

# First do NC events:
for f in data_files.values():
    dummy = tables.openFile(f,mode='r')
data_nc = HDFChain(data_files.values()).root
nc_cut_list = cut_list + [('I3MCWeightDict','InteractionType',2)]
cuts_nc = get_arb_cuts(data_nc,nc_cut_list)
arrays_nc = get_arrays(data_nc,cuts_nc,args.files_per_run)
logging.warn("NC number of events: %d"%np.sum(cuts_nc))

# Now do CC events, and write to file:
cc_cut_list = cut_list + [('I3MCWeightDict','InteractionType',1)]
for flavor in data_files.keys():
    data = tables.openFile(data_files[flavor],'r').root

    cuts_cc = get_arb_cuts(data,cc_cut_list) #get_cuts_cc(data,flavor)
    arrays_cc = get_arrays(data,cuts_cc,args.files_per_run)
    logging.warn("flavor %s number of events: %d"%(flavor,np.sum(cuts_cc)))

    logging.info("Saving %s..."%flavor)
    write_to_hdf5(outfilename,flavor,arrays_cc,arrays_nc)

    flavor+='_bar'
    logging.info("Saving %s..."%flavor)
    write_to_hdf5(outfilename,flavor,arrays_cc,arrays_nc)


