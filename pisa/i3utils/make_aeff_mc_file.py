#! /usr/bin/env python
#
# make_aeff_mc_file.py
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   March 15, 2015
#
# Makes aeff MC file from the raw PyTables HDF5
# simulation/reconstruction i3 files. Keeps separate effective areas
# for all flavours and interaction types (UNLIKE in the reco MC stage).
#

import tables, h5py
import numpy as np

from pisa.utils.log import logging,set_verbosity
from pisa.i3utils.hdfchain import HDFChain
from pisa.i3utils.sim_utils import get_arb_cuts

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

CMSQ_TO_MSQ = 1.0e-4

def get_reco_arrays(data,cuts,files_per_run,reco_string=None,
                    mcnu='MCNeutrino'):
    '''
    Forms arrays of reco events for true/reco energy/coszen from the
    data_files
    '''

    logging.warn('Getting reconstructions from: %s'%reco_string)

    nfiles = len(set(data.root.I3EventHeader.col('Run')))*files_per_run
    sim_weight = ((2.0*data.root.I3MCWeightDict.col('OneWeight')[cuts]*CMSQ_TO_MSQ)/
                  (data.root.I3MCWeightDict.col('NEvents')[cuts]*nfiles))

    try:
        true_cz = np.cos(data.root.__getattr__(mcnu).col('zenith'))[cuts]
        true_egy = data.root.__getattr__(mcnu).col('energy')[cuts]
        reco_cz = np.cos(data.root.__getattr__(reco_string).col('zenith'))[cuts]
        reco_egy = data.root.__getattr__(reco_string).col('energy')[cuts]
    except:
        true_cz = np.cos(data.root.__getattribute__(mcnu).col('zenith'))[cuts]
        true_egy = data.root.__getattribute__(mcnu).col('energy')[cuts]
        reco_cz = np.cos(data.root.__getattribute__(reco_string).col('zenith'))[cuts]
        reco_egy = data.root.__getattribute__(reco_string).col('energy')[cuts]

    arrays = [true_egy,true_cz,reco_egy,reco_cz,sim_weight]

    return arrays

def write_group(nu_group,intType,data_nu):
    '''
    Helper function to write all sim_wt arrays to file
    '''
    if intType not in ['cc','nc']:
        raise Exception('intType: %s unexpected. Expects cc or nc...'%intType)

    sub_group = nu_group.create_group(intType)
    sub_group.create_dataset('true_energy',data=data_nu[0],dtype=np.float32)
    sub_group.create_dataset('true_coszen',data=data_nu[1],dtype=np.float32)
    sub_group.create_dataset('reco_energy',data=data_nu[2],dtype=np.float32)
    sub_group.create_dataset('reco_coszen',data=data_nu[3],dtype=np.float32)
    sub_group.create_dataset('weighted_aeff',data=data_nu[4],dtype=np.float32)

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
parser = ArgumentParser(description='''Takes the simulated (and reconstructed) data files (in hdf5 format) as input and writes out the sim_wt arrays for use in the aeff and reco stage of the template maker. Keeps the nu/nubar and nu nc/ nubar nc separate.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('nue', metavar='HDF5',type=str,
                    help='''nue_data file, which should be a <geometry>_nue.hd5 file.''')
parser.add_argument('numu', metavar='HDF5',type=str,
                    help='''numu_data file, which should be a <geometry>_numu.hd5 file.''')
parser.add_argument('nutau', metavar='HDF5',type=str,
                    help='''nutau_data file, which should be a <geometry>_nutau.hd5 file.''')
parser.add_argument('outfile',metavar='HDF5',type=str,
                    help='''output filename''')
parser.add_argument("--nfiles_nue",type=float,default=200.0,
                    help="Num simulation files for this run")
parser.add_argument("--nfiles_numu",type=float,default=200.0,
                    help="Num simulation files for this run")
parser.add_argument("--nfiles_nutau",type=float,default=200.0,
                    help="Num simulation files for this run")
parser.add_argument('--mn_reco',metavar='STR',type=str,default='MultiNest_8D_Neutrino',
                    help='Reco field to use to access reconstruction parameters')
parser.add_argument('--mcnu',metavar='STR',type=str,default='MCNeutrino',
                    help='Key in hdf5 file from which to extract MC True information')
parser.add_argument('--old_pid',action='store_true',default=False,
                    help='Use older convention for PID enumeration.')
# Step1/Step2 cuts options (or NONE - for most DeepCore analyses):
hcut = parser.add_mutually_exclusive_group(required=True)
hcut.add_argument('--V3cuts',action='store_true',default=False,
                  help='Use V3 version of the cuts')
hcut.add_argument('--V4cuts',action='store_true',default=False,
                  help='Use V4 version of the cuts')
hcut.add_argument('--V5cuts',action='store_true',default=False,
                  help='Use V5 selection cuts')
hcut.add_argument('--nocuts',action='store_true',default=False,
                  help='Do not use any stage of the selection cuts on the files.')
hcut.add_argument('--custom',action='store_true',default=False,
                  help='Use custom cuts, given by args.custom_str')
parser.add_argument('--custom_str',metavar='STR',type=str,
                    default="[('IC86_Dunkman_L6','result',True)]",
                    help='''Python string which will be evaluated as a list of tuples of cuts
                    where tuple is (Attribute,ColName,bool) See function
                    sim_utils.get_arb_cuts for more details if needed.''')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

data_files = {
    'nue': {'filename': args.nue,'nfiles': args.nfiles_nue},
    'numu': {'filename': args.numu,'nfiles': args.nfiles_numu},
    'nutau': {'filename': args.nutau,'nfiles': args.nfiles_nutau}}

logging.info("input files:\n%s"%data_files)

# Ensure overwrite of existing filename...
outfilename = args.outfile
fh = h5py.File(outfilename,'w')
fh.close()
logging.info("Writing to file: %s",outfilename)

# Define V3, V4, or V5 cuts:
cut_list = []
if args.V3cuts:
    logging.warn("Using cuts V3...")
    cut_list.append(('NewestBgRejCutsStep1','value',True))
    cut_list.append(('NewestBgRejCutsStep2','value',True))
elif args.V4cuts:
    logging.warn("Using cuts V4...")
    cut_list.append(('Cuts_V4_Step1','value',True))
    cut_list.append(('Cuts_V4_Step2','value',True))
elif args.V5cuts:
    logging.warn("Using cuts V5...")
    cut_list.append(('Cuts_V5_Step1','value',True))
    cut_list.append(('Cuts_V5_Step2','value',True))
elif args.nocuts:
    logging.warn("Using NO S1/S2 selection CUTS")
    cut_list = []
elif args.custom:
    logging.warn("Using CUSTOM cuts: %s..."%args.custom_str)
    cut_list = eval(args.custom_str)
else:
    # This should never happen...
    logging.warn("NO CUT OPTION DEFINED!!!")


nuDict = {}
if args.old_pid:
    nuDict = {'nue':66,'numu':68,'nutau':133,'nue_bar':67,'numu_bar':69,'nutau_bar':134}
else:
    nuDict = {'nue':12,'numu':14,'nutau':16,'nue_bar':-12,'numu_bar':-14,'nutau_bar':-16}


# Now do CC events, and write to file:
cc_cut_list = cut_list + [('I3MCWeightDict','InteractionType',1)]
nc_cut_list = cut_list + [('I3MCWeightDict','InteractionType',2)]
for flavor in data_files.keys():
    data = tables.openFile(data_files[flavor]['filename'],'r')

    #############
    # First do nu:
    cuts_cc = get_arb_cuts(data,cc_cut_list,nuIDList=[nuDict[flavor]],mcnu=args.mcnu)
    arrays_cc = get_reco_arrays(data,cuts_cc,data_files[flavor]['nfiles'],
                                reco_string=args.mn_reco,mcnu=args.mcnu)
    logging.warn("flavor %s number of CC events: %d"%(flavor,np.sum(cuts_cc)))

    cuts_nc = get_arb_cuts(data,nc_cut_list,nuIDList=[nuDict[flavor]],mcnu=args.mcnu)
    arrays_nc = get_reco_arrays(data,cuts_nc,data_files[flavor]['nfiles'],
                                reco_string=args.mn_reco,mcnu=args.mcnu)
    logging.warn("flavor %s number of NC events: %d"%(flavor,np.sum(cuts_nc)))

    logging.info("Saving %s..."%flavor)
    write_to_hdf5(outfilename,flavor,arrays_cc,arrays_nc)

    ################
    # Next do nu_bar:
    flav_bar = flavor+'_bar'
    cuts_cc = get_arb_cuts(data,cc_cut_list,nuIDList=[nuDict[flav_bar]],mcnu=args.mcnu)
    arrays_cc = get_reco_arrays(data,cuts_cc,data_files[flavor]['nfiles'],
                                reco_string=args.mn_reco,mcnu=args.mcnu)
    logging.warn("flavor %s number of CC events: %d"%(flav_bar,np.sum(cuts_cc)))

    cuts_nc = get_arb_cuts(data,nc_cut_list,nuIDList=[nuDict[flav_bar]],mcnu=args.mcnu)
    arrays_nc = get_reco_arrays(data,cuts_nc,data_files[flavor]['nfiles'],
                                reco_string=args.mn_reco,mcnu=args.mcnu)
    logging.warn("flavor %s number of NC events: %d"%(flav_bar,np.sum(cuts_nc)))

    logging.info("Saving %s..."%flav_bar)
    write_to_hdf5(outfilename,flav_bar,arrays_cc,arrays_nc)

    data.close()

