#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   2015 September 17
#
# Tool for post processing on the llr analysis runs that have been run
# on a HPCC (GPU or CPU). Aggregates the structured data of the llh
# runs as well as the highly unstructured data of the log files.
#
# WARNING: I don't THINK this will handle the case (in the logging portion)
# when the --no_alt_fit flag is given.
#


from __future__ import division, print_function

import time
import os
import h5py
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import set_verbosity, logging
from pisa.utils.jsons import from_json
from LogProcessing import getTimeStamp, processLogFile
from LlrProcessing import processTrial, appendTrials, saveDict, saveNonDict

def modifyHierarchyKeys(output_data):
    """
    The keys designating the "Normal" or "Inverted" hierarchy have
    originally been named "NMH" and "IMH" but we'd like to switch to
    "NH" and "IH".

    This function visits all keys in the output_data dict of the form
    "true_*" or "hypo_*" and changes NMH/IMH to NH/IH.
    """

    for key1 in output_data.keys():
        if 'true' not in key1: continue
        key1_new = key1.replace('MH','H')
        output_data[key1_new] = output_data.pop(key1)
        for key2 in output_data[key1_new].keys():
            for key3 in output_data[key1_new][key2].keys():
                if 'hypo' not in key3: continue
                key3_new = key3.replace('MH','H')
                output_data[key1_new][key2][key3_new] = output_data[key1_new][key2].pop(key3)

    return

def saveOutput(data, outfile):
    """
    Saves the output_data dictionary into the provided outfile.
    Nothing is modified or returned in this function.
    """

    fh = h5py.File(args.outfile,'w')

    for key in sorted(data.keys()):
        if type(data[key]) is dict:
            group = fh.create_group(key)
            saveDict(data[key],group)
        else:
            saveNonDict(key, data[key], fh)

    fh.close()
    return


parser = ArgumentParser(
    description="""Processes the llr analysis runs that have been generated on
    a HPCC on multiple files.""",formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('-d','--data_dir', metavar='DIR', type=str, required=True,
                    help='Directory where the llh analysis run data is stored.')
parser.add_argument('-l','--log_dir', metavar='DIR', type=str, required=False,
                    default=None,
                    help='Directory where the llh analysis run log info is.')
parser.add_argument('-o', '--outfile', metavar='STR', type=str, required=True,
                    help="Output file to store processed, combined llh file.")
parser.add_argument('--fix_keys', action='store_true', default=False,
                    help='If keys are named "NMH"/"IMH", change to "NH"/"IH"')
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='''set verbosity level''')
# Do we need a flag for no_alt_fit?

args = parser.parse_args()
set_verbosity(args.verbose)

llhfiles = glob(os.path.join(args.data_dir,'llh_data*'))

if args.log_dir is not None:
    logfiles = glob(os.path.join(args.log_dir,'log*'))
    # These MUST have the same number initialized if we are using the logging
    # information. Otherwise, perhaps one of the directories are incorrect.
    # Sometimes there are fewere llh files, since they crash before writing out.
    assert(len(llhfiles) <= len(logfiles)),"Data and log directories don't match?"

# Output to save to hdf5 file:
output_data = {'minimizer_settings': {},
               'template_settings': {},
               'true_NMH': {},
               'true_IMH': {}}

logging.warn("Processing {0:d} files".format(len(llhfiles)))

mod = len(llhfiles)//20
start = time.time()
for i,filename in enumerate(llhfiles):

    if (mod > 0) and (i%mod == 0):
        logging.info("  >> {0:d} files done...".format(i))

    try:
        data = from_json(filename)
    except Exception as inst:
        #print(inst)
        print("Skipping file: ",filename)
        continue

    if not output_data['minimizer_settings']:
        output_data['minimizer_settings'] = data['minimizer_settings']

    if not output_data['template_settings']:
        output_data['template_settings'] = data['template_settings']

    for key in ['true_NMH','true_IMH']:
        appendTrials(output_data[key],data[key])


    if args.log_dir is not None:
        # Now process corresponding log file:
        # ASSUMES that llh, log files are written to directory as:
        #   llh_data_<#>, log_<#>
        # where '#' in range(1, nfiles)
        logfilename = os.path.join(args.log_dir,
                                   ('log_'+filename.split('_')[-1]))
        
        #print("File: ",logfilename)
        fh = open(logfilename, 'r')
        all_lines = fh.readlines()
        iline = 0
        fh.close()

        # If this is the first pass through, write logging containers:
        if 'timestamp' not in output_data.keys():

            # First write timestamp if not yet recorded:
            try:
                timestamp, iline = getTimeStamp(iline, all_lines)
            except:
                print("File failed: \n    ",logfilename)
                raise
                
            output_data['timestamp'] = timestamp

            # Second write the logging portion of the dictionary container
            for k1 in ['true_NMH', 'true_IMH']:
                for k2 in ['true_h_fiducial','false_h_best_fit']:
                    for k3 in ['hypo_NMH', 'hypo_IMH']:
                        output_data[k1][k2][k3]['optimizer_time'] = []
                        output_data[k1][k2][k3]['mean_template_time'] = []
                        output_data[k1][k2][k3]['warnflag'] = []
                        output_data[k1][k2][k3]['task'] = []
                        output_data[k1][k2][k3]['nit'] = []
                        output_data[k1][k2][k3]['funcalls'] = []
        
        # Now gather all log file information for this partial run:
        try:
            processLogFile(iline, all_lines, output_data)
        except:
            print("File failed: \n    ",logfilename)
            raise


delta_sec = (time.time() - start)
logging.warn("Time to process the LLR Run: {0:.4f} sec".format(delta_sec))

# Fixing the keys from NMH/IMH --> NH/IH
if args.fix_keys: modifyHierarchyKeys(output_data)

base_key = 'true_NH' if args.fix_keys else 'true_NMH'
logging.info("num trials for NH: ")
# In general, this level will have keys: 'true_h_fiducial' and 'false_h_best_fit'
# But when run with flag '--no_alt_fit', the 'false_h_best_fit' is missing
for key1 in ['true_h_fiducial','false_h_best_fit']:
    if key1 not in output_data[base_key].keys(): continue
    for key2 in output_data[base_key][key1].keys():
        if 'hypo' not in key2: continue
        logging.info(
            "key1: {0:s}, key2: {1:s}, ntrials: {2:d}".format(
                key1, key2,len(output_data[base_key][key1][key2]['llh'])))

saveOutput(output_data, args.outfile)
