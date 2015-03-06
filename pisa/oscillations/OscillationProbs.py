#! /usr/bin/env python
#
# OscillationProbs.py
#
# This module calculates the bare neutrino oscillation probabilities
# for all neutrino flavours and writes them to disk.
# It is not part of the main pisa simulation chain!
#
# author: Lukas Schulte
#         lschulte@physik.uni-bon.de
#
# date:   July 28, 2014
#

import os,sys
from argparse import ArgumentParser
import numpy as np
import h5py
from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.proc import report_params, get_params, add_params
from pisa.oscillations.TableOscillationService import TableOscillationService
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.NucraftOscillationService import NucraftOscillationService

# Until python2.6, default json is very slow.
try:
    import simplejson as json
except ImportError, e:
    import json


def to_hdf5(oscprob_dict, filename, param_dict):

    fh = h5py.File(filename,'w')

    edata = fh.create_dataset('ebins',data=oscprob_dict['ebins'],dtype=np.float32)
    czdata = fh.create_dataset('czbins',data=oscprob_dict['czbins'],dtype=np.float32)

    for key in oscprob_dict.keys():
        if 'maps' in key:
            logging.debug("  key %s",key)
            group_base = fh.create_group(key)
            for subkey in oscprob_dict[key].keys():
                logging.debug("    subkey %s",subkey)
                dset = group_base.create_dataset(subkey,data=oscprob_dict[key][subkey],dtype=np.float32)
                dset.attrs['ebins'] = edata.ref
                dset.attrs['czbins'] = czdata.ref

    param_group = fh.create_group("params")
    logging.debug("  saving param dict...")
    for key in param_dict.keys():
        param_group.create_dataset(key,data=param_dict[key])

    fh.close()
    return


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(description='Takes the oscillation parameters '
                            'as input and writes out a set of osc flux maps')
    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
        help= '''Edges of the energy bins in units of GeV, default is '''
              '''80 edges (79 bins) from 1.0 to 80 GeV in logarithmic spacing.''',
        default = np.logspace(np.log10(1.),np.log10(80),40))
    parser.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
        help= '''Edges of the cos(zenith) bins, default is '''
              '''21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.''',
        default = np.linspace(-1.,0.,21))
    parser.add_argument('--deltam21',type=float,default=7.54e-5,
                        help='''deltam21 value [eV^2]''')
    parser.add_argument('--deltam31',type=float,default=0.00246,
                        help='''deltam31 value [eV^2]''')
    parser.add_argument('--theta12',type=float,default=0.5873,
                        help='''theta12 value [rad]''')
    parser.add_argument('--theta13',type=float,default=0.1562,
                        help='''theta13 value [rad]''')
    parser.add_argument('--theta23',type=float,default=0.6745,
                        help='''theta23 value [rad]''')
    parser.add_argument('--deltacp',type=float,default=np.pi,
                        help='''deltaCP value to use [rad]''')
    parser.add_argument('--earth-model',type=str,default='oscillations/PREM_60layer.dat',
                        dest='earth_model',
                        help='''Earth model data (density as function of radius)''')
    parser.add_argument('--energy-scale',type=float,default=1.0,
                        dest='energy_scale',
                        help='''Energy off scaling due to mis-calibration.''')
    parser.add_argument('--code',type=str,choices = ['prob3','table','nucraft'],
                        default='prob3',
                        help='''Oscillation code to use, one of
                        [table, prob3, nucraft]''')
    parser.add_argument('--oversample_e', type=int, default=13,
			help='''oversampling factor for energy;
			i.e. every energy bin will be oversampled by this factor''')
    parser.add_argument('--oversample_cz', type=int, default=12,
			help='''oversampling factor for cos(zen);
			i.e. every energy bin will be oversampled by this factor''')
    parser.add_argument('--detector-depth', type=float, default=2.0,
                        dest='detector_depth',
                        help='''Detector depth in km''')
    parser.add_argument('--propagation-height', default=None,
                        dest='prop_height',
                        help='''Height in the atmosphere to begin propagation in km.
                        Prob3 default: 20.0 km
                        NuCraft default: 'sample' from a distribution''')
    parser.add_argument('--precision', type=float, default=5e-4,
                        dest='osc_precision',
                        help='''Requested precision for unitarity (NuCraft only)''')
    parser.add_argument('--datadir', dest='datadir', metavar='DIR',
                        type=str, action='store',default="oscillations",
                        help='''Directory holding the pre-calculated oscillation
                        probabilities in hdf5 format (default: oscillations, only
                        needed if code==table)''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store',default="osc_probs.json",
                        help='''File to store the output. Format is guessed
                        from filename extension: either JSON (.json/.js) or
                        HDF5 (.hdf/.hd5/.hdf5). Default: osc_probs.json''')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    
    #Collect settings in a dict
    settings = vars(args).copy()
    
    #Set verbosity level
    set_verbosity(args.verbose)

    #Check requested format
    fmt = os.path.splitext(args.outfile)[1]
    if fmt in ['.json', '.js']:
        fmt = 'JSON'
    elif fmt in ['.hdf', '.hd5', '.hdf5']:
        fmt = 'HDF5'
    else:
        raise NotImplementedError('Unknown file format: %s'%fmt)
    logging.info('Will store in %s format'%fmt)

    #Initialize an oscillation service
    iniargs = {'earth_model': args.earth_model,
               'detector_depth': args.detector_depth,
               'prop_height': args.prop_height,
               'osc_precision': args.osc_precision,
               'datadir': args.datadir}
    
    logging.info('Using %s oscillation code'%args.code)
    if args.code=='prob3':
      if iniargs['prop_height'] is None: iniargs['prop_height'] = 20.0
      osc_service = Prob3OscillationService(args.ebins,args.czbins,**iniargs)
    elif args.code=='nucraft':
      if iniargs['prop_height'] is None: iniargs['prop_height'] = 'sample'
      else: iniargs['prop_height']=float(iniargs['prop_height'])
      osc_service = NucraftOscillationService(args.ebins,args.czbins,**iniargs)
    elif args.code=='table':
      osc_service = TableOscillationService(args.ebins,args.czbins,**iniargs)
    else:
      raise NotImplementedError('Unknown oscillation service: %s'%args.code)
    
    #Do calculation
    logging.info("Calculating oscillation probabilities")
    settings.pop('ebins')
    settings.pop('czbins')
    osc_prob_maps = osc_service.get_osc_prob_maps(**settings)
    
    #Remove irrelevant parameters from settings
    for par in ['verbose', 'outfile']:#, 'ebins', 'czbins']:
        settings.pop(par)
    
    #Write out
    logging.info("Saving output to: %s",args.outfile)
    if fmt=='JSON':
        osc_prob_maps['params'] = settings
        to_json(osc_prob_maps, args.outfile)
    elif fmt=='HDF5':
        to_hdf5(osc_prob_maps, args.outfile, settings)
        logging.debug('Wrote %.2f MBytes to %s'%
                     (os.path.getsize(args.outfile)/(1024.**2),
                      os.path.basename(args.outfile)))
