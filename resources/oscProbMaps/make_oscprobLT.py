#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   18 March 2014
#

from argparse import ArgumentParser

parser = ArgumentParser(description='Makes a full oscillation probability map, for atmospheric neutrinos (nue, numu, nue_bar, numu_bar) propagating through the earth. The binning in energy and coszenith is very fine, so that it can be averaged over later to produce the smoothed maps. Finally, the result is outputted as an hdf5 file.')
######## Physics Parameters ########
parser.add_argument("deltam31",action='store',type=float,
                    help='oscillation parameter \DeltaM_{31}^2. Use negative sign for IMH. [eV^2]')
parser.add_argument("theta23",action='store',type=float,
                    help='oscillation parameter \theta_{23} [deg]')

parser.add_argument("--deltam21",action='store',type=float,default=7.54e-5,
                    help='oscillation parameter \DeltaM_{21}^2. [eV^2]')
parser.add_argument("--theta12",action='store',type=float,default=33.647,
                    help='oscillation parameter \theta_{12} [deg]')
parser.add_argument("--theta13",action='store',type=float,default=8.931,
                    help='oscillation parameter \theta_{13} [deg]')
parser.add_argument("--deltacp",action='store',type=float,default=0.0,
                    help='deltaCP, CP violating phase parameter [-2 pi, 2 pi].')
parser.add_argument("--earth_model",action='store',type=str,default="prem",
                    help="nuCraft earth model to use.")
######## Binning Parameters ########
parser.add_argument("--num_ebins",action='store',type=int,default=800,
                    help="Number of energy bin points to sample.")
parser.add_argument("--emin",action="store",type=float,default=1.0,
                    help="minimum energy to use [GeV]")
parser.add_argument("--emax",action="store",type=float,default=80.0,
                    help="maximum energy to use [GeV]")
parser.add_argument("--logE",action="store_true",default=False,
                    help="Use a log10 energy scale.")
parser.add_argument("--num_czbins",action='store',type=int,default=800,
                    help="Number of cz bin points to sample [-1 to 1].")
######## Output Files ########
parser.add_argument("-o", "--outdir", dest="outdir", metavar="DIR",
            type=str, action="store", default=None,
            help="directory to store the output files.")
parser.add_argument("-v", "--verbose", action="count", default=0,
            help="set verbosity level")
args = parser.parse_args()

def DefineBinning(args):
    ebins = np.logspace(log10(args.emin),log10(args.emax),args.num_ebins+1) if args.logE else np.linspace(args.emin,args.emax,args.num_ebins+1)
    czbins = np.linspace(-1,1,args.num_czbins+1)

    return czbins, ebins

def CreateFileName(args):
    # NOTE: This is a temporary solution to creating a filename until we come up with 
    # a naming standard/dictionary lookup standard.
    return "oscProbLT_dm31_"+str(args.deltam31*100.0)+"_th23_"+str(args.theta23)+".hdf5"

import logging
from OscProbMaps import OscProbMaps
import numpy as np
from math import log10

levels = {0:logging.ERROR,
          1:logging.INFO,
          2:logging.DEBUG}
logging.basicConfig(format='[%(levelname)8s] %(message)s')
logging.root.setLevel(levels[min(2,args.verbose)])

czbins,ebins = DefineBinning(args)

osc_prob_map = OscProbMaps(czbins, ebins, args.deltam31, args.theta23, 
                           deltam21=args.deltam21,theta12=args.theta12,
                           theta13=args.theta13, deltacp=args.deltacp, 
                           earth_model=args.earth_model)

from datetime import datetime
time1 = datetime.now()
oscprob_dict = osc_prob_map.GetOscProbLT()
print "Time to create the dictionary: ",(datetime.now() - time1)

filename = CreateFileName(args)
osc_prob_map.SaveHDF5(filename,oscprob_dict)
