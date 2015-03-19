#! /usr/bin/env python
#
# This is a quick script that takes the reconstruction
# parameterizations in the PaPA setitngs file format and convert it to
# the pisa reco settings format.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.jsons import from_json,to_json
from copy import deepcopy as copy

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('papa_file',type=str,metavar='JSON',
                    help='papa settings file, containing the resolution parameterizations.')
parser.add_argument('outfile',type=str,metavar='JSON',
                    help='output file to store resolutions in PISA-format.')
args = parser.parse_args()

papa_settings = from_json(args.papa_file)
parameterizations = papa_settings['fiducial']['reco_parametrization']['value']

pisa_flavs = ['nue','numu','nutau']
mID = ['','_bar']
intType = ['cc','nc']
recoType = ['coszen','energy']

papa_NC = parameterizations['NC']
egy_res = papa_NC['e']
papa_NC.pop('e')
papa_NC['energy'] = egy_res

pisa_reco_settings = {}
for flav in pisa_flavs:
    papa_res = parameterizations[flav]
    egy_res = papa_res['e']
    papa_res.pop('e')
    papa_res['energy'] = egy_res

    for ID in mID:
        pisa_reco_settings[flav+ID] = {'cc':{},'nc':{}}
        pisa_reco_settings[flav+ID]['cc'] = copy(papa_res)
        pisa_reco_settings[flav+ID]['nc'] = copy(papa_NC)

to_json(pisa_reco_settings,args.outfile)


# Now, read in the file as if it were a text file and replace n. with np.
fh = open(args.outfile,'r')
output_lines = []
for line in fh.readlines():
    output_lines.append(line.replace("n.","np."))
fh.close()

fh = open(args.outfile,'w')
for line in output_lines:
    fh.write(line)
fh.close()
