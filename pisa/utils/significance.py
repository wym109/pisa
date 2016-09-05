#! /usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from pisa.utils.fileio import from_file


parser = ArgumentParser()
parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
parser.add_argument('-f','--file',metavar='dir',help='single json files', default='') 
args = parser.parse_args()
res = {}


if args.file is not '':
    fnames = [args.file]
else:
    fnames = [os.path.join(args.dir, f) for f in os.listdir(args.dir)]
    

for filename in fnames:
    if filename.endswith('.json'):
        file = from_file(args.dir+'/'+filename)
        name,_ = filename.split('.')
        assert(not file[0][0]['warnflag'][0] and not file[0][1]['warnflag'])
        cond_llh = file[0][0]['llh'][0]
        glob_llh = file[0][1]['llh']
        signif = np.sqrt(2*(cond_llh - glob_llh))
        res[name] = signif
        #print '%s\t%.4f'%(name,signif)

if res.has_key('nominal'):
    nominal = res['nominal']
    print 'sys\tdelta\tpercent'
else:
    nominal = None
for key, val in res.items():
    if nominal is not None:
        print '%s\t%.4f\t%.2f %%'%(key,val-nominal,(val-nominal)/nominal*100)
    else:
        print '%s\t%.4f sigma'%(key,val)
