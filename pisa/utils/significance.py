#! /usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from pisa.utils.fileio import from_file


parser = ArgumentParser()
parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
args = parser.parse_args()
res = {}
for filename in os.listdir(args.dir):
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
    for key, val in res.items():
        print '%s\t%.4f\t%.2f %%'%(key,val-nominal,(val-nominal)/nominal*100)
