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
    fnames =  os.listdir(args.dir)
    

for filename in fnames:
    if filename.endswith('.json'):
        file = from_file(args.dir+'/'+filename)
        name,_ = filename.split('.json')
        #assert(not file[0][0]['warnflag'][0] and not file[0][1]['warnflag']), name
        if file[0][0].has_key('llh'): metric = 'llh'
        elif file[0][0].has_key('conv_llh'): metric = 'conv_llh'
        elif file[0][0].has_key('chi2'): metric = 'chi2'
        elif file[0][0].has_key('mod_chi2'): metric = 'mod_chi2'
        else: continue

        cond_llh = file[0][0][metric][0]
        glob_llh = file[0][1][metric]
        if 'chi2' in metric:
            signif = np.sqrt(cond_llh - glob_llh)
        else:
            signif = np.sqrt(2*(cond_llh - glob_llh))
        res[name] = signif
        #print '%s\t%.4f'%(name,signif)

if res.has_key('nominal'):
    nominal = res['nominal']
    print '%i systematics'%(len(res)-1)
    print '%-20s\tsign\tdelta\tpercent'%'sys'
else:
    nominal = None
for key, val in sorted(res.items(),key=lambda s: s[1], reverse=True):
    if nominal is not None:
        print '%-20s\t%.3f\t%.4f\t%.2f %%'%(key,val,val-nominal,(val-nominal)/nominal*100)
    else:
        print '%s\t%.4f sigma'%(key,val)
