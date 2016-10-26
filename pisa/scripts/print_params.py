from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from pisa import ureg, Q_
from pisa.utils.log import set_verbosity
from pisa.utils.fileio import from_file


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
    		    metavar='fit file', default=None,
    		    help='settings for the generation of "data"')
    args = parser.parse_args()
    
    params = from_file(args.file)
    for key,val in params[0].items():
        try:
            if len(val[1]) == 0:
                q = '%.2e'%val[0]
            else:
                if val[1][0][1] == 1.:
                    q = '%.2e %s'%(val[0],val[1][0][0])
                else:
                    q = '%.2e %s^%s'%(val[0],val[1][0][0],val[1][0][1])
        except TypeError:
            q = '%.2e'%val
        print '%s\t%s'%(key,q)
