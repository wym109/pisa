#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016
from pisa.core.template_maker import TemplateMaker

class Analysis(object):
    def __init__(self, configs):
        self.pipelines = []
        for config in configs:
            template_settings = parse_cfg(config) 
            self.pipelines.append(TemplateMaker(template_settings))

    def scan_llh(self, pname, values):
        llhs = []
        m0 = self.pipelines[0].get_output_map_set()
        for val in values:
            fp = self.pipelines[1].free_params
            fp[pname].value = val
            self.pipelines[1].update_params(fp)
            m1 = self.pipelines[1].get_output_map_set()    
            llhs.append(m0.total_llh(m1))
        return llhs

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    import pint
    ureg = pint.UnitRegistry()
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_cfg import parse_cfg

    parser = ArgumentParser()
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='configfile', required=True,
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="out.json",
                        help='file to store the output')
    args = parser.parse_args()
    
    template_settings0 = from_file(args.template_settings)
    template_settings0.set('stage:flux','param.test.fixed','True')
    template_settings1 = from_file(args.template_settings)

    ana = Analysis([template_settings0, template_settings1])
    print ana.scan_llh('test', np.arange(0,30,1)*ureg.foot)
