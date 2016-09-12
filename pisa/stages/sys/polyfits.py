# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016


import copy
from itertools import product

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.events import Events
from pisa.utils.flavInt import flavintGroupsFromString
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile
from pisa.utils.fileio import from_file


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class polyfits(Stage):
    """TODO: documentme"""
    def __init__(self, params, input_binning, output_binning,
                 disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'dom_eff', 'dom_eff_file',
            'hole_ice_fwd', 'hole_ice_fwd_file',
            'hole_ice', 'hole_ice_file',
            'reco_cz_res', 'reco_cz_res_file',
        )

        # Define the names of objects expected in inputs and produced as
        # outputs
        #input_names = ('nue_cc_trck','nue_cc_cscd','nue_nc_trck','nue_nc_cscd',
        #               'nuebar_cc_trck','nuebar_cc_cscd','nuebar_nc_trck','nuebar_nc_cscd',
        #               'numu_cc_trck','numu_cc_cscd','numu_nc_trck','numu_nc_cscd',
        #               'numubar_cc_trck','numubar_cc_cscd','numubar_nc_trck','numubar_nc_cscd',
        #               'nutau_cc_trck','nutau_cc_cscd','nutau_nc_trck','nutau_nc_cscd',
        #               'nutaubar_cc_trck','nutaubar_cc_cscd','nutaubar_nc_trck','nutaubar_nc_cscd'
        #)
        #input_names = ('nue_cc_trck','nue_cc_cscd','nuall_nc_trck','nuall_nc_cscd',
        #               'nuebar_cc_trck','nuebar_cc_cscd',
        #               'numu_cc_trck','numu_cc_cscd',
        #               'numubar_cc_trck','numubar_cc_cscd','nuallbar_nc_trck','nuallbar_nc_cscd',
        #               'nutau_cc_trck','nutau_cc_cscd',
        #               'nutaubar_cc_trck','nutaubar_cc_cscd',
        #)
        input_names = ('cscd', 'trck')
        output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='sys',
            service_name='polyfits',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

    def _compute_nominal_transforms(self):
        """TODO: documentme"""
        self.pnames = [pname for pname in self.params.names if not
            pname.endswith('_file')]
        self.fit_results = {}
        self.categories = None
        for pname in self.pnames:
            self.fit_results[pname] = from_file(self.params[pname+'_file'].value)
            #assert (input_binning == self.fit_results[pname]['binning']),\
            #    'incompatible binning between fit results and provided maps'
            if self.categories is None:
                self.categories = self.fit_results[pname]['categories']
            else:
                assert (self.categories == self.fit_results[pname]['categories']), 'use of different categories not supported'

    @profile
    def _compute_transforms(self):
        """TODO: documentme"""
        # TODO: use iterators to collapse nested loops
        transforms = []
        for cat in self.categories:
            transform = None
            for name in self.input_names:
                if name.endswith(cat):
                    if transform is None:
                        for pname in self.pnames:
                            p_value = (self.params[pname].magnitude -
                                       self.fit_results[pname]['nominal'])
                            exec(self.fit_results[pname]['function'])
                            fit_params  = self.fit_results[pname][cat]
                            nx, ny, _ = fit_params.shape
                            if transform is None:
                                transform = np.ones((nx, ny))
                            for i, j in np.ndindex((nx,ny)):
                                transform[i,j] *= fit_fun(p_value,
                                        *fit_params[i,j,:])

                    xform = BinnedTensorTransform(
                        input_names=(name),
                        output_name=name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=transform
                    )
                    transforms.append(xform)
        return TransformSet(transforms)
