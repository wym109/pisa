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
    def __init__(self, params, input_binning, output_binning,
                 disk_cache=None, error_method=None, input_names=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        """TODO: documentme"""

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'dom_eff', 'dom_eff_file',
            'hole_ice_fwd', 'hole_ice_fwd_file',
            'hole_ice', 'hole_ice_file',
            #'reco_cz_res', 'reco_cz_res_file',
        )

        if isinstance(input_names, basestring):
            input_names = input_names.replace(' ', '').split(',')
        elif input_names is None:
             input_names = ('trck','cscd')
        output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            error_method=error_method,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

    def _compute_nominal_transforms(self):
        """TODO: documentme"""
        self.pnames = [pname for pname in self.params.names if not
            pname.endswith('_file')]
        self.fit_results = {}
        for pname in self.pnames:
            self.fit_results[pname] = from_file(self.params[pname+'_file'].value)
            assert self.input_names == self.fit_results[pname]['map_names']

    @profile
    def _compute_transforms(self):
        """TODO: documentme"""
        # TODO: use iterators to collapse nested loops
        transforms = []
        for name in self.input_names:
            transform = None
            for pname in self.pnames:
                p_value = (self.params[pname].magnitude -
                           self.fit_results[pname]['nominal'])
                exec(self.fit_results[pname]['function'])
                fit_params = self.fit_results[pname][name]
                small_shape = fit_params.shape[:-1]
                if transform is None:
                    transform = np.ones(small_shape)
                for idx in np.ndindex(*small_shape):
                    transform[idx] *= fit_fun(p_value,
                            *fit_params[idx])

            xform = BinnedTensorTransform(
                input_names=(name),
                output_name=name,
                input_binning=self.input_binning,
                output_binning=self.output_binning,
                xform_array=transform,
                error_method=self.error_method,
            )
            transforms.append(xform)
        return TransformSet(transforms)
