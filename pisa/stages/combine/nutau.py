# authors: P.Eller (pde3@psu.edu)
# date:   September 2016

import sys, os
import numpy as np
import time

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.core.events import Events
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.map import Map, MapSet
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj

class nutau(Stage):
    '''
    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params are:

            nu_nc_norm : quantity (dimensionless)
            nutau_cc_norm : quantity (dimensionless)
    '''

    def __init__(self, params, input_binning, output_binning, disk_cache=None,
                 memcache_deepcopy=True, error_method=None,
                 outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'nu_nc_norm',
            #'nutau_cc_norm',
            )

        input_names = ('nue_cc+nuebar_cc', 'numu_cc+numubar_cc', 'nutau_cc+nutaubar_cc', 'nuall_nc+nuallbar_nc')
        output_names = ('nu')

        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            input_binning=input_binning,
            debug_mode=debug_mode
        )

    def _compute_transforms(self):
    
        dims = self.input_binning.names
        xform_shape = [len(self.input_names)] + [self.input_binning[d].num_bins for d in dims]

        # TODO: populate explicitly by flavor, don't assume any particular
        # ordering of the outputs names!
        transforms = []
        for output_name in self.output_names:
            xform = np.ones(xform_shape)
            input_names = [n for n in self.input_names if output_name in n]
            for i,name in enumerate(input_names):
                #if 'nutau' in name:
                #    xform[i] *= self.params.nutau_cc_norm.value.m_as('dimensionless')
                if '_nc' in name:
                    xform[i] *= self.params.nu_nc_norm.value.m_as('dimensionless')

            transforms.append(
                BinnedTensorTransform(
                    input_names=input_names,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )

        return TransformSet(transforms=transforms)
