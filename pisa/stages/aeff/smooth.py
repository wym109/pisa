# authors: J.Lanfranchi/P.Eller/M.Weiss
# date:   March 2, 2017
"""
Smoothed-histogram effective areas stage
"""

import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
from scipy.interpolate import bisplrep, bisplev, RectBivariateSpline
from scipy import ndimage

from pisa import ureg
from pisa.core.map import Map
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.log import logging
from pisa.utils.plotter import Plotter
from pisa.utils.profiler import profile
from pisa.utils.spline_smooth import spline_smooth
from pisa.utils.flavInt import NuFlavInt

__all__ = ['smooth_simple']

# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

# TODO: remove the input_names instantiation arg since these are computed
# from the `particles` arg?

class smooth(Stage):
    """Smooth each effective area transform by fitting splines

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        aeff_events
        livetime
        aeff_scale
        aeff_e_smooth_factor
        aeff_cz_smooth_factor
        nutau_cc_norm
        transform_events_keep_criteria

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : None, string or sequence of strings
        If None, defaults are derived from `particles`.

    error_method

    transform_groups : string
        Specifies which particles/interaction types to use for computing the
        transforms. (See Notes.)

    sum_grouped_flavints : bool
        Whether to sum the event-rate maps for the flavint groupings
        specified by `transform_groups`. If this is done, the output map names
        will be the group names (as well as the names of any flavor/interaction
        types not grouped together). Otherwise, the output map names will be
        the same as the input map names. Combining grouped flavints' is
        computationally faster and results in fewer maps, but it may be
        desirable to not do so for, e.g., debugging.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    spline_binning:  MultiDimBinning
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `input_binning`.
        This binning is used to construct a histogram to be splined. The binning
        chosen here is typically more coarse, in order to get higher precision
        points from which the splines are constructed. The splines are evaluated
        however using the input_binning
        

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    Notes
    -----
    See Conventions section in the documentation for more informaton on
    particle naming scheme in PISA.

    """
    def __init__(self, params, particles, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning, spline_binning,
                 memcache_deepcopy, transforms_cache_depth,
                 outputs_cache_depth, input_names=None, error_method=None,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        self.sum_grouped_flavints = sum_grouped_flavints

        self.spline_binning = spline_binning

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = [
            'aeff_events', 'livetime', 'aeff_scale',
            'aeff_e_smooth_factor', 'aeff_cz_smooth_factor',
            'transform_events_keep_criteria',
        ]
        if particles == 'neutrinos':
            expected_params.append('nutau_cc_norm')

        if isinstance(input_names, basestring):
            input_names = input_names.replace(' ', '').split(',')
        elif input_names is None:
            if particles == 'neutrinos':
                input_names = ('nue', 'nuebar', 'numu', 'numubar', 'nutau',
                               'nutaubar')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                input_flavints = NuFlavIntGroup(input_names)
                output_names = [str(fi) for fi in input_flavints]
        elif self.particles == 'muons':
            raise NotImplementedError
        else:
            raise ValueError('Particle type `%s` is not valid' % self.particles)

        logging.trace('transform_groups = %s' %self.transform_groups)
        logging.trace('output_names = %s' %' :: '.join(output_names))

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )
        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')


    def slice_smooth(self, xform, xform_errors, spline_binning, smooth_factor):
        # Dimensions and order used for computation
        dim1 = 'true_energy'
        dim2= 'true_coszen'
        xform, errors = spline_smooth(xform,
                                spline_binning[dim1],
                                self.input_binning[dim1],
                                self.input_binning.index(dim1),
                                0.1,
                                3,
                                errors=xform_errors)
        xform, _ = spline_smooth(xform, 
                                spline_binning[dim2], 
                                self.input_binning[dim2], 
                                self.input_binning.index(dim2), 
                                0.2,
                                3,
                                errors=None)

        # clip unphysical (negative)  values
        return xform.clip(0)

    def bi_smooth(self, xform, xform_errors, spline_binning, smooth_factor):
        #x, y = np.meshgrid(spline_binning.dimensions[0].midpoints, spline_binning.dimensions[1].midpoints)
        #xb = spline_binning.dimensions[0].bin_edges[0].m
        #xe = spline_binning.dimensions[0].bin_edges[-1].m
        #yb = spline_binning.dimensions[1].bin_edges[0].m
        #ye = spline_binning.dimensions[1].bin_edges[-1].m
        z = xform
        #w = 1./xform_errors
        #w = np.ones(len(x.ravel()))
        #s = len(x.ravel())
        #spline = bisplrep(x.ravel(), y.ravel(), z.ravel(), w.ravel(), s=s, xb=xb, yb=yb, xe=xe, ye=ye)
        spline = RectBivariateSpline(spline_binning.dimensions[0].midpoints, spline_binning.dimensions[1].midpoints,z,s=smooth_factor)
        x_eval, y_eval = np.meshgrid(self.input_binning.dimensions[0].midpoints.m, self.input_binning.dimensions[1].midpoints.m)
        #smoothed = bisplev(x_eval.ravel(), y_eval.ravel(), spline)
        smoothed = spline(self.input_binning.dimensions[0].midpoints.m, self.input_binning.dimensions[1].midpoints.m)
        #smoothed = smoothed.reshape(z.shape)
        return smoothed.clip(0)


    @profile
    def _compute_nominal_transforms(self):
        self.load_events(self.params.aeff_events)
        self.cut_events(self.params.transform_events_keep_criteria)

        # Units must be the following for correctly converting a sum-of-
        # OneWeights-in-bin to an average effective area across the bin.
        comp_units = dict(true_energy='GeV', true_coszen=None,
                          true_azimuth='rad')

        # Only works if energy is in input_binning
        if 'true_energy' not in self.input_binning:
            raise ValueError('Input binning must contain "true_energy"'
                             ' dimension, but does not.')

        # coszen and azimuth are both optional, but no further dimensions are
        excess_dims = set(self.input_binning.names).difference(
            comp_units.keys()
        )
        if len(excess_dims) > 0:
            raise ValueError('Input binning has extra dimension(s): %s'
                             %sorted(excess_dims))

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        # Account for "missing" dimension(s) (dimensions OneWeight expects for
        # computation of bin volume), and accommodate with a factor equal to
        # the full range. See IceCube wiki/documentation for OneWeight for
        # more info.
        missing_dims_vol = 1
        if 'true_azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        if 'true_coszen' not in input_binning:
            missing_dims_vol *= 2

        # Make binning for smoothing
        # TODO Add support for azimuth
        assert 'true_coszen' in input_binning.names
        assert 'true_energy' in input_binning.names
        assert len(input_binning.names) == 2

        transforms = []

        for xform_flavints in self.transform_groups:
            logging.info("Working on %s effective areas xform" %xform_flavints)

            #rel_error = 10.
            spline_binning = self.spline_binning
            raw_hist = self.remaining_events.histogram(
                kinds=xform_flavints,
                binning=spline_binning,
                weights_col='weighted_aeff',
                errors=True
            )
            raw_transform = unp.nominal_values(raw_hist.hist)
            raw_errors = unp.std_devs(raw_hist.hist)



            #while rel_error > 0.15:
            #    raw_hist = self.remaining_events.histogram(
            #        kinds=xform_flavints,
            #        binning=spline_binning,
            #        weights_col='weighted_aeff',
            #        errors=True
            #    )
            #    raw_transform = unp.nominal_values(raw_hist.hist)
            #    raw_errors = unp.std_devs(raw_hist.hist)
            #    rel_error = raw_errors/raw_transform
            #    rel_error = np.median(rel_error[raw_transform != 0])
            #    e_idx = spline_binning.index('true_energy')
            #    if rel_error > 0.15:
            #        factors = [1]*len(spline_binning.dimensions)
            #        factors[e_idx] = 2
            #        logging.warning('Downsampling spline energy binning by factor of 2 for better statistics! Now at %.2f'%rel_error)
            #        try:
            #            spline_binning = spline_binning.downsample(*factors)
            #        except:
            #            logging.error('Not enough statistics in %s, maybe try grouping flavours or change spline binning'%xform_flavints)
            #            break

            #magic_factor = 2./rel_error
            #magic_factor = spline_binning.dimensions[e_idx].num_bins

            # Divide histogram by
            #   (energy bin width x coszen bin width x azimuth bin width)
            # volumes to convert from sums-of-OneWeights-in-bins to
            # effective areas. Note that volume correction factor for
            # missing dimensions is applied here.
            bin_volumes = spline_binning.bin_volumes(attach_units=False)
            raw_transform /= (bin_volumes * missing_dims_vol)
            # for bins woth zero events, we get an eror of zero
            # but these zeros do hae some stat. imprecision
            # here assume this is equal to the median non-zero error
            #raw_errors[raw_errors == 0] = np.min(raw_errors[raw_errors != 0])
            raw_errors /= (bin_volumes * missing_dims_vol)


            # what's the stat. situation here?
            rel_error = raw_errors/raw_transform
            rel_error = np.median(rel_error[raw_transform != 0])
            print rel_error

            # now use gaussian smoothing on those, muahaha
            e_idx = spline_binning.index('true_energy')
            if e_idx == 0:
                sigma_cz = raw_transform.shape[1]*0.1
                sigma_e = raw_transform.shape[0]*0.025
                sigma1 = (0,sigma_cz)
                sigma2 = (sigma_e,0)
            else:
                sigma_cz = raw_transform.shape[0]*0.1
                sigma_e = raw_transform.shape[1]*0.025
                sigma1 = (sigma_cz,0)
                sigma2 = (0,sigma_e)
            mode = 'reflect'
            raw_sumw2 = np.square(raw_errors)
            smooth_transform = ndimage.filters.gaussian_filter(raw_transform, sigma1, mode='reflect')
            smooth_sumw2 = ndimage.filters.gaussian_filter(raw_sumw2, sigma1, mode='reflect')
            smooth_transform = ndimage.filters.gaussian_filter(smooth_transform, sigma2, mode='nearest', truncate=1.)
            smooth_sumw2 = ndimage.filters.gaussian_filter(smooth_sumw2, sigma2, mode='nearest', truncate=1.)
            smooth_errors = np.sqrt(smooth_sumw2)

            # smooth and eval at binning
            smooth_transform = self.slice_smooth(smooth_transform, smooth_errors, spline_binning, 0.2)
            
            #smooth_transform = self.bi_smooth(raw_transform, raw_errors, spline_binning, magic_factor)

            # for CC tau interaction, regions < 3.5 GeV are below threshold, therefore set to zero:
            if abs(xform_flavints[0].flavCode()) == 16 and xform_flavints[0].isCC():
                logging.info('Cutting off effective area below 3.5 GeV for %s'%xform_flavints[0])
                for i,energy in enumerate(input_binning['true_energy'].bin_edges[1:]):
                    if energy.magnitude < 3.5:
                        if input_binning.index('true_energy') == 0:
                            smooth_transform[i,:] *= 0
                        else:
                            smooth_transform[:,i] *= 0
                 

            # For each member of the group, save the raw aeff transform and
            # its smoothed and interpolated versions
            flav_names = [str(flav) for flav in xform_flavints.flavs()]

            # If combining grouped flavints:
            # Create a single transform for each group and assign all flavors
            # that contribute to the group as the transform's inputs. Combining
            # the event rate maps will be performed by the
            # BinnedTensorTransform object upon invocation of the `apply`
            # method.
            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    if len(set(xform_flavints).intersection(input_flavs)) > 0:
                        xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if output_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=output_name,
                        input_binning=input_binning,
                        output_binning=input_binning,
                        xform_array=smooth_transform,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    transforms.append(xform)

            # If *not* combining grouped flavints:
            # Copy the transform for each input flavor, regardless if the
            # transform is computed from a combination of flavors.
            else:
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    # Since aeff "splits" neutrino flavors into
                    # flavor+interaction types, need to check if the output
                    # flavints are encapsulated by the input flavor(s).
                    if len(set(xform_flavints).intersection(input_flavs)) == 0:
                        continue
                    for output_name in self.output_names:
                        if output_name not in xform_flavints:
                            continue
                        xform = BinnedTensorTransform(
                            input_names=input_name,
                            output_name=output_name,
                            input_binning=nput_binning,
                            output_binning=input_binning,
                            xform_array=smooth_transform,
                        )
                        transforms.append(xform)

        return TransformSet(transforms=transforms)

    @profile
    def _compute_transforms(self):
        """Compute new effective areas transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        livetime_s = self.params.livetime.value.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        if self.particles == 'neutrinos':
            nutau_cc_norm = self.params.nutau_cc_norm.m_as('dimensionless')
            if nutau_cc_norm != 1:
                assert NuFlavIntGroup('nutau_cc') in self.transform_groups
                assert NuFlavIntGroup('nutaubar_cc') in self.transform_groups

        new_transforms = []
        for xform_flavints in self.transform_groups:
            repr_flav_int = xform_flavints[0]
            flav_names = [str(flav) for flav in xform_flavints.flavs()]
            raw_transform = None
            for transform in self.nominal_transforms:
                if (transform.input_names[0] in flav_names
                        and transform.output_name in xform_flavints):
                    if raw_transform is None:
                        scale = aeff_scale * livetime_s
                        if (self.particles == 'neutrinos' and
                                ('nutau_cc' in transform.output_name
                                 or 'nutaubar_cc' in transform.output_name)):
                            scale *= nutau_cc_norm
                        raw_transform = transform.xform_array * scale

                    new_xform = BinnedTensorTransform(
                        input_names=transform.input_names,
                        output_name=transform.output_name,
                        input_binning=transform.input_binning,
                        output_binning=transform.output_binning,
                        xform_array=raw_transform,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    new_transforms.append(new_xform)

        return TransformSet(new_transforms)
