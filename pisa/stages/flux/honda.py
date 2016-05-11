#
# honda.py
#
# This flux service provides flux values from tables provided by Honda. It can
# be for a grid of energy / cos(zenith) (for maps) or just singular values
# (for events). This is achieved through b-spline interpolation, done in both
# energy and cosZenith dimensions simultaneously in log10(flux).
#
# Most of the functionality will be ported from PISA with any necessary
# changes/improvements applied.
#
# PISA authors: Sebastian Boeser
#               sboeser@physik.uni-bonn.de
#               Steven Wren
#               steven.wren@icecube.wisc.edu
#
# CAKE author: Steven Wren
#              steven.wren@icecube.wisc.edu
#
# date:   2016-05-11


import numpy as np
import pint
ureg = pint.UnitRegistry()

from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.resources.resources import open_resource
from pisa.utils.hash import hash_obj


class honda(Stage):
    """
    This is a Flux Service for performing interpolation on the Honda tables

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet
        Parameters with which to instantiate the class.
        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <Param object or passable to Param()>}.
    output_binning : The binning desired for the output maps
    disk_cache : None, str, or DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object. Must be concurrent-access-safe (across threads and processes).
    memcaching_enabled
    outputs_cache_depth
    """
    def __init__(self, params, output_binning, disk_cache=None,
                 memcaching_enabled=True, propagate_errors=True,
                 outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'atm_delta_index', 'energy_scale', 'nu_nubar_ratio',
            'nue_numu_ratio', 'test', 'example_file', 'oversample_e',
            'oversample_cz', 'flux_file', 'flux_mode'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Invoke the init method from the parent class, which does a lot of
        # work for you. Note that we do not specify `input_names` here, since
        # there are no "inputs" used by this stage. (Of course there are
        # parameters, and files with info, but no maps or MC events are used
        # and transformed directly by this stage to produce its output.)
        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='flux',
            service_name='honda',
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            disk_cache=disk_cache,
            memcaching_enabled=memcaching_enabled,
            propagate_errors=propagate_errors,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )

        # Initialisation of this service should load the flux tables
        # Also, the splining should only be done once, so do that too
        self.load_table(flux_mode, smooth=0.05)

    def load_table(self, smooth=0.05):
        flux_file = self.params['flux_file'].value
        logging.info("Loading atmospheric flux table %s" % flux_file)

        # Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        # columns in Honda files are in the same order
        cols = ['energy'] + self.primaries

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.iterkeys():
            # There are 20 lines per zenith range
            flux_dict[key] = np.array(np.split(flux_dict[key], 20))
            if not key=='energy':
                flux_dict[key] = flux_dict[key].T

        # Set the zenith and energy range
        flux_dict['energy'] = flux_dict['energy'][0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)

        # Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')

        flux_mode = self.params['flux_mode'].value

        if flux_mode == 'bisplrep':

            logging.debug('Doing quick bivariate spline interpolation')
            # do this in log of energy and log of flux (more stable)
            logE, C = np.meshgrid(np.log10(flux_dict['energy']),
                                  flux_dict['coszen'])

            self.spline_dict = {}
            for nutype in self.primaries:
                # Get the logarithmic flux
                log_flux = np.log10(flux_dict[nutype]).T
                # Get a spline representation
                spline =  bisplrep(logE, C, log_flux, s=smooth)
                # and store
                self.spline_dict[nutype] = spline

        elif flux_mode == 'integral-preserving':

            logging.debug('Doing this integral-preserving. Will take longer')

            self.spline_dict = {}

            # Do integral-preserving method as in IceCube's NuFlux
            # This one will be based purely on SciPy rather than ROOT
            # Stored splines will be 1D in integrated flux over energy
            int_flux_dict = {}
            # Energy and CosZenith bins needed for integral-preserving
            # method must be the edges of those of the normal tables
            int_flux_dict['logenergy'] = np.linspace(-1.025,4.025,102)
            int_flux_dict['coszen'] = np.linspace(-1,1,21)
            for nutype in primaries:
                # spline_dict now wants to be a set of splines for
                # every table cosZenith value.
                splines = {}
                CZiter = 1
                for energyfluxlist in flux_dict[nutype]:
                    int_flux = []
                    tot_flux = 0.0
                    int_flux.append(tot_flux)
                    for energyfluxval, energyval in zip(energyfluxlist, flux_dict['energy']):
                        # Spline works best if you integrate flux * energy
                        tot_flux += energyfluxval*energyval
                        int_flux.append(tot_flux)

                    spline = splrep(int_flux_dict['logenergy'],int_flux,s=0)
                    CZvalue = '%.2f'%(1.05-CZiter*0.1)
                    splines[CZvalue] = spline
                    CZiter += 1
                    
                self.spline_dict[nutype] = splines

    def _compute_outputs(self, inputs=None):
        # Following is just so that we only produce new maps when params
        # change, but produce the same maps with the same param values
        # (for a more realistic test of caching).
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        # Convert a parameter that the user can specify in any (compatible)
        # units to the units used for compuation
        height = self.params['test'].value.to('meter').magnitude

        output_maps = []
        for output_name in self.output_names:
            # Generate the fake per-bin "fluxes", modified by the parameter
            hist = np.ones(self.output_binning.shape) * height

            # Put the "fluxes" into a Map object, give it the output_name
            m = Map(name=output_name, hist=hist, binning=self.output_binning)

            # Optionally turn on errors here, that will be propagated through
            # rest of pipeline (slows things down, but essential in some cases)
            #m.set_poisson_errors()
            output_maps.append(m)

        # Combine the output maps into a single MapSet object to return.
        # The MapSet contains the varous things that are necessary to make
        # caching work and also provides a nice interface for the user to all
        # of the contained maps
        return MapSet(maps=output_maps, name='flux maps')

    def validate_params(self, params):
        # do some checks on the parameters
        assert (params['flux_mode'].value == 'integral-preserving' or params['flux_mode'].value == 'bisplrep')
        assert ('honda' in params['flux_file'].value)
