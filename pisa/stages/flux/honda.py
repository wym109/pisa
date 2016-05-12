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

        # Set the neutrio primaries
        self.primaries = ['nue', 'numu', 'nue_bar', 'numu_bar']

        # Initialisation of this service should load the flux tables
        # Can work with either 2D (E,Z,AA) or 3D (E,Z,A) tables.
        # Also, the splining should only be done once, so do that too
        if len(output_binning.names) == 3:
            self.load_3D_table(smooth=0.05)
        else:
            self.load_2D_table(smooth=0.05)
        
    def load_2D_table(self, smooth=0.05):
        '''
        Method for manipulating 2 dimensional flux tables.
        2D is expected to mean energy and cosZenith.
        Azimuth is averaged over before being stored in the table.
        The zenith range should be both hemispheres.
        '''
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

        # Set the zenith and energy range as they are in the tables
        # The energy may change, but the zenith should always be
        # 20 bins, full sky.
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

    def load_3D_table(self, smooth=0.05):
        '''
        Method for manipulating 3 dimensional flux tables.
        3D is expected to mean energy, cosZenith, and azimuth.
        The angles coverage should be full sky.
        '''
        flux_file = self.params['flux_file'].value
        logging.info("Loading atmospheric flux table %s" %flux_file)

        #Load the data table
        table = np.loadtxt(open_resource(flux_file)).T

        #columns in Honda files are in the same order
        cols = ['energy']+primaries

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.iterkeys():

            #There are 20 lines per zenith range
            coszenith_lists = np.array(np.split(flux_dict[key], 20))
            azimuth_lists = []
            for coszenith_list in coszenith_lists:
                azimuth_lists.append(np.array(np.split(coszenith_list,12)).T)
            flux_dict[key] = np.array(azimuth_lists)
            if not key=='energy':
                flux_dict[key] = flux_dict[key].T

        #Set the zenith and energy range
        flux_dict['energy'] = flux_dict['energy'][0].T[0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)
        flux_dict['azimuth'] = np.linspace(15,345,12)

        #Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')

        flux_mode = self.params['flux_mode'].value

        if flux_mode == 'bisplrep':

            logging.debug('Doing quick bsplrep spline interpolation in 3D')
            # do this in log of energy and log of flux (more stable)
            logE, C = np.meshgrid(np.log10(flux_dict['energy']), flux_dict['coszen'])

            self.spline_dict = {}
        
            for nutype in primaries:
                self.spline_dict[nutype] = {}
                #Make 1 2D bsplrep in E,CZ for each azimuth value
                for az, f in zip(flux_dict['azimuth'],flux_dict[nutype]):
                    #Get the logarithmic flux
                    log_flux = np.log10(f.T)
                    #Get a spline representation
                    spline =  bisplrep(logE, C, log_flux, s=smooth*4.)
                    #Found smoothing has to be weaker here. Not sure why.
                    #and store
                    self.spline_dict[nutype][az] = spline

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
                # In 3D mode we have a set of these sets for every
                # table azimuth value.
                az_splines = {}
                for az, f in zip(flux_dict['azimuth'],flux_dict[nutype]):
                    splines = {}
                    CZiter = 1
                    for energyfluxlist in f.T:
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

                    az_splines[az] = splines
                    
                self.spline_dict[nutype] = az_splines

    def _compute_outputs(self, inputs=None):
        # Following is just so that we only produce new maps when params
        # change, but produce the same maps with the same param values
        # (for a more realistic test of caching).
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        if len(self.output_binning.names) == 3:
            output_maps = self.compute_3D_outputs()
        else:
            output_maps = self.compute_2D_outputs()

        # Combine the output maps into a single MapSet object to return.
        # The MapSet contains the varous things that are necessary to make
        # caching work and also provides a nice interface for the user to all
        # of the contained maps
        return MapSet(maps=output_maps, name='flux maps')

    def _compute_2D_outputs(self):
        '''
        Method for computing 2 dimensional fluxes. Binning always expected 
        in energy and cosZenith. Splines are manipulated based on whether
        they were set up as bisplrep or integral-preserving.
        '''

        cache_key = hash_obj((ebins, czbins, prim))
        try:
            return self.raw_flux_cache[cache_key]
        except KeyError:
            pass

        ebins = self.output_binning['ebins'].bin_edges
        evals = ebins.weighted_centers
        ebin_sizes = ebins.bin_sizes
        
        czbins = self.output_binning['czbins'].bin_edges
        czvals = czbins.weighted_centers
        czbin_sizes = czbins.bin_sizes

        flux_mode = self.params['flux_mode'].value

        if flux_mode == 'bisplrep':

            # Assert that spline dict matches what is expected
            # i.e. One spline for each primary
            assert(type(self.spline_dict[self.primaries[0]]) != 'dict')

            # Get the spline interpolation, which is in
            # log(flux) as function of log(E), cos(zenith)
            return_table = bisplev(np.log10(evals), czvals, self.spline_dict[prim])
            return_table = np.power(10., return_table).T

        elif flux_mode == 'integral-preserving':

            # Assert that spline dict matches what is expected
            # i.e. One dict of splines for every table cosZenith value
            #      0.95 is used for no particular reason
            assert(type(self.spline_dict[self.primaries[0]]['0.95']) != 'dict')

            return_table = []

            for energyval in evals:
                logenergyval = np.log10(energyval)
                spline_vals = []
                for czkey in np.linspace(-0.95,0.95,20):
                    # Have to multiply by bin widths to get correct derivatives
                    # Here the bin width is in log energy, is 0.05
                    spline_vals.append(splev(logenergyval,self.spline_dict[prim]['%.2f'%czkey],der=1)*0.05)
                int_spline_vals = []
                tot_val = 0.0
                int_spline_vals.append(tot_val)
                for val in spline_vals:
                    tot_val += val
                    int_spline_vals.append(tot_val)

                spline = splrep(np.linspace(-1,1,21),int_spline_vals,s=0)
                
                # Have to multiply by bin widths to get correct derivatives
                # Here the bin width is in cosZenith, is 0.1
                czfluxes = splev(czvals,spline,der=1)*0.1/energyval
                return_table.append(czfluxes)

            return_table = np.array(return_table).T

        # Flux is given per sr and GeV, so we need to multiply
        # by bin width in both dimensions
        # Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)

        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])
        return_table = return_table.T

        self.raw_flux_cache[cache_key] = return_table
        
        return return_table

    def _compute_3D_outputs(self):
        '''
        Method for computing 3 dimensional fluxes when binning is also 
        called in azimuth. Binning always expected in energy and
        cosZenith, and the previous compute_2D_outputs is called in
        that case instead.

        For now this just mimics the dummy functionality. Will add it in
        properly later
        '''

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
        return output_maps

    def validate_params(self, params):
        # do some checks on the parameters
        assert (params['flux_mode'].value == 'integral-preserving' or params['flux_mode'].value == 'bisplrep')
        assert ('honda' in params['flux_file'].value)
        assert ('aa' in params['flux_file'].value if len(self.output_binning.names) == 2)
        assert ('aa' not in params['flux_file'].value if len(self.output_binning.names) == 3)

        
