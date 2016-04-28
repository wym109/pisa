import collections
import inspect

from pisa.utils import cache
from pisa.utils.hash import hash_obj
from pisa.core.param import ParamSet


class GenericStage(object):
    """
    PISA stage generic class. Should encompass all behaviors common to all
    stages.

    Specialization should be done via subclasses.

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet
        Parameters with which to instantiate the class.
        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <param_val>}.

    disk_cache : None, str, or utils.DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object.

    Properties
    ----------
    stage_name : str
        Name of the stage (e.g., 'flux', 'osc', 'aeff', 'reco', 'pid', ...)

    service_name : str
        Name of the service, e.g. 'AeffServiceSliceSmooth'

    params : ParamSet or sequence with which to instantiate a ParamSet
        All stage parameters, returned in alphabetical order by param name.
        The format of the returned dict is
            {'<param_name_0>': <param_val_0>, ...,
             '<param_name_N>': <param_val_N>}

    expected_params: list of strings
        list containing param names used in that stage

    source_code_hash
        Hash for the class's source code.

    state_hash
        Combines source_code_hash and params_hash for checking/tagging
        provenance of persisted (on-disk) objects.

    disk_cache
       Concurrent-access-safe disk-caching object instantiated if a valid
       `disk_cache` resource is specified by user. This implements get() and
       set() methods for loading/storing an object from/to the disk cache.

    Methods
    -------
    load_params
        Load parameter values from a template settings ini file.

    fix_params
        Remove param(s) with specified name(s) from those able to be varied by
        minimizer or auto-scanner methods.

    unfix_params
        Free param(s) with specified name(s) to be varied by minimizer or
        auto-scanner methods.

    Override
    --------
    The following methods should be overridden in derived classes
        get_output_map_set
            Do the actual work to produce the stage's output.
        validate_params
            Perform validation on any parameters.
    """
    def __init__(self, stage_name='', service_name='', params=None,
            expected_params=None, disk_cache=None):
        self.stage_name = stage_name
        self.service_name = service_name
        self.expected_params = expected_params
        self.__params = None
        self.__free_param_names = set()
        self.__source_code_hash = None

        self.disk_cache = disk_cache
        if params is not None:
            self.params = params

    def get_output_map_set(self):
        raise NotImplementedError()

    def check_params(self, params):
        """ make sure that expected_params is defined and that exactly the params specifued in
        self.expected_params are present """
        assert self.expected_params is not None
        try:
            assert sorted(self.expected_params) == list(params.names) 
        except AssertionError:
            raise Exception('Expected parameters %s while getting\
                    %s'%(self.expected_params, list(params.names)))

    def validate_params(self, params):
        """ method to test if params are valid, e.g. check range and
        dimenionality """
        pass

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, p):
        if not isinstance(p, ParamSet):
            raise TypeError('Unhandled `params` type "%s"; expected ParmSet' %
                            type(p))
        self.check_params(p)
        self.validate_params(p)
        self.__params = p

    @property
    def source_code_hash(self):
        """Returns a hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self.__source_code_hash is None:
            self.__source_code_hash = hash(inspect.getsource(self))
        return self.__source_code_hash

    @property
    def state_hash(self):
        return hash_obj((self.source_code_hash, self.params.state_hash))


class NoInputStage(GenericStage):
    def __init__(self, stage_name='', service_name='', params=None,
                expected_params=None, disk_cache=None,
                cache_class=cache.MemoryCache, result_cache_depth=10):
        super(NoInputStage, self).__init__(stage_name=stage_name,
                service_name=service_name,
                params=params, expected_params=expected_params,
                disk_cache=disk_cache)
        self.result_cache_depth = result_cache_depth
        self.result_cache = cache_class(self.result_cache_depth, is_lru=True)

    def get_output_map_set(self):
        """Compute output maps."""
        result_hash = self.params.values_hash
        try:
            return self.result_cache[result_hash]
        except KeyError:
            pass
        output_map_set = self._derive_output()
        output_map_set.hash = result_hash
        self.result_cache[result_hash] = output_map_set
        return output_map_set

    def _derive_output(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()


class InputStage(GenericStage):
    def __init__(self, stage_name='', service_name='', params=None,
                 expected_params=None, disk_cache=None,
                 cache_class=cache.MemoryCache, transform_cache_depth=10,
                 result_cache_depth=10):
        super(InputStage, self).__init__(stage_name=stage_name,
                service_name=service_name,
                params=params, expected_params=expected_params,
                disk_cache=disk_cache)
        self.transform_cache_depth = transform_cache_depth
        self.transform_cache = cache_class(self.transform_cache_depth,
                                           is_lru=True)

        self.result_cache_depth = result_cache_depth
        self.result_cache = cache_class(self.result_cache_depth, is_lru=True)

    def get_output_map_set(self, input_map_set):
        """Compute output maps by applying the transform to input maps.

        Parameters
        ----------
        input_map_set : MapSet

        """
        xform = self.get_transform()
        result_hash = hash_obj(input_map_set.hash + xform.hash)
        try:
            return self.result_cache[result_hash]
        except KeyError:
            pass
        output_map_set = xform.apply(input_map_set)
        output_map_set.hash = result_hash
        self.result_cache[result_hash] = output_map_set
        return output_map_set

    def get_transform(self):
        """Load a cached transform (keyed on hash of free parameters) if it is
        in the cache, or else derive a new transform from currently-set
        parameter values and store this new transform to the cache.

        Notes
        -----
        The hash used here is only meant to be valid within the scope of a
        session; a hash on the full parameter set used to generate the
        transform, as well as the version of the generating software, is
        required for non-volatile storage.
        """
        xform_hash = self.params.values_hash
        try:
            return self.transform_cache[xform_hash]
        except KeyError:
            pass
        xform = self._derive_transform()
        xform.hash = xform_hash
        #self.transform_cache[cache_key] = xform
        return xform

    def _derive_transform(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()
