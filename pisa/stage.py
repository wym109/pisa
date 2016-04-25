
import collections
import inspect

from pisa.utils import utils
from pisa.utils import cache
from pisa.utils.utils import hash_obj


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
        Name of the stage ('flux', 'osc', 'aeff', 'reco', or 'pid')

    service_name : str
        Name of the service, e.g. 'AeffServiceMC'

    params : ParamSet or sequence with which to instantiate a ParamSet
        All stage parameters, returned in alphabetical order by param name.
        The format of the returned dict is
            {'<param_name_0>': <param_val_0>, ...,
             '<param_name_N>': <param_val_N>}

    free_params : OrderedDict
        Those `params` that are not fixed, returned in same format as `params`
        and in alphabetical order by param name.

    priors_llh : float
        Summed log likelihoods of all parameters' priors

    priors_chisquare : float
        Summed chi-squared values of all parameters' priors

    params_hash, free_params_hash
        Hashes for all params (`params_hash`) or just free params
        (`free_params_hash`). The former is most thorough, while the latter is
        provided for faster operation and suffices for identifying temporary
        (in-memory) references to objects.

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
    The following methods should be overriden in derived classes
        get_output_map_set
            Do the actual work to produce the stage's output.
        validate_params
            Perform validation on any parameters.
        add_cmdline_args
            The stage base class should override this to add generic
            stage args applicable across all implementations of the stage;
            further, each service (implementation of a stage) should add args
            specific to it here as well (after calling the stage base class).
    """
    def __init__(self, stage_name='', service_name='', params=None,
                 disk_cache=None):
        self.stage_name = stage_name
        self.service_name = service_name
        self.__params = dict()
        self.__free_param_names = set()

        self.__params_hash = None
        self.__free_params_hash = None
        self.__source_code_hash = None

        self.disk_cache = disk_cache
        if params is not None:
            self.params = params

    def get_output_map_set(self):
        raise NotImplementedError()

    def validate_params(self, params):
        raise NotImplementedError()

    @staticmethod
    def add_cmdline_args(parser):
        raise NotImplementedError()

    @property
    def params(self):
        ordered = collections.OrederedDict()
        [ordered[k].__setitem__(self.__params[k])
         for k in sorted(self.__params)]
        return ordered

    @params.setter
    def params(self, p):
        if isinstance(p, basestring):
            self.load_params(p)
            return
        elif isinstance(p, collections.Mapping):
            pass
        else:
            raise TypeError('Unhandled `params` type "%s"' % type(p))
        self.validate_params(p)
        self.__params.update(p)

        # Invalidate hashes so they get recomputed next time they're requested
        self.__params_hash = None
        self.__free_params_hash = None

    #def load_params(self, resource):
    #    # TODO: load from ini file format
    #    if isinstance(resource, basestring):
    #        params_dict = fileio.from_file(resource)
    #    elif isinstance(collections.Mapping):
    #        params_dict = resource
    #    else:
    #        raise TypeError('Unhandled `rsource` type "%s"' % type(resource))
    #    self.params = params_dict

    #@property
    #def free_params(self):
    #    ordered = collections.OrederedDict()
    #    [ordered[k].__setitem__(self.__params[k])
    #     for k in sorted(self.__free_param_names)]
    #    return ordered

    #@property
    #def num_params(self):
    #    return len(self.__params)

    #@property
    #def num_free_params(self):
    #    return len(self.__free_param_names)

    #@free_params.setter
    #def free_params(self, p):
    #    if isinstance(p, (collections.Iterable, collections.Sequence)):
    #        p = {pname: p[n]
    #             for n, pname in enumerate(sorted(self.__free_param_names))}
    #        assert len(p) == len(self.__free_param_names)

    #    if not isinstance(p, collections.Mapping):
    #        raise TypeError('Unhandled `params` type "%s"' % type(p))

    #    assert set(p.keys()).issubset(self.__free_param_names)
    #    self.validate(p)
    #    self.__params.update(p)

    #    # Invalidate hashes so they get recomputed next time they're requested
    #    self.__params_hash = None
    #    self.__free_params_hash = None

    #def fix_params(self, params, ignore_missing=False):
    #    if np.isscalar(params):
    #        params = [params]
    #    if ignore_missing:
    #        [self.free_params.add(p) for p in params if p in self.__params]
    #    else:
    #        assert
    #        self.__free_param_names.difference_update(params)

    #def unfix_params(self, params, ignore_missing=False):
    #    if ignore_missing:
    #        self.__free_param_names.difference_update(params)
    #    else:
    #        [self.__free_param_names.remove(p) for p in params]
    #@property
    #def params_hash(self):
    #    """Returns a hash for *all* parameters. Note that This can be slow!"""
    #    if self.__params_hash is None:
    #        self.__params_hash = utils.hash_obj(self.__params)
    #    return self.__params_hash

    #@property
    #def free_params_hash(self):
    #    """Returns a hash for just free parameters; should be faster than
    #    params_hash."""
    #    if self.__params_hash is None:
    #        self.__params_hash = utils.hash_obj(self.__params)
    #    return self.__params_hash

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
        return hash((self.source_code_hash, self.params.state_hash))



class NoInputStage(GenericStage):
    def __init__(self, params=None, disk_cache=None,
                 cache_class=cache.MemoryCache, result_cache_depth=10):
        super(NoInputStage, self).__init__(params=params,
                                               disk_cache=disk_cache)
        self.result_cache_depth = result_cache_depth
        self.result_cache = cache_class(self.result_cache_depth, is_lru=True)

    def get_output_map_set(self):
        """Compute output maps."""
        result_hash = self.free_params_hash
        try:
            return self.result_cache[result_hash]
        except KeyError:
            pass
        output_map_set = self._derive_output()
        output_map_set.update_hash(result_hash)
        self.result_cache[result_hash] = output_map_set
        return output_map_set

    def _derive_output(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()


class InputStage(GenericStage):
    def __init__(self, params=None, disk_cache=None,
                 cache_class=cache.MemoryCache, transform_cache_depth=10,
                 result_cache_depth=10):
        super(InputStage, self).__init__(params=params,
                                         disk_cache=disk_cache)
        self.transform_cache_depth = cache_depth
        self.transform_cache = cache_class(self.transform_cache_depth,
                                           is_lru=True)

        self.result_cache_depth = result_cache_depth
        self.result_cache = cache_class(self.result_cache_depth, is_lru=True)

    def get_output_map_set(self, input_map_set):
        """Compute output maps by applying the transform to input maps.

        Parameters
        ----------
        input_map_set : utils.DictWithHash

        """
        xform = self.get_transform()
        result_hash = hash_obj(input_map_set.hash, xform.hash)
        try:
            return self.result_cache[result_hash]
        except KeyError:
            pass
        output_map_set = xform.apply(input_map_set)
        output_map_set.update_hash(result_hash)
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
        xform_hash = self.free_params_hash
        try:
            return self.transform_cache[xform_hash]
        except KeyError:
            pass
        xform = self._derive_transform()
        xform.update_hash(xform_hash)
        self.transform_cache[cache_key] = xform
        return xform

    def _derive_transform(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()
