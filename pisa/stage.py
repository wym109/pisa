
import collections
import inspect

from pisa.utils import utils
import pisa.utils.utils.hash_obj as hash_obj


class GenericStage(object):
    """
    PISA stage generic class. Should encompass all behaviors common to all
    stages.

    Specialization should be done via subclasses.

    Parameters
    ----------
    params : None, dict, str
        Parameters with which to instantiate the class.
        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <param_val>}.

    disk_cache_dir : None or str
        Path to a disk cache directory. If None, no disk cache will be
        available.

    Properties
    ----------
    params : OrderedDict
        All stage parameters, returned in alphabetical order by param name.
        The format of the returned dict is
            {'<param_name_0>': <param_val_0>, ...,
             '<param_name_N>': <param_val_N>}

    free_params : OrderedDict
        Those `params` that are not fixed, returned in same format as `params`
        and in alphabetical order by param name.

    params_hash, free_params_hash
        Hashes for all params (`params_hash`) or just free params
        (`free_params_hash`). The former is most thorough, while the latter is
        provided for faster operation and suffices for identifying temporary
        (in-memory) references to objects.

    source_code_hash
        Hash for the class's source code.

    complete_state_hash
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

    Override following methods in derived class(es):
        compute_output_maps
        validate_params
        add_stage_cmdline_args
        add_service_cmdline_args

    """
    def __init__(self, params=None, disk_cache_dir=None):
        self.__params = dict()
        self.__free_param_names = set()

        self.__params_hash = None
        self.__free_params_hash = None
        self.__source_code_hash = None

        self.disk_cache_dir = disk_cache_dir
        if params is not None:
            self.params = params

    def compute_output_maps(self):
        """Override to produce the stage output."""
        raise NotImplementedError()

    def validate_params(self, params):
        """Override to perform validation on any parameters."""
        raise NotImplementedError()

    def _derive_transform(self, **kwargs):
        """Each stage implementation ("service") must override this method"""
        raise NotImplementedError()

    @staticmethod
    def add_service_cmdline_args(parser):
        """Each stage implementation (aka service) should add generic stage
        args as well as args for systematics specific to it here"""
        raise NotImplementedError()

    @staticmethod
    def add_stage_cmdline_args(parser):
        """Each stage's base class should override this. Here, add generic
        stage args -- as well as args for systematics applicable across all
        implementations of the stage -- to `parser`.
        """
        raise NotImplementedError()

    @property
    def params_hash(self):
        """Returns a hash for *all* parameters. Note that This can be slow!"""
        if self.__params_hash is None:
            self.__params_hash = utils.hash_obj(self.__params)
        return self.__params_hash

    @property
    def free_params_hash(self):
        """Returns a hash for just free parameters; should be faster than
        params_hash."""
        if self.__params_hash is None:
            self.__params_hash = utils.hash_obj(self.__params)
        return self.__params_hash

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
    def complete_state_hash(self):
        return hash((self.source_code_hash, self.params_hash))

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

    def load_params(self, resource):
        if isinstance(resource, basestring):
            params_dict = fileio.from_file(resource)
        elif isinstance(collections.Mapping):
            params_dict = resource
        else:
            raise TypeError('Unhandled `rsource` type "%s"' % type(resource))
        self.params = params_dict

    @property
    def free_params(self):
        ordered = collections.OrederedDict()
        [ordered[k].__setitem__(self.__params[k])
         for k in sorted(self.__free_param_names)]
        return ordered

    @free_params.setter
    def free_params(self, p):
        if isinstance(p, (collections.Iterable, collections.Sequence)):
            p = {pname: p[n]
                 for n, pname in enumerate(sorted(self.__free_param_names))}
            assert len(p) == len(self.__free_param_names)

        if not isinstance(p, collections.Mapping):
            raise TypeError('Unhandled `params` type "%s"' % type(p))

        assert set(p.keys()).issubset(self.__free_param_names)
        self.validate(p)
        self.__params.update(p)

        # Invalidate hashes so they get recomputed next time they're requested
        self.__params_hash = None
        self.__free_params_hash = None

    def fix_params(self, params, ignore_missing=False):
        if np.isscalar(params):
            params = [params]
        if ignore_missing:
            [self.free_params.add(p) for p in params if p in self.__params]
        else:
            assert
            self.__free_param_names.difference_update(params)

    def unfix_params(self, params, ignore_missing=False):
        if ignore_missing:
            self.__free_param_names.difference_update(params)
        else:
            [self.__free_param_names.remove(p) for p in params]


class StageWithInput(GenericStage):
    def __init__(self, params=None, disk_cache_dir=None,
                 cache_class=utils.LRUCache, transform_cache_depth=10,
                 result_cache_depth=10):
        super(StageWithInput, self).__init__(params=params,
                                             disk_cache_dir=disk_cache_dir)
        self.transform_cache_depth = cache_depth
        self.transform_cache = cache_class(self.transform_cache_depth)

        self.result_cache_depth = result_cache_depth
        self.result_cache = cache_class(self.result_cache_depth)

    def compute_output_maps(self, input_maps):
        """Compute output maps by applying the transform to input maps.

        Parameters
        ----------
        input_maps : utils.DictWithHash
        """
        xform = self.get_transform()

        result_hash = hash_obj(input_maps.hash, xform.hash)
        try:
            return self.result_cache.get(result_hash)
        except KeyError:
            pass

        output_maps = xform.apply(input_maps)
        output_maps.update_hash(output_maps)

        return output_maps

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
        xform_hash = hash_obj(self.free_params)
        try:
            return self.transform_cache.get(xform_hash)
        except KeyError:
            pass
        xform = self._derive_transform()
        xform.update_hash(xform_hash)
        self.transform_cache.set(cache_key, xform)
        return xform

    def _derive_transform(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()

    @staticmethod
    def add_stage_cmdline_args(parser):
        """Each stage's base class should override this. Here, add generic
        stage args -- as well as args for systematics applicable across all
        implementations of the stage -- to `parser`.
        """
        raise NotImplementedError()

    @staticmethod
    def add_service_cmdline_args(parser):
        """Each stage implementation (aka service) should add generic stage
        args as well as args for systematics specific to it here"""
        raise NotImplementedError()

