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
        get_outputs
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

    def get_outputs(self):
        raise NotImplementedError()

    def check_params(self, params):
        """ make sure that expected_params is defined and that exactly the
        params specifued in self.expected_params are present """
        assert self.expected_params is not None
        exp_p, got_p = set(self.expected_params), set(params.names)
        if exp_p == got_p:
            return
        excess = got_p.difference(exp_p)
        missing = exp_p.difference(got_p)
        err_strs = []
        if len(excess) > 0:
            err_strs.append('Excess params provided: %s'
                            %', '.join(sorted(excess)))
        if len(missing) > 0:
            err_strs.append('Missing params: %s'
                            %', '.join(sorted(missing)))
        raise ValueError('Expected parameters: %s;\n' %', '.join(sorted(exp_p))
                         + ';\n'.join(err_strs))

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

    def get_outputs(self):
        """Compute output maps."""
        result_hash = self.params.values_hash
        try:
            return self.result_cache[result_hash]
        except KeyError:
            pass
        outputs = self._derive_output()
        outputs.hash = result_hash
        self.result_cache[result_hash] = outputs
        return outputs

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

    def get_outputs(self, inputs):
        """Compute output maps by applying the transforms to input maps.

        Parameters
        ----------
        inputs : MapSet

        Notes
        -----
        Caching is handled within the TransformSet and each Transform, so as
        to allow "sideband" objects to be passed through to the output; these
        are objects upon which the parameters and transforms have no effect,
        and so should not be cached alongside the computed outputs (but should
        be passed along through the stage).

        """
        xforms = self.get_transforms(cache=self.transform_cache)
        outputs = xforms.apply(inputs, cache=self.result_cache)
        return outputs

    def get_transforms(self, cache=None):
        """Load a cached transform (keyed on hash of parameter values) if it
        is in the cache, or else derive a new transform from currently-set
        parameter values and store this new transform to the cache.

        This calls the private method _derive_transforms, which must be
        implemented in subclasses, to generate a new transform if none is in
        the cache already.

        Notes
        -----
        The hash used here is only meant to be valid within the scope of a
        session; a hash on the full parameter set used to generate the
        transform *and* the version of the generating software is required for
        non-volatile storage.

        """
        # Generate hash from param values
        pval_hash = self.params.values_hash
        print 'pnames:', self.params.names
        print 'pvals:', [v.m for v in self.params.values]
        print 'pval_hash:', pval_hash

        # Load and return existing transforms if in the cache
        if cache is not None and pval_hash in cache:
            return self.transform_cache[pval_hash]

        # Otherwise: compute transforms anew, set hash value, and store to
        # cache
        xforms = self._derive_transforms()
        xforms.hash = pval_hash
        if cache is not None:
            cache[pval_hash] = xforms

        return xforms

    def _derive_transforms(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()
