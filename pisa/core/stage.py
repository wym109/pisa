import collections
import inspect

from pisa.core.transform import TransformSet
from pisa.core.param import ParamSet
from pisa.core.map import MapSet
from pisa.utils.cache import MemoryCache
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity

# TODO: mode for not propagating errors. Probably needs hooks here, but meat of
# implementation would live inside map.py and/or transform.py.

# TODO: make service_name dynamically found from class name, rather than as arg

class Stage(object):
    """
    PISA stage base class. Should encompass all behaviors common to (almost)
    all stages.

    Specialization should be done via subclasses.

    Parameters
    ----------
    use_transforms : bool (required)
        Whether or not this stage takes inputs to be transformed (and hence
        implements transforms).
    stage_name : string
    service_name : string
    params : ParamSet or sequence with which to instantiate a ParamSet
        Parameters with which to instantiate the class.
        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <Param object or passable to Param()>}.
    expected_params : list of strings
        List containing required `params` names.
    disk_cache : None, str, or DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object. Must be concurrent-access-safe (across threads and processes).
    memcaching_enabled
    outputs_cache_depth
    transforms_cache_depth
    input_binning : None or interpretable as MultiDimBinning
    output_binning : None or interpretable as MultiDimBinning

    Properties
    ----------
    disk_cache
    expected_params : list of strings
        List containing required param names.
    use_transforms : bool
        Whether or not this stage takes inputs to be transformed (and hence
        implements transforms).
    memcaching_enabled
    params : ParamSet
        All stage parameters, returned in alphabetical order by param name.
    outputs : None or Mapping
        Last-computed outputs, and None if no outputs have been computed yet.
    outputs_cache : None or MemoryCache
        Cache for storing the outputs of the stage, but *without* sideband
        objects.
    service_name : str
        Name of the service, e.g. 'AeffServiceSliceSmooth'
    source_code_hash
        Hash for the class's source code.
    stage_name : str
        Name of the stage (e.g., 'flux', 'osc', 'aeff', 'reco', 'pid', ...)
    state_hash
        Combines source_code_hash and params_hash for checking/tagging
        provenance of persisted (on-disk) objects.
    transforms : TransformSet
        A stage that takes to-be-transformed inputs and has had these
        transforms computed stores these here. Before computation, `transforms`
        is an empty TransformSet. A stage that does not make use of these (such
        as a no-input stage) has an empty TransformSet.
    transforms_cache : None or MemoryCache

    Methods
    -------
    load_params
        Load parameter values from a template settings ini file.
    fix_params
        Remove param(s) with specified name(s) from those able to be varied by
        minimizer or auto-scanner methods. Values can still be modified
        manually by setting the param.value attribute.
    unfix_params
        Free param(s) with specified name(s) to be varied by minimizer or
        auto-scanner methods.

    Override
    --------
    The following methods should be overridden in derived classes, if
    applicable:
        _compute_transforms
            Do the actual work to produce the stage's transforms. For stages
            that specify use_transforms=False, this method is never called.
        _compute_outputs
            Do the actual work to compute the stage's output. Default
            implementation is to call self.transforms.apply(inputs); override
            if no transforms are present or if more needs to be done to
            compute outputs than this.
        validate_params
            Perform validation on any parameters.
    """
    def __init__(self, use_transforms, stage_name='', service_name='',
                 params=None, expected_params=None, disk_cache=None,
                 memcaching_enabled=True, propagate_errors=True,
                 transforms_cache_depth=10,
                 outputs_cache_depth=10, input_binning=None,
                 output_binning=None):
        self.use_transforms = use_transforms
        self.stage_name = stage_name
        self.service_name = service_name
        self.expected_params = expected_params
        self.input_binning = input_binning
        self.output_binning = output_binning
        self._source_code_hash = None

        # Storage of latest transforms and outputs; default to empty
        # TransformSet and None, respectively.
        self.transforms = TransformSet([])
        self.outputs = None

        self.memcaching_enabled = memcaching_enabled

        self.transforms_cache_depth = transforms_cache_depth
        self.transforms_cache = MemoryCache(self.transforms_cache_depth,
                                            is_lru=True)
        self.outputs_cache_depth = outputs_cache_depth
        self.outputs_cache = MemoryCache(self.outputs_cache_depth, is_lru=True)
        self.disk_cache = disk_cache
        self.params = params

    def compute_transforms(self):
        """Load a cached transform (keyed on hash of parameter values) if it
        is in the cache, or else compute a new transform from currently-set
        parameter values and store this new transform to the cache.

        This calls the private method _compute_transforms, which must be
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
        xforms_hash = self.params.values_hash
        logging.trace('xforms_hash: %s' %xforms_hash)

        # Load and return existing transforms if in the cache
        if self.memcaching_enabled and self.transforms_cache is not None \
                and xforms_hash in self.transforms_cache:
            logging.trace('loading xforms from cache.')
            self.transforms = self.transforms_cache[xforms_hash]
            return self.transforms

        # Otherwise: compute transforms anew, set hash value, and store to
        # cache
        logging.trace('computing transforms.')
        xforms = self._compute_transforms()
        xforms.hash = xforms_hash
        self.transforms = xforms
        if self.memcaching_enabled and self.transforms_cache is not None:
            self.transforms_cache[xforms_hash] = self.transforms
        return self.transforms

    def compute_outputs(self, inputs=None):
        """Compute and return outputs.

        Parameters
        ----------
        inputs : None or Mapping
            Any inputs to be transformed, plus any sideband objects that are to
            be passed on (untransformed) to subsequent stages.

        """
        if self.use_transforms:
            self.compute_transforms()

        id_objects = []
        if self.input_names is not None:
            [id_objects.append(inputs[name].hash) for name in self.input_names]
        # TODO: include binning hash(es) in id_objects
        id_objects.append(self.params.values_hash)
        outputs_hash = hash_obj(id_objects)
        logging.trace('outputs_hash: %s' %outputs_hash)

        if self.memcaching_enabled and outputs_hash in self.outputs_cache:
            logging.trace('loading outputs from cache')
            outputs = self.outputs_cache[outputs_hash]
        else:
            logging.trace('deriving outputs')
            outputs = self._compute_outputs(inputs)
            if self.memcaching_enabled:
                outputs.hash = outputs_hash
                self.outputs_cache[outputs_hash] = outputs

        # Keep outputs also as property of object for later inspection
        self.outputs = outputs

        # TODO: make generic container object that can be operated on
        # similar to MapSet (but that doesn't require Map as children)

        # Create a new output container different from `outputs` but copying
        # the contents, for purposes of attaching sideband objects. (These
        # should not be stored in the cache.)
        augmented_outputs = MapSet(outputs)

        if inputs is None:
            inputs = []

        # Attach sideband objects (unused inputs) to output...
        # 1. Start with all names in inputs.
        unused_input_names = set([i.name for i in inputs])
        # 2. Remove names that were used by a transform.
        if self.transforms is not None:
            [unused_input_names.difference_update(xform.input_names)
             for xform in self.transforms]
        # 3. Append these unused objects to the output.
        for name in unused_input_names:
            augmented_outputs.append(inputs[name])

        return augmented_outputs

    def check_params(self, params):
        """ make sure that `expected_params` is defined and that exactly the
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
        raise ValueError('Expected parameters: %s;\n'
                         %', '.join(sorted(exp_p))
                         + ';\n'.join(err_strs))

    def validate_params(self, params):
        """ method to test if params are valid, e.g. check range and
        dimenionality """
        pass

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        if not isinstance(p, ParamSet):
            raise TypeError('Unhandled `params` type "%s"; expected ParmSet' %
                            type(p))
        self.check_params(p)
        self.validate_params(p)
        self._params = p

    @property
    def source_code_hash(self):
        """Returns a hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(inspect.getsource(self))
        return self._source_code_hash

    @property
    def state_hash(self):
        return hash_obj((self.source_code_hash, self.params.state_hash))

    @property
    def input_names(self):
        return self.transforms.input_names

    def _compute_transforms(self):
        """Stages that apply transforms to inputs should override this method
        for deriving the transform. No-input stages should leave this as-is,
        simply returning None."""
        return TransformSet([])

    def _compute_outputs(self, inputs):
        """Override this method for no-input stages which do not use transforms.
        Input stages that compute a TransformSet needn't override this, as the
        work for computing outputs is done by the TransfromSet below."""
        return self.transforms.apply(inputs)
