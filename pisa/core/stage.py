# Authors

import collections
import inspect

from pisa.core.map import MapSet
from pisa.core.param import ParamSet
from pisa.core.transform import TransformSet
from pisa.utils.cache import MemoryCache
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile

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
    input_names
    output_names
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
    input_names : list of strings
    output_names : list of strings
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
        Combines source_code_hash and params.hash for checking/tagging
        provenance of persisted (on-disk) objects.
    transforms : TransformSet
        A stage that takes to-be-transformed inputs and has had these
        transforms computed stores these here. Before computation, `transforms`
        is an empty TransformSet. A stage that does not make use of these (such
        as a no-input stage) has an empty TransformSet.
    transforms_cache : None or MemoryCache

    Methods
    -------
    get_outputs
    get_transforms
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
    The following methods can be overridden in derived classes where
    applicable:
        _derive_nominal_transforms_hash
        _compute_nominal_transforms
            This is called during initialization to compute what are termed
            "nominal" transforms -- i.e, transforms with all systematic
            parameters set to their nominal values, such that they have no
            effect on the transform. It is optional to use this stage, but if
            it *is* used, then the result will be cached to memory (and
            optionally to disk cache, if one is provided) for future use. A
            nominal transform is useful when systematic parameters merely have
            the effect of modifying the nominal transform, rather than
            requiring a complete recomputation of the transform.
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
                 params=None, expected_params=None, input_names=None,
                 output_names=None, disk_cache=None,
                 memcaching_enabled=True, propagate_errors=True,
                 transforms_cache_depth=10,
                 outputs_cache_depth=10, input_binning=None,
                 output_binning=None):
        self.use_transforms = use_transforms
        self.stage_name = stage_name
        self.service_name = service_name
        self.expected_params = expected_params
        self._input_names = [] if input_names is None else input_names
        self._output_names = [] if output_names is None else output_names
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
        self.nominal_transforms_cache = MemoryCache(10, is_lru=True)

        self.outputs_cache_depth = outputs_cache_depth
        self.outputs_cache = MemoryCache(self.outputs_cache_depth, is_lru=True)
        self.disk_cache = disk_cache
        self.params = params

    def get_nominal_transforms(self):
        """Load a cached transform from the nominal transform memory cache
        (which is backed by a disk cache, if one is specified) if the nominal
        transform is in the cache, or else recompute it and store to the
        cache(s).

        This method calls the `_compute_nominal_transforms` method, which by
        default does nothing.

        However, if you want to use the nominal transforms feature, override
        the `_compute_nominal_transforms` method and fill in the logic there.

        Deciding whether to invoke the `_compute_nominal_transforms` method or
        to load the nominal transforms from cache is done here, so you needn't
        think about any of this within the `_compute_nominal_transforms`
        method.

        """
        nom_hash = self._derive_nominal_transforms_hash()
        recompute = True
        if nom_hash is None:
            return

        if nom_hash in self.nominal_transforms_cache:
            self.nominal_transforms = \
                    self.nominal_transforms_cache[nom_hash]
            recompute = False
        elif self.disk_cache is not None:
            try:
                self.nominal_transforms = self.disk_cache[nom_hash]
            except KeyError:
                pass
            else:
                recompute = False
                self.nominal_transforms_cache[nom_hash] = \
                        self.nominal_transforms

        if not recompute:
            return self.nominal_transforms

        self.nominal_transforms = self._compute_nominal_transforms()
        if self.nominal_transforms is None:
            return

        self.nominal_transforms.hash = nom_hash
        if nom_hash is not None:
            self.nominal_transforms_cache[nom_hash] = self.nominal_transforms
            if self.disk_cache is not None:
                self.disk_cache[nom_hash] = self.nominal_transforms

        return self.nominal_transforms

    def get_transforms(self):
        """Load a cached transform (keyed on hash of parameter values) if it
        is in the cache, or else compute a new transform from currently-set
        parameter values and store this new transform to the cache.

        This calls the private method _compute_transforms (which must be
        implemented in subclasses if the nominal transform feature is desired)
        to generate a new transform if the nominal transform is not found in
        the nominal transform cache.

        Notes
        -----
        The hash used here is only meant to be valid within the scope of a
        session; a hash on the full parameter set used to generate the
        transform *and* the version of the generating software is required for
        non-volatile storage.

        """
        # Compute nominal transforms; if feature is not used, this doesn't
        # actually do much of anything. To do more than this, override the
        # `_compute_nominal_transforms` method.
        self.get_nominal_transforms()

        # Generate hash from param values
        xforms_hash = self._derive_transforms_hash()
        logging.trace('xforms_hash: %s' %xforms_hash)

        # Load and return existing transforms if in the cache
        if self.memcaching_enabled and self.transforms_cache is not None \
                and xforms_hash in self.transforms_cache:
            logging.trace('loading xforms from cache.')
            transforms = self.transforms_cache[xforms_hash]

        # Otherwise: compute transforms, set hash, and store to cache
        else:
            logging.trace('computing transforms.')
            transforms = self._compute_transforms()
            transforms.hash = xforms_hash
            if self.memcaching_enabled and self.transforms_cache is not None:
                self.transforms_cache[xforms_hash] = transforms

        self.check_transforms(transforms)
        self.transforms = transforms
        return transforms

    @profile
    def get_outputs(self, inputs=None):
        """Top-level function for computing outputs. Use this method to get
        outputs if you live outside this stage/service.

        Caching is handled here, so if the output hash returned by
        `_derive_outputs_hash` is in `outputs_cache`, it is simply returned.
        Otherwise, the `_compute_outputs` private method is invoked to do the
        actual work of computing outputs.

        Parameters
        ----------
        inputs : None or Mapping
            Any inputs to be transformed, plus any sideband objects that are to
            be passed on (untransformed) to subsequent stages.

        See also
        --------
        Overloadable methods called directly from this:
            _derive_outputs_hash
            _compute_outputs

        """
        # Keep inputs for internal use and for inspection later
        self.inputs = {} if inputs is None else inputs

        outputs_hash = self._derive_outputs_hash()

        logging.trace('outputs_hash: %s' %outputs_hash)

        if self.memcaching_enabled and outputs_hash is not None and \
                outputs_hash in self.outputs_cache:
            logging.trace('Loading outputs from cache.')
            outputs = self.outputs_cache[outputs_hash]
        else:
            logging.trace('Need to compute outputs...')

            if self.use_transforms:
                self.get_transforms()

            logging.trace('... now computing outputs.')
            outputs = self._compute_outputs(inputs=self.inputs)
            outputs.hash = outputs_hash
            self.check_outputs(outputs)

            # Store output to cache
            if self.memcaching_enabled and outputs_hash is not None:
                self.outputs_cache[outputs_hash] = outputs

        # Keep outputs for inspection later
        self.outputs = outputs

        # TODO: make an intermediate outputs / final outputs distinction, to
        # keep e.g. maps before being summed up?

        # TODO: make generic container object that can be operated on
        # similar to MapSet (but that doesn't require Map as children)

        # Attach sideband objects (i.e., inputs not specified in
        # `self.input_names`) to the "augmented" output object
        names_in_inputs = set([i.name for i in self.inputs])
        unused_input_names = names_in_inputs.difference(self.input_names)

        if len(unused_input_names) == 0:
            return outputs

        # Create a new output container different from `outputs` but copying
        # the contents, for purposes of attaching the sideband objects found.
        augmented_outputs = MapSet(outputs)
        [augmented_outputs.append(inputs[name]) for name in unused_input_names]

        return augmented_outputs

    def check_params(self, params):
        """ Make sure that `expected_params` is defined and that exactly the
        params specified in self.expected_params are present """
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

    def check_transforms(self, transforms):
        """Check that transforms' inputs and outputs match those specified
        for this service.

        Parameters
        ----------
        transforms

        Raises
        ------
        ValueError if transforms' inputs/outputs don't match stage spec

        """
        assert set(transforms.input_names) == set(self.input_names), \
                "Transforms' inputs: " + str(transforms.input_names) + \
                "\nStage inputs: " + str(self.input_names)

        assert set(transforms.output_names) == set(self.output_names), \
                "Transforms' outputs: " + str(transforms.output_names) + \
                "\nStage outputs: " + str(self.output_names)

    def check_outputs(self, outputs):
        assert set(outputs.names) == set(self.output_names), \
                "Outputs: " + str(outputs.names) + \
                "\nStage outputs: " + str(self.output_names)

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
    def input_names(self):
        return tuple(self._input_names)

    @property
    def output_names(self):
        return tuple(self._output_names)

    @property
    def source_code_hash(self):
        """Returns a hash for the source code of this object's class.

        Not meant to be perfect, but should suffice for tracking provenance of
        an object stored to disk that were produced by a Stage.
        """
        if self._source_code_hash is None:
            self._source_code_hash = hash_obj(inspect.getsource(self.__class__))
        return self._source_code_hash

    @property
    def state_hash(self):
        return hash_obj((self.source_code_hash, self.params.state_hash))

    def _derive_nominal_transforms_hash(self):
        """Derive a hash to uniquely identify the nominal transform. This
        should be unique across processes and invocations bacuase the nominal
        transforms can be non-volatile (cached to disk) and must still be
        valid given their hash value upon loading from disk in the future.

        This implementation uses the nominal parameter values' hash
        combined with the source code hash to generate the final nominal
        transforms hash.

        Notes
        -----
        The hashing scheme implemented here might be sufficiently unique for
        many cases, but override this method in services according to the
        following guidelines:

        * Stages that use a nominal transform should override this method if
          the hash is more accurately computed differently from here.

        * Stages that use transforms but do not use nominal transforms can
          override this method with a simpler version that simply returns None
          to save computation time (if this method is found to be a significant
          performance hit). (This method is called each time an output
          is computed if `self.use_transforms == True`.)

        * Stages that use no transforms (i.e., `self.use_transforms == False`)
          will not call any built-in methods related to transforms, so
          overriding this method is irrelevant to such stages.

        If this method *is* overridden (and not just to return None), since the
        nominal transform may be stored to a disk cache, make sure that
        `self.source_code_hash` is included in the objects used to compute the
        final hash value. Even if all parameters are the same, a nominal
        transform stored to disk is ***invalid if the source code changes***,
        and `_derive_nominal_transforms_hash` must reflect this.

        """
        return hash_obj((self.params.nominal_values, self.source_code_hash))

    def _derive_transforms_hash(self):
        """Compute a hash that uniquely identifies the transforms that will be
        produced from the current configuration. Note that this hash needs only
        to be valid for this run (i.e., it is a volatile hash).

        This implementation returns a hash from the current parameters' values.

        """
        id_objects = []
        # Hash on input and/or output binning, if it is specified
        if self.input_binning is not None:
            id_objects.append(self.input_binning.hash)
        if self.output_binning is not None:
            id_objects.append(self.output_binning.hash)

        id_objects.append(self.params.values_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if any([(h == None) for h in id_objects]):
            transforms_hash = None
        else:
            transforms_hash = hash_obj(tuple(id_objects))

        return transforms_hash

    def _derive_outputs_hash(self):
        """Derive a hash value that unique identifies the outputs that will be
        generated based upon the current state of the stage.

        This implementation hashes together:
        * Input and output binning objects' hash values (if either input or
          output binning is not None)
        * Current params' values hash
        * Hashes from any input objects with names in `self.input_names`

        If any of the above objects is specified but returns None for its hash
        value, the entire output hash is invalidated, and None is returned.

        """
        id_objects = []

        # If stage uses inputs, grab hash from the inputs container object
        if len(self.input_names) > 0:
            logging.trace('inputs.hash = %s' %self.inputs.hash)
            id_objects.append(self.inputs.hash)

        # If stage uses transforms, get hash from the transforms
        if self.use_transforms:
            id_objects.append(self._derive_transforms_hash())
            logging.trace('derived transforms hash = %s' %id_objects[-1])

        # Otherwise, generate sub-hash on binning and param values here
        else:
            id_subobjects = []
            # Include binning hashes, if one or both are specified
            if self.input_binning is not None:
                id_subobjects.append(self.input_binning.hash)
            if self.output_binning is not None:
                id_subobjects.append(self.output_binning.hash)

            # Include all parameter values
            id_subobjects.append(self.params.values_hash)
            if any([(h == None) for h in id_subobjects]):
                sub_hash = None
            else:
                sub_hash = hash_obj(tuple(id_subobjects))
            id_objects.append(sub_hash)

        # If any hashes are missing (i.e, None), invalidate the entire hash
        if any([(h == None) for h in id_objects]):
            outputs_hash = None
        else:
            outputs_hash = hash_obj(tuple(id_objects))

        return outputs_hash

    def _compute_nominal_transforms(self):
        """Stages that start with a nominal transform and use systematic
        parameters to modify the nominal transform in order to obtain the final
        transforms should override this method for deriving the nominal
        transform."""
        return None

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

    def validate_params(self, params):
        """Override this method to test if params are valid; e.g., check range
        and dimensionality."""
        pass
