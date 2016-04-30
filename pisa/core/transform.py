
import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.hash import hash_obj

# TODO: use a generic container, *not* a MapSet to store sets of maps for
# inputs and outputs (in fact I think there should be no MapSet object at all,
# so we can trnsparently handle e.g. events alongside maps where one is a
# sideband object for the other in a given stage)

# TODO: Add Sequence capabilities to TransformSet (e.g. it'd be nice to have at least append, extend, ...)
class TransformSet(object):
    """
    Parameters
    ----------
    transforms
    name

    Properties
    ----------
    hash
    input_names
    name
    num_inputs
    num_outputs
    output_names
    transforms

    Methods
    -------
    check_predecessor_compat
    check_successor_compat
    apply

    """
    def __init__(self, transforms, name=None):
        self.transforms = transforms
        self.name = name

    def __iter__(self):
        return iter(self.transforms)

    @property
    def hash(self):
        xform_hashes = [x.hash for x in transforms]
        if all([(h != None) for h in xform_hashes]):
            if all([(h == xform_hashes[0]) for h in xform_hashes]):
                return xform_hashes[0]
            return hash_obj(tuple(xform_hashes))
        return None

    @hash.setter
    def hash(self, val):
        for xform in self:
            xform.hash = val

    @property
    def input_names(self):
        input_names = set()
        [input_names.update(x.input_names) for x in self]
        return tuple(sorted(input_names))

    @property
    def num_inputs(self):
        return len(self.input_names)

    @property
    def num_outputs(self):
        return len(self)

    @property
    def output_names(self):
        output_names = []
        [output_names.extend(x.output_name) for x in self]
        return tuple(output_names)

    def apply(self, inputs, cache=None):
        """Apply each transform to `inputs`; pass results and sideband objects
        through to the output.

        Parameters
        -----------
        inputs : sequence of objects
        cache : cache object or None

        Returns
        -------
        outputs : container with results of transforms and sideband objects

        """
        outputs = [xform.apply(inputs, cache=cache) for xform in self]

        # Start with all names in inputs
        unused_input_names = set([i.name for i in inputs])
        # Remove names that were used for a transform
        [unused_input_names.difference_update(x.input_names) for x in self]

        # Pass any unused objects through to the output; these are considered
        # to be "sideband" objects, which can be passed through to a later
        # stage without having an effect on the stages leading up to that one
        for name in unused_input_names:
            outputs.append(inputs[name])

        # TODO: what to set for name, tex, ... ?
        return MapSet(maps=outputs)


class Transform(object):
    """
    Transform.

    Properties
    ----------
    input_names
    num_inputs
    output_name
    """
    # Attributes that __setattr__ will allow setting
    _slots = ('_input_names', '_output_name', '_name', '_tex', '_hash')
    # Attributes that should be retrieved to fully describe state
    _state_attrs = ('input_names', 'output_name', 'name', 'tex', 'hash')

    def __init__(self, input_names, output_name, name=None, tex=None,
                 hash=None):
        if isinstance(input_names, basestring):
            input_names = [input_names]
        self._input_names = input_names
        self._output_name = output_name
        self._tex = tex
        self.name = name
        self.hash = hash

    @property
    def input_names(self):
        return self._input_names

    @property
    def num_inputs(self):
        return len(self.input_names)

    @property
    def output_name(self):
        return self._output_name

    def apply(self, inputs, cache=None):
        hashes = [self.hash]
        hashes.extend([inputs[name].hash for name in self.input_names])

        # All hashes must be present (for the transform and for each input
        # used by a transform) for a valid hash to be applied automatically to
        # the output
        if all([(h != None) for h in hashes]):
            output_hash = hash_obj(tuple(hashes))
        else:
            output_hash = None

        # Try to load result from cache, or recompute
        if cache is not None and output_hash in cache:
            output = cache[output_hash]
        else:
            output = self._apply(inputs)

        # TODO: tex, etc.?
        output.name = self.output_name
        return output

    def _apply(self, inputs):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_transform(xform):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_input(self, inputs):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')


class BinnedTensorTransform(Transform):
    """
    """
    _slots = tuple(list(Transform._slots) +
                   ['_input_binning', '_output_binning', '_xform_array'])

    _state_attrs = tuple(list(Transform._state_attrs) +
                         ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, input_names, output_name, input_binning, output_binning,
                 xform_array, name=None, tex=None, hash=None):
        super(self.__class__, self).__init__(input_names=input_names,
                                             output_name=output_name,
                                             name=name, tex=tex, hash=hash)
        self._input_binning = None
        self._output_binning = None
        self._xform_array = None
        self.input_binning = input_binning
        self.output_binning = output_binning
        self.xform_array = xform_array

    @property
    def input_binning(self):
        return self._input_binning

    @input_binning.setter
    def input_binning(self, binning):
        if not isinstance(binning, MultiDimBinning):
            binning = MultiDimBinning(binning)
        if self.xform_array is not None and self.output_binning is not None:
            self.validate_transform(input_binning=binning,
                                    output_binning=self.output_binning,
                                    xform_array=self.xform_array)
        self._input_binning = binning

    @property
    def xform_array(self):
        return self._xform_array

    @xform_array.setter
    def xform_array(self, x):
        self.validate_transform(self.input_binning, self.output_binning, x)
        self._xform_array = x

    # TODO: validate transform...
    def validate_transform(self, input_binning, output_binning, xform_array):
        """Superficial validation that the transform being set is reasonable.

        As of now, only checks shape.

        Expected transform shape is:
            (
             <input binning n_ebins>,
             <input binning n_czbins>,
             {if num_inputs > 1: <num_inputs>,}
             <output binning n_ebins>,
             <output binning n_czbins>,
             {if num_outputs > 1: <num_outputs>}
            )

        """
        #in_dim = [] if self.num_inputs == 1 else [self.num_inputs]
        #out_dim = [] if self.num_outputs == 1 else [self.num_outputs]
        #assert xform_array.shape == tuple(list(input_binning.shape) + in_dim +
        #                                  list(output_binning.shape) + out_dim)
        pass

    def validate_input(self, inputs):
        for input_name in self.input_names:
            assert input_name in inputs, \
                    'Input "%s" expected but not present.' % input_name
            assert inputs[input_name].binning == self.input_binning

    # TODO: make _apply work with multiple inputs (i.e., concatenate
    # these into a higher-dimensional array) and make logic for applying
    # element-by-element multiply and tensordot generalize to any dimension
    # given the (concatenated) input dimension and the dimension of the
    # transform kernel

    def _apply(self, inputs, cache=None):
        """Apply transforms to input maps to derive output maps.

        Parameters
        ----------
        inputs : Mapping
            Container class that must contain the maps to be transformed.
            There can be extra objects in `inputs` that are not used by this
            transform ("sideband" objects, which are simply ignored here).
            If multiple input maps are used by the transform, they are
            combined via
              numpy.stack((map0, map1, ... ), axis=0)
            I.e., the first dimension of the input sent to the transform has
            a length the same number of input maps requested by the transform.

        Returns
        -------
        output : Map
            Result of applying the transform to the input map(s).

        Notes
        -----
        For an input map that is M_ebins x N_czbins, the transform must either
        be 2-dimensional of shape (M x N) or 4-dimensional of shape
        (M x N x M x N). The latter case can be thought of as a 2-dimensional
        (M x N) array, each element of which is a 2-dimensional (M x N) array,
        and is currently used for the reconstruction stage's convolution
        kernels where there is one (M_ebins x N_czbins)-size kernel for each
        (energy, coszen) bin.

        """
        self.validate_input(inputs)
        if self.num_inputs > 1:
            input_array = np.stack([inputs[name].hist
                                    for name in self.input_names], axis=-1)
        else:
            input_array = inputs[self.input_names[0]].hist

        if self.xform_array.shape == input_array.shape:
            output = input_array * self.xform_array

        # TODO: Check that
        #   len(xform.shape) == 2*len(input_array.shape)
        # and then check that
        #   xform.shape == (input_array.shape, input_array.shape) (roughly)
        # and then apply tensordot appropriately for this generic case...
        elif len(self.xform_array.shape) == 2*len(input_array.shape):
            output = np.tensordot(input_array, self.xform_array,
                                  axes=([0,1], [0,1]))
        else:
            raise NotImplementedError(
                'Unhandled shapes for input(s) "%s": %s and'
                ' transform "%s": %s.'
                %(', '.join(self.input_names), input_array.shape, self.name,
                  self.xform_array.shape)
            )

        if self.num_inputs > 1:
            output = np.sum(output, axis=-1)

        # TODO: do rebinning here? (aggregate, truncate, and/or
        # concatenate 0's?)

        output = Map(name=self.output_name,
                     hist=output,
                     binning=self.output_binning)

        return output


def test_BinnedTensorTransform():
#if __name__ == "__main__":
    from pisa.core.map import Map, MapSet
    from pisa.core.binning import MultiDimBinning

    binning = MultiDimBinning([
        dict(name='energy', units='GeV', is_log=True, domain=(1,80),
             n_bins=10),
        dict(name='coszen', units=None, is_lin=True, domain=(-1,0), n_bins=5)
    ])

    nue_map = Map(name='nue',
                  binning=binning,
                  hist=np.random.random(binning.shape))
    nue_map.set_poisson_errors()
    numu_map = Map(name='numu',
                  binning=binning,
                  hist=np.random.random(binning.shape))
    numu_map.set_poisson_errors()
    inputs = MapSet(
        name='inputs',
        maps=[nue_map, numu_map],
    )

    xform0 = BinnedTensorTransform(
        name='nue_scale',
        input_names='nue',
        output_name='nue',
        input_binning=binning,
        output_binning=binning,
        xform_array=2*np.ones(binning.shape)
    )

    xform1 = BinnedTensorTransform(
        name='numu_scale',
        input_names=['numu'],
        output_name='numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=3*np.ones(binning.shape)
    )

    xform2 = BinnedTensorTransform(
        name='nue_scale+numu_scale',
        input_names=['nue', 'numu'],
        output_name='nue_numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=np.stack([2*np.ones(binning.shape),
                              3*np.ones(binning.shape)], axis=-1)
    )

    xforms = TransformSet(
        name='scaling',
        transforms=[xform0, xform1, xform2],
    )

    outputs = xforms.apply(inputs)
    print 'inputs:\n', inputs
    print '\noutputs:\n', outputs
    print '\nout.nue/in.nue:\n', (outputs.nue / inputs.nue).hist[0,0]
    print '\nout.numu/in.numu:\n', (outputs.numu / inputs.numu).hist[0,0]
    print '\nout.nue_numu/(2*in.nue+3*in.numu):\n', \
            (outputs.nue_numu / (2*inputs.nue+3*inputs.numu)).hist[0,0]


if __name__ == "__main__":
    test_BinnedTensorTransform()
