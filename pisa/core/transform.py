
import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity

# TODO: Include option for propagating/not propagating errors, so that while
# e.g. a minimizer runs to match templates to "data," the overhead is not
# incurred. But this then requires -- if the user does want errors -- for a
# final iteration after a match has been found where all results are
# re-computed but with the propagate_errors option set to True. The output
# caches must all then "miss" so that actual output including error is
# computed.

# TODO: use a generic container, *not* a MapSet to store sets of maps for
# inputs and outputs (in fact I think there should be no MapSet object at all,
# so we can trnsparently handle e.g. events alongside maps where one is a
# sideband object for the other in a given stage, but which is which should be
# irrelevant).

# TODO: Add Sequence capabilities to TransformSet (e.g. it'd be nice to have at
# least append, extend, ...)
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
    nonvolatile_hash
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
        self.hash = None

    def __iter__(self):
        return iter(self.transforms)

    #@property
    #def hash(self):
    #    xform_hashes = [x.hash for x in transforms]
    #    if all([(h != None) for h in xform_hashes]):
    #        if all([(h == xform_hashes[0]) for h in xform_hashes]):
    #            return xform_hashes[0]
    #        return hash_obj(tuple(xform_hashes))
    #    return None

    # TODO: implement a non-volatile hash that includes source code hash in
    # addition to self.hash from the contained transforms
    @property
    def nonvolatile_hash(self):
        return None

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

    def apply(self, inputs):
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
        outputs = [xform.apply(inputs) for xform in self]

        # TODO: what to set for name, tex, ... ?
        return MapSet(maps=outputs)


class Transform(object):
    """
    Transform.

    Parameters
    ----------

    Properties
    ----------
    hash
    input_names
    num_inputs
    output_name
    tex

    """
    # Attributes that __setattr__ will allow setting
    _slots = ('_input_names', '_output_name', '_tex', '_params_hash', '_hash')
    # Attributes that should be retrieved to fully describe state
    _state_attrs = ('input_names', 'output_name', 'tex', 'hash')

    def __init__(self, input_names, output_name, input_binning=None,
                 output_binning=None, tex=None, params_hash=None):
        if isinstance(input_names, basestring):
            input_names = [input_names]
        assert isinstance(output_name, basestring)
        self._input_names = input_names
        self._output_name = output_name
        if input_binning is not None:
            self._input_binning = MultiDimBinning(input_binning)
        else:
            self._input_binning = None
        if output_binning is not None:
            self._output_binning = MultiDimBinning(output_binning)
        else:
            self._output_binning = None
        self._tex = tex if tex is not None else output_name
        self._params_hash = params_hash

    @property
    def hash(self):
        return self._hash

    @property
    def input_names(self):
        return self._input_names

    @property
    def num_inputs(self):
        return len(self.input_names)

    @property
    def output_name(self):
        return self._output_name

    @property
    def input_binning(self):
        return self._input_binning

    @property
    def output_binning(self):
        return self._output_binning

    def apply(self, inputs):
        logging.trace('applying transform to inputs.')
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
    Parameters
    ----------
    input_names
    output_name
    input_binning
    output_binning
    params_hash

    Properties
    ----------
    hash
    params_hash
    source_hash
    """
    _slots = tuple(list(Transform._slots) +
                   ['_input_binning', '_output_binning', '_xform_array'])

    _state_attrs = tuple(list(Transform._state_attrs) +
                         ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, input_names, output_name, input_binning, output_binning,
                 xform_array, tex=None, params_hash=None):
        super(self.__class__, self).__init__(input_names=input_names,
                                             output_name=output_name,
                                             input_binning=input_binning,
                                             output_binning=output_binning,
                                             tex=tex,
                                             params_hash=params_hash)
        self.xform_array = xform_array

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
        """Apply transforms to input maps to compute output maps.

        Parameters
        ----------
        inputs : Mapping
            Container class that must contain (at least) the maps to be
            transformed.

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

        There can be extra objects in `inputs` that are not used by this
        transform ("sideband" objects, which are simply ignored here). If
        multiple input maps are used by the transform, they are combined via
          numpy.stack((map0, map1, ... ), axis=0)
        I.e., the first dimension of the input sent to the transform has a
        length the same number of input maps requested by the transform.

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
        input_names='nue',
        output_name='nue',
        input_binning=binning,
        output_binning=binning,
        xform_array=2*np.ones(binning.shape)
    )

    xform1 = BinnedTensorTransform(
        input_names=['numu'],
        output_name='numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=3*np.ones(binning.shape)
    )

    xform2 = BinnedTensorTransform(
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
