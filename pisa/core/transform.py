
import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.map import MapSet


class TransformSet(object):
    """
    Parameters
    ----------
    transforms
    name
    hash

    Properties
    ----------
    name
    hash
    transforms
    inputs
    outputs
    dependencies

    Methods
    -------
    check_predecessor_compat
    check_successor_compat
    apply

    """
    def __init__(self, transforms, name=None, hash=None):
        self.transforms = transforms
        self.name = name

    def __iter__(self):
        return iter(self.transforms)

    @property
    def hash(self):
        xform_hashes = [x.hash for x in transforms]
        if all([(h != None) for h in xform_hashes]):
            return hash_obj(xform_hashes)
        return None

    @property
    def inputs(self):
        inputs = []
        [inputs.extend(x.inputs) for x in self]
        return tuple(inputs)

    @property
    def outputs(self):
        outputs = []
        [outputs.extend(x.output) for x in self]
        return tuple(outputs)

    def apply(self, input_maps, cache=None):
        output_maps = [xform.apply(input_maps, cache=cache) for xform in self]
        used_input_names = set()
        for xform in self:
            used_input_names = used_input_names.union(xform.inputs)

        # Pass any unused maps through to the output
        for input_map in input_maps:
            if input_map.name not in used_input_names:
                output_maps.append(input)

        # TODO: what to set for name, tex, ... ?
        return MapSet(maps=output_maps)


class Transform(object):
    # Attributes that __setattr__ will allow setting
    _slots = ('_inputs', '_output', '_name', '_tex', '_hash')
    # Attributes that should be retrieved to fully describe state
    _state_attrs = ('inputs', 'output', 'name', 'tex', 'hash')

    def __init__(self, inputs, output, name=None, tex=None, hash=None):
        if isinstance(inputs, basestring):
            inputs = [inputs]
        self._inputs = inputs
        self._output = output
        self._tex = tex
        self.name = name
        self.hash = hash

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output

    def apply(self, input_maps, cache=None):
        hashes = [self.hash]
        hashes.extend([input_maps[name].hash for name in self.inputs])

        # All hashes must be present (transform and for each input map
        # used) for a valid hash to be applied automatically to the output
        # map
        if all([(h != None) for h in hashes]):
            outmap_hash = hash_obj(tuple(hashes))
        else:
            outmap_hash = None

        # Try to load result from cache, or recompute
        if cache is not None and outmap_hash in cache:
            outmap = cache[outmap_hash]
        else:
            outmap = self._apply(input_maps)

        # TODO: tex, etc.?
        outmap.name = self.output
        return outmap

    def _apply(self, input_maps):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_transform(xform):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')

    def validate_input(self, input_maps):
        """Override this method in subclasses"""
        raise NotImplementedError('Override this method in subclasses')


class LinearTransform(Transform):
    _slots = tuple(list(Transform._slots) +
                   ['_input_binning', '_output_binning', '_xform_array'])

    _state_attrs = tuple(list(Transform._state_attrs) +
                         ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, inputs, output, input_binning, output_binning,
                 xform_array, name=None, tex=None, hash=None):
        super(LinearTransform, self).__init__(inputs=inputs, output=output,
                                              name=name, tex=tex, hash=hash)
        self._input_binning = None
        self._output_binning = None
        self._xform_array = None
        self.input_binning = MultiDimBinning(input_binning)
        self.output_binning = MultiDimBinning(output_binning)
        self.xform_array = xform_array
        self.num_inputs = len(inputs)
        self.num_outputs = 1

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

    def validate_transform(self, input_binning, output_binning, xform_array):
        """Superficial validation that the transform being set is reasonable.

        As of now, only checks shape.

        Expected transform shape is:
            (
             <input binning n_ebins>,
             <input binning n_czbins>,
             {if n_inputs > 1: <n_inputs>,}
             <output binning n_ebins>,
             <output binning n_czbins>,
             {if n_outputs > 1: <n_outputs>}
            )

        """
        #in_dim = [] if self.num_inputs == 1 else [self.num_inputs]
        #out_dim = [] if self.num_outputs == 1 else [self.num_outputs]
        #assert xform_array.shape == tuple(list(input_binning.shape) + in_dim +
        #                                  list(output_binning.shape) + out_dim)
        pass

    def validate_input(self, map_set):
        for i in self.inputs:
            assert i in map_set, 'Input "%s" not in input map set' % i
            assert map_set[i].binning == self.input_binning

    # TODO: make _apply work with multiple inputs (i.e., concatenate
    # these into a higher-dimensional array) and make logic for applying
    # element-by-element multiply and tensordot generalize to any dimension
    # given the (concatenated) input dimension and the dimension of the linear
    # transform kernel

    def _apply(self, input_maps, cache=None):
        """Apply linear transforms to input maps to derive output maps.

        Parameters
        ----------
        input_maps : MapSet
            Maps to be transformed. There can be extra maps in `input_maps`
            that are not used by this transform. If multiple input maps are
            used in the transformation, they are combined via
              numpy.stack((map0, map1, ... ), axis=0)
            I.e., the first dimension of the input sent to the transform has
            length of number of maps being combined.

        Returns
        -------
        output_map : map

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
        self.validate_input(input_maps)
        if len(self.inputs) > 1:
            input_map = np.stack([input_maps[n]
                                  for n in self.inputs])
        else:
            input_map = input_maps[self.inputs[0]]

        if self.xform_array.shape == input_map.shape:
            output_map = input_map * self.xform_array

        # TODO: Check that
        #   len(xform.shape) == 2*len(input.shape)
        # and then check that
        #   xform.shape == (input.shape, input.shape) (essentially)
        # and then apply tensordot appropriately for this generic case...
        elif len(sub_xform.shape) == 4 and len(input_maps.shape) == 2:
            output_map = np.tensordot(input_map, sub_xform,
                                      axes=([0,1],[0,1]))
        else:
            raise NotImplementedError(
                'Unhandled shapes for input (%s) and transform (%s).'
                %(input_maps.shape, self.xform_array.shape)
            )

        # TODO: do rebinning here? (aggregate, truncate, and/or
        # concatenate 0's?)

        return output_map


def test_LinearTransform():
    from pisa.core.map import Map, MapSet

    binning = [
        dict(name='energy', units='GeV', is_log=True, domain=(1,80),
             n_bins=10),
        dict(name='coszen', units=None, is_lin=True, domain=(-1,0), n_bins=5)
    ]
    input_maps = MapSet(
        name='input',
        maps=[
            Map(
                name='nue',
                binning=binning,
                hist=np.random.random((10, 5))
            ),
            Map(
                name='numu',
                binning=binning,
                hist=np.random.random((10, 5))
            )
        ],
    )

    xform0 = LinearTransform(
        name='nue_scale',
        inputs=['nue'],
        output='nue',
        input_binning=binning,
        output_binning=binning,
        xform_array=2*np.ones((10,5)),
    )

    xform1 = LinearTransform(
        name='numu_scale',
        inputs=['numu'],
        output='numu',
        input_binning=binning,
        output_binning=binning,
        xform_array=3*np.ones((10,5)),
    )

    xform_set = TransformSet(
        name='scaling',
        transforms=[xform0, xform1],
    )

    output_maps = xform_set.apply(input_maps)
    print 'input_maps:', input_maps
    print 'output_maps:', output_maps
    print 'out/in:', (output_maps / input_maps)[:,:].hist


if __name__ == "__main__":
    test_LinearTransform()
