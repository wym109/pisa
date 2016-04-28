
import numpy as np

from pisa.utils.binning import MultiDimBinning


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
    __slots = ('_inputs', '_output', '_name', '_hash')
    # Attributes that should be retrieved to fully describe state
    __state_attrs = ('inputs', 'output', 'name', 'hash')
    def __init__(self, inputs, output, name=None, hash=None):
        if isinstance(inputs, basestring):
            inputs = [inputs]
        self._inputs = inputs
        self._output = output
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
    __slots = tuple(list(super(LinearTransform, self).__slots) +
                    ['_input_binning', '_output_binning', '_xform_array'])

    __state_attrs = tuple(list(super(LinearTransform, self).__state_attrs) +
                          ['input_binning', 'output_binning', 'xform_array'])

    def __init__(self, inputs, output, input_binning, output_binning,
                 xform_array, name=None, tex=None, hash=None):
        super(LinearTransform, self).__init__(inputs=inputs, output=output,
                                              name=name, tex=tex, hash=hash)
        self._input_binning = None
        self._output_binning = None
        self._xform_array = None
        self.input_binning = input_binning
        self.output_binning = output_binning
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
        self.validate_transform(x)
        self._xform_array = x

    @staticmethod
    def validate_transform(input_binning, output_binning, xform_array):
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
        in_dim = [] if self.num_inputs == 1 else [self.num_inputs]
        out_dim = [] if self.num_outputs == 1 else [self.num_outputs]
        assert xform_array.shape == tuple(list(input_binning.shape) + in_dim +
                                          list(output_binning.shape) + out_dim)

    def validate_input(map_set):
        for i in inputs:
            assert i in map_set, 'Input "%s" not in input map set' % i
            assert map_set[i].binning == self.input_binning

    # TODO: make the following work with multiple inputs (i.e., concatenate
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
            input_map = np.stack([input_map[n]
                                  for n in input_name in self.inputs])
        else:
            input_map = input_map[inputs[0]]

        if self.xform_array.shape == input_map.shape:
            output_map = input_map * self.xform_array
        elif len(sub_xform.shape) == 4:
            output_map = np.tensordot(input_map, sub_xform,
                                      axes=([0,1],[0,1]))

        # TODO: do rebinning here? (aggregate, truncate, and/or
        # concatenate 0's?)

        return output_map
