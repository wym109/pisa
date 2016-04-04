
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

    """
    def __init__(self, transforms, name=None, hash=None):
        self.transforms = transforms
        self.name = name
        self.hash = hash

    @property
    def inputs(self):
        pass

    @property
    def outputs(self):
        pass

    @property
    def dependencies(self):
        pass


class Transform(object):
    # Attributes that __setattr__ will allow setting
    __slots = ('_inputs', '_outputs', '_name', '_hash')
    # Attributes that should be retrieved to fully describe state
    __state_attrs = ('inputs', 'outputs', 'name', 'hash')
    def __init__(self, inputs, outputs, name=None, hash=None):
        self._inputs = inputs
        self._outputs = outputs
        self.name = name
        self.hash = hash

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def validate_transform(xform):
        raise NotImplementedError()


class LinearTransform(Transform):
    __slots = tuple(list(super(LinearTransform, self).__slots) +
                    ['_input_binning', '_output_binning', '_xform_array'])

    __state_attrs = tuple(list(super(LinearTransform, self).__state_attrs) +
                          ['input_binning', 'output_binning', 'xform_array']

    def __init__(self, inputs, outputs, input_binning, output_binning,
                 xform_array, name=None, hash=None):
        super(LinearTransform, self).__init__(inputs=inputs, outputs=outputs,
                                              name=name, hash=hash)
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

    @xform_array.setter
    def xform_array(self, x):
        self.validate_transform(x)
        self._xform_array = x

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
             <output binning N_czbins>,
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

    def apply(self, input_maps):
        """Apply linear transforms to input maps to derive output maps.

        Parameters
        ----------
        input_maps : MapSet
            Maps to be transformed.
            (??)If `input_maps` is None, simply returns `xform`.(??)

        Returns
        -------
        output_maps : MapSet

        Notes
        -----
        For an input map that is M_ebins x N_czbins, the transform must either
        be 2-dimensional of shape (M x N) or 4-dimensional of shape
        (M x N x M x N). The latter case can be thought of as a 2-dimensional
        (M x N) array, each element of which is a 2-dimensional (M x N) array,
        and is currently used for the reconstruction stage's convolution
        kernels where there is one (M_ebins x N_czbins)-size kernel for each
        (energy, coszen) bin.

        One special case is if transform is None

        Re-binning before, during, and/or after applying the transform is not
        implemented, but should be!
        """
        self.validate_input(input_maps)
        output_maps = utils.DictWithHash()
        for output_key, input_xforms in xform.iteritems():
            for input_key, sub_xform in xforms.iteritems():
                input_map = input_maps[input_key]
                if sub_xform is None:
                    output_map = input_map
                elif len(sub_xform.shape) == 2:
                    output_map = input_map * sub_xform
                elif len(sub_xform.shape) == 4:
                    output_map = np.tensordot(input_map, sub_xform,
                                              axes=([0,1],[0,1]))

                # TODO: do rebinning here? (aggregate, truncate, and/or
                # concatenate 0's?)

                if output_key in output_maps:
                    output_maps[output_key] += output_map
                else:
                    output_maps[output_key] = output_map

