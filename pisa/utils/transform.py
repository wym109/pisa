
import numpy as np

class Transform(DictWithHash):
    def __init__(self):
        super(Transform, self).__init()
        self.__xform = None

    @property
    def transforms(self):
        return self.__xforms

    @xform.setter
    def transforms(self, x):
        self.validate_transforms(x)
        self.__xforms = val

    def validate_transforms(x):
        for k, v in x.iteritems():
            len_shape = len(u.shape)
            assert len_shape in [2, 4]

    def apply(self, input_maps=None):
        """Apply linear transforms to input maps to derive output maps.

        Parameters
        ----------
        input_maps : None or utils.DictWithHash
            Maps to be transformed. If `input_maps` is None, simply returns
            `xform`, to accomodate Flux stage, which has no input.

        Returns
        -------
        output_maps : utils.DictWithHash

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

