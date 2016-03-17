from pisa.utils import utils
import pisa.utils.utils.hash_obj as hash_obj


class Transform(DictWithHash):
    def __init__(self):
        super(Transform, self).__init()

    def apply(self, input_maps=None):
        """Apply linear transforms to input maps to derive output maps.

        Parameters
        ----------
        xform : utils.Transform
            Transform to be applied.
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


class Stage(object):
    def __init__(self, cache_class=utils.LRUCache, xfrom_cache_depth=10,
                 result_cache_depth=10, disk_cache_dir=None):
        self.cache_depth = cache_depth
        self.transform_cache = cache_class(self.cache_depth)
        self.result_cache = cache_class(self.cache_depth)
        self.disk_cache_dir = disk_cache_dir

        self.params = {}
        self.systematic_params = set()
        self.free_params = set()
        self.all_params = set()

    def set_params(self, params):
        """
        Parameters
        ----------
        params : str, dict

        """
        if isinstance(params, basestring):
            self.load_params(params)
        assert self.validate_params(params)
        self.params.update(params)

    def get_params(self, free_only=False, values_only=False):
        return self.params

    def get_free_params(self):
        return {fp: self.params[fp] for fp in self.free_params}

    def unfix_params(self, params):
        [self.free_params.remove(p) for p in params if p in self.free_params]

    def fix_params(self, params):
        [self.free_params.remove(p) for p in params if p in self.free_params]

    def compute_output_maps(self, input_maps=None):
        """Given input maps and (possibly) systematic parameters, compute the
        output maps.

        Parameters
        ----------
        input_maps : None or utils.DictWithHash
            For flux stages, `input_maps` = None.
        """
        xform = self.get_transform(params)

        if input_maps is None:
            return xform

        result_hash = hash_obj(input_maps.hash, xform.hash)
        try:
            return self.result_cache.get(result_hash)
        except KeyError:
            pass

        output_maps = xform.apply(input_maps)
        output_maps.update_hash(output_maps)

        return output_maps

    def get_transform(self, **kwargs):
        """"""
        systematic_params = self.get_systematic_params(**kwargs)
        systematics_hash = hash_obj(systematic_params)
        xform_hash = hash_obj(nominal_xform.hash, systematics_hash)
        try:
            return self.transform_cache.get(xform_hash)
        except KeyError:
            pass
        full_xform = self.apply_systematics(nominal_xform, **kwargs)
        full_xform.update_hash(full_xform_hash)
        self.transform_cache.set(cache_key, full_xform)
        return full_xform

    def get_systematic_params(self, **kwargs):
        """Each stage's base class must override this method"""
        raise NotImplementedError()

    def apply_systematics(self, nominal_xform, **kwargs):
        """Each stage's base class must override this method"""
        raise NotImplementedError()

    def get_nominal_transform(self, **kwargs):
        """Each stage implementation (aka "mode") must override this method"""
        raise NotImplementedError()

    @staticmethod
    def load_params(resource):

    @staticmethod
    def add_stage_cmdline_args(parser):
        """Each stage's base class should add generic stage args as well as
        args for systematics to a passed argparse parser"""
        raise NotImplementedError()

    @staticmethod
    def add_mode_cmdline_args(parser):
        """Each stage implementation (aka "mode") should add generic stage args
        as well as args for systematic"""
        raise NotImplementedError()
