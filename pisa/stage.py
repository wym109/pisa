
import collections

from pisa.utils import utils
import pisa.utils.utils.hash_obj as hash_obj


class Stage(object):
    def __init__(self, params=None, no_input_stage=False,
                 cache_class=utils.LRUCache, xfrom_cache_depth=10,
                 result_cache_depth=10, disk_cache_dir=None):
        self.__no_input_stage = no_input_stage
        self.__params = dict()
        self.__free_param_names = set()

        self.cache_depth = cache_depth
        self.transform_cache = cache_class(self.cache_depth)
        self.result_cache = cache_class(self.cache_depth)
        self.disk_cache_dir = disk_cache_dir

        if params is not None:
            self.params = params

    def compute_output_maps(self, input_maps=None):
        """Compute the output maps by applying the stage's transform to the
        input -- except in the case of a no-input stage, simply return the
        transform.

        Parameters
        ----------
        input_maps : None or utils.DictWithHash
            For flux stages, `input_maps` = None.
        """
        xform = self.get_transform()

        if self.__no_input_stage:
            assert input_maps is None:
            return xform

        result_hash = hash_obj(input_maps.hash, xform.hash)
        try:
            return self.result_cache.get(result_hash)
        except KeyError:
            pass

        output_maps = xform.apply(input_maps)
        output_maps.update_hash(output_maps)

        return output_maps

    def get_transform(self):
        """Load a cached transform (keyed on hash of free parameters) if it is
        in the cache, or else derive a new transform from currently-set
        parameter values and store this to the transform cache.
        """
        xform_hash = hash_obj(self.free_params)
        try:
            return self.transform_cache.get(xform_hash)
        except KeyError:
            pass
        xform = self._derive_transform()
        xform.update_hash(xform_hash)
        self.transform_cache.set(cache_key, xform)
        return xform

    @property
    def params(self):
        ordered = collections.OrederedDict()
        [ordered[k].__setitem__(self.__params[k])
         for k in sorted(self.__params)]
        return ordered

    @params.setter
    def params(self, p):
        if isinstance(p, basestring):
            self.load_params(p)
            return
        elif isinstance(p, collections.Mapping):
            pass
        else:
            raise TypeError('Unhandled `params` type "%s"' % type(p))
        self.validate_params(p)
        self.__params.update(p)

    def load_params(self, resource):
        if isinstance(resource, basestring):
            params_dict = fileio.from_file(resource)
        elif isinstance(collections.Mapping):
            params_dict = resource
        else:
            raise TypeError('Unhandled `rsource` type "%s"' % type(resource))
        self.params = params_dict

    @property
    def free_params(self):
        ordered = collections.OrederedDict()
        [ordered[k].__setitem__(self.__params[k])
         for k in sorted(self.__free_param_names)]
        return ordered

    @free_params.setter
    def free_params(self, p):
        if isinstance(p, (collections.Iterable, collections.Sequence)):
            p = {pname: p[n]
                 for n, pname in enumerate(sorted(self.__free_param_names))}
            assert len(p) == len(self.__free_param_names)
        if isinstance(p, collections.Mapping):
            assert set(p.keys()).issubset(self.__free_param_names)
            self.validate(p)
            self.__params.update(p)
        else:
            raise TypeError('Unhandled `params` type "%s"' % type(p))

    def unfix_params(self, params, ignore_missing=False):
        if ignore_missing:
            self.__free_param_names.difference_update(params)
        else:
            [self.__free_param_names.remove(p) for p in params]

    def fix_params(self, params, ignore_missing=False):
        if np.isscalar(params):
            params = [params]
        if ignore_missing:
            [self.free_params.add(p) for p in params if p in self.__params]
        else:
            assert
            self.__free_param_names.difference_update(params)

    def _derive_transform(self, **kwargs):
        """Each stage implementation (aka service) must override this method"""
        raise NotImplementedError()

    @staticmethod
    def add_stage_cmdline_args(parser):
        """Each stage's base class should override this. Here, add generic
        stage args -- as well as args for systematics applicable across all
        implementations of the stage -- to `parser`.
        """
        raise NotImplementedError()

    @staticmethod
    def add_service_cmdline_args(parser):
        """Each stage implementation (aka service) should add generic stage
        args as well as args for systematics specific to it here"""
        raise NotImplementedError()

