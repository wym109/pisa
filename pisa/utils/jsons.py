# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27
"""
A set of utilities for reading (and instantiating) objects from and writing
objects to JSON files.

Import json from this module everywhere (if you need to at all, and can not
just use from_json, to_json) for... faster JSON serdes?
"""
# TODO: why the second line above?


import os

import numpy as np
import simplejson as json

import pint
from pisa import ureg, Q_
from pisa.resources.resources import open_resource
from pisa.utils.log import logging


def json_string(string):
    """Decode a json string"""
    return json.loads(string)


def dumps(content, indent=2):
     return json.dumps(content, cls=NumpyEncoder, indent=indent, sort_keys=True)


def loads(s):
     return json.loads(s, cls=NumpyDecoder)


def from_json(filename):
    """Open a file in JSON format an parse the content"""
    try:
        content = json.load(open_resource(filename), cls=NumpyDecoder)
        return content
    except (IOError, json.JSONDecodeError), e:
        logging.error('Unable to read JSON file "%s"' %filename)
        logging.error(e)
        raise e


def to_json(content, filename, indent=2, overwrite=True):
    """Write content to a JSON file using a custom parser that automatically
    converts numpy arrays to lists.

    Parameters
    ----------
    filename : str
    indent : int
    overwrite : bool

    """
    if hasattr(content, 'to_json'):
        return content.to_json(filename, indent=indent, overwrite=overwrite)
    fpath = os.path.expandvars(os.path.expanduser(filename))
    if os.path.exists(fpath):
        if overwrite:
            logging.warn('Overwriting file at ' + fpath)
        else:
            raise Exception('Refusing to overwrite path ' + fpath)

    with open(filename, 'w') as outfile:
        json.dump(content, outfile, indent=indent, cls=NumpyEncoder,
                  sort_keys=True, allow_nan=True, ignore_nan=False)
        logging.debug('Wrote %.2f kB to %s' % (outfile.tell()/1024., filename))


class NumpyEncoder(json.JSONEncoder):
    """Encode special objects to be representable as JSON."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # TODO: poor form to have a way to get this into a JSON file but no way
        # to get it out of a JSON file... so either write a deserializer, or
        # remove this and leave it to other objects to do the following.
        elif isinstance(obj, pint.quantity._Quantity):
            return obj.to_tuple()
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            raise Exception('JSON serialization for %s not implemented'
                            %type(obj).__name__)

class NumpyDecoder(json.JSONDecoder):
    """Decode JSON array(s) as numpy.ndarray, also returns python strings
    instead of unicode."""
    def __init__(self, encoding=None, object_hook=None, parse_float=None,
                 parse_int=None, parse_constant=None, strict=True,
                 object_pairs_hook=None):

        super(NumpyDecoder, self).__init__(encoding, object_hook, parse_float,
                                           parse_int, parse_constant, strict,
                                           object_pairs_hook)
        # Only need to override the default array handler
        self.parse_array = self.json_array_numpy
        self.parse_string = self.json_python_string
        self.scan_once = json.scanner.py_make_scanner(self)

    def json_array_numpy(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        try:
            values = np.array(values, dtype=float)
        except:
            pass
        return values, end

    def json_python_string(self, s, end, encoding, strict):
        values, end = json.decoder.scanstring(s, end, encoding, strict)
        return values.encode('utf-8'), end


# TODO: finish this little bit
def test_NumpyEncoderDecoder():
    import tempfile
    from pisa.utils.comparisons import recursiveEquality
    from pisa.utils.log import logging, set_verbosity
    set_verbosity(3)
    nda1 = np.array([-np.inf, np.nan, np.inf, -1, 0, 1, ])
    testdir = tempfile.mkdtemp()
    fname = os.path.join(testdir, 'nda1.json')
    to_json(nda1, fname)
    nda2 = from_json(fname)
    assert np.allclose(nda2, nda1, rtol=1e-12, atol=0, equal_nan=True), \
            'nda1=\n%s\nnda2=\n%s\nsee file: %s' %(nda1, nda2, fname)
    d1 = {'nda1': nda1}
    fname = os.path.join(testdir, 'd1.json')
    to_json(d1, fname)
    d2 = from_json(fname)
    assert recursiveEquality(d2, d1), \
            'd1=\n%s\nd2=\n%s\nsee file: %s' %(d1, d2, fname)
    logging.info('<< PASSED : test_NumpyEncoderDecoder >>')


if __name__ == '__main__':
    test_NumpyEncoderDecoder()
