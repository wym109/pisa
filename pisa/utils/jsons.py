#
# jsons.py
#
# A set of utilities for dealing with JSON files.
# Import json from this module everywhere (if you need to at all,
# and can not just use from_json, to_json)
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27


import os

import numpy as np

from pisa.utils.log import logging
from pisa.resources.resources import open_resource

# Try and get the much faster simplejson if we can
try:
    import simplejson as json
    from simplejson import JSONDecodeError
    logging.trace("Using simplejson")
except ImportError:
    import json as json
    # No DecodeError in default json, dummy one
    class JSONDecodeError(ValueError):
        pass
    logging.trace("Using json")


def json_string(string):
    """Decode a json string"""
    return json.loads(string)


def dumps(content, indent=2):
     return json.dumps(content, cls=NumpyEncoder, indent=indent,
                       sort_keys=True)


def loads(s):
     return json.loads(s, cls=NumpyDecoder)


def from_json(filename):
    """Open a file in JSON format an parse the content"""
    try:
        content = json.load(open_resource(filename),
                            cls=NumpyDecoder)
        return content
    except (IOError, JSONDecodeError), e:
        logging.error("Unable to read JSON file \'%s\'"%filename)
        logging.error(e)
        raise e


def to_json(content, filename, indent=2, overwrite=True):
    """Write content to a JSON file using a custom parser that automatically
    converts numpy arrays to lists."""
    fpath = os.path.expandvars(os.path.expanduser(filename))
    if os.path.exists(fpath):
        if overwrite:
            logging.warn('Overwriting file at ' + fpath)
        else:
            raise Exception('Refusing to overwrite path ' + fpath)

    with open(filename, 'w') as outfile:
        json.dump(content, outfile, cls=NumpyEncoder,
                  indent=indent, sort_keys=True)
        logging.debug('Wrote %.2f kB to %s' % (outfile.tell()/1024., filename))


class NumpyEncoder(json.JSONEncoder):
    """Encode to JSON converting numpy.ndarrays to lists"""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif type(o).__name__ in ['MultiDimBinning']:
            return o.__dict__
        elif type(o).__name__ in ['MapSet']:
            return o.maps
        elif type(o).__name__ in ['Quantity']:
            return o.to_tuple()
        elif type(o).__name__ in ['OneDimBinning', 'Map']:
            d = {}
            for slot in o._state_attrs:
                d[slot] = getattr(o, slot)
            return d
        try:
            return json.JSONEncoder.default(self, o)
        except:
            raise Exception('JSON serialization for %s not implemented'%type(o).__name__)


class NumpyDecoder(json.JSONDecoder):
    """Encode to numpy.ndarrays from JSON array, also returns python strings
    instead of unicode."""
    def __init__(self, encoding=None, object_hook=None, parse_float=None,
                 parse_int=None, parse_constant=None, strict=True,
                 object_pairs_hook=None):

        super(NumpyDecoder, self).__init__(encoding, object_hook, parse_float,
                                           parse_int, parse_constant, strict,
                                           object_pairs_hook)
        #only need to override the default array handler
        self.parse_array = self.json_array_numpy
        self.parse_string = self.json_python_string
        #self.memo = {}
        self.scan_once = json.scanner.py_make_scanner(self)

    def json_array_numpy(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        return np.array(values), end

    def json_python_string(self, s, end, encoding, strict):
        values, end = json.decoder.scanstring(s, end, encoding, strict)
        return values.encode('utf-8'), end
