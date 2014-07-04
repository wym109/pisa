#
# jsons.py
#
# A set of utilities for dealing with JSON files.
# Import json from this module everywhere (if you need,
# and can not just use from_json, to_json)
# 
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import sys
import logging
import numpy as np

#try and get the much faster simplejson if we can
try:
    import simplejson as json
    from simplejson import JSONDecodeError
    logging.debug("Using simplejson")
except ImportError:
    import json as json
    #No DecodeError in default json, dummy one
    class JSONDecodeError(ValueError):
      pass
    logging.debug("Using json")

def json_string(string):
    '''Decode a json string'''
    return json.loads(string)

def from_json(filename):
    '''Open a file in JSON format an parse the content'''
    try:
        content = json.load(open(os.path.expandvars(filename)),cls=NumpyDecoder)
        return content
    except (IOError, JSONDecodeError), e:
        logging.error("Unable to read JSON file \'%s\'"%filename)
        logging.error(e)
        sys.exit(1)

def to_json(content, filename,indent=2):
    '''Write content to a JSON file using a custom parser that
       automatically converts numpy arrays to lists.'''
    with open(filename,'w') as outfile:
        json.dump(content,outfile, cls=NumpyEncoder,
                  indent=indent, sort_keys=True)
        logging.debug('Wrote %.2f kBytes to %s'%
                  (outfile.tell()/1024.,os.path.basename(filename)))
        

class NumpyEncoder(json.JSONEncoder):
    """
    Encode to JSON converting numpy.ndarrays to lists
    """
    def default(self, o):

        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


class NumpyDecoder(json.JSONDecoder):
    """ 
    Encode to numpy.ndarrays from JSON array
    """
    def __init__(self, encoding=None, object_hook=None, parse_float=None,
                 parse_int=None, parse_constant=None, strict=True,
                 object_pairs_hook=None):
        
        super(NumpyDecoder,self).__init__(encoding, object_hook, parse_float,
                                              parse_int, parse_constant, strict,
                                              object_pairs_hook)
        #only need to override the default array handler
        self.parse_array = self.json_array_numpy
        #self.memo = {}
        self.scan_once = json.scanner.py_make_scanner(self)

    def json_array_numpy(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        return np.array(values), end
