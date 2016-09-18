import cPickle as pickle
import hashlib
import numpy as np
import struct

from pisa.utils.log import logging, set_verbosity
#from pisa.utils.profiler import profile


# TODO: add sigfigs arg:
# sigfigs : None or integer
#     If specified, round all numerical quantities being hashed prior to
#     serializing them, such that values that would evaluate to be equal
#     within that number of significant figures will hash to the same value.
#@profile
def hash_obj(obj, hash_to='int'):
    """Return hash for an object. Object can be a numpy ndarray or matrix
    (which is serialized to a string), an open file (which has its contents
    read), or any pickle-able Python object.

    Note that only the first most-significant 8 bytes (64 bits) from the MD5
    sum are used in the hash.

    Parameters
    ----------
    obj : object
        Object to hash. Note that the larger the object, the longer it takes to
        hash.

    hash_to : string
        'i', 'int', or 'integer': First 8 bytes of the MD5 sum are interpreted
            as an integer.
        'b', 'bin', or 'binary': MD5 sum digest; returns an 8-character string 
        'h', 'x', 'hex': MD5 sum hexdigest, (string of 16 characters)
        'b64', 'base64': first 8 bytes of MD5 sum are base64 encoded (with '+'
            and '-' as final two characters of encoding). Returns string of 11
            characters.

    Returns
    -------
    hash_val : int or string

    See also
    --------
    hash_file : hash a file on disk by filename

    """
    if hash_to is None:
        hash_to = 'int'
    hash_to = hash_to.lower()

    if hasattr(obj, 'hash') and obj.hash is not None and obj.hash == obj.hash:
        return obj.hash

    # Handle numpy arrays and matrices specially
    # TODO: is this still needed now that we use pickle?
    if isinstance(obj, np.ndarray) or isinstance(obj, np.matrix):
        return hash_obj(obj.tostring())
    # Handle e.g. an open file specially
    if hasattr(obj, 'read'):
        return hash_obj(obj.read())
    try:
        pkl = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    except:
        logging.error('Failed to pickle `obj` "%s" of type "%s"'
                      %(obj, type(obj)))
        raise
    md5hash = hashlib.md5(pkl)
    #md5hash = hashlib.md5(repr(obj))
    if hash_to in ['i', 'int', 'integer']:
        hash_val, = struct.unpack('<q', md5hash.digest()[:8])
    elif hash_to in ['b', 'bin', 'binary']:
        hash_val = md5hash.digest()[:8]
    elif hash_to in ['h', 'x', 'hex', 'hexadecimal']:
        hash_val = md5hash.hexdigest()[:16]
    elif hash_to in ['b64', 'base64']:
        hash_val = base64.b64encode(md5hash.digest()[:8], '+-')
    else:
        raise ValueError('Unrecognized `hash_to`: "%s"' % (hash_to,))
    return hash_val


def hash_file(fname, hash_to=None):
    """Return a hash for a file, passing contents through hash_obj function."""
    resource = find_resource(fname)
    with open(resource, 'rb') as f:
        return hash_obj(f, hash_to=hash_to)


def test_hash_obj():
    assert hash_obj('x') == 5342080905610180975
    assert hash_obj('x') == 5342080905610180975
    #assert hash_obj('x', hash_to='bin') == '\xfdn ]\xda\xe4\x8a\xde&\x80xNg+f'.encode,\
    #        (hash_obj('x', hash_to='bin')).decode('ascii')
    assert hash_obj('x', hash_to='hex') == '6fb94ab447e2224a422e2cd0271d66c1'
    assert hash_obj(object) == 7177477609730129002
    assert hash_obj(object()) != hash_obj(object)
    print '<< PASSED : test_hash_obj >>'

# TODO: test_hash_file function requires a "standard" file to test on

if __name__ == "__main__":
    test_hash_obj()
