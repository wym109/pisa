import struct
import hashlib
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np


# TODO: add sigfigs arg:
# sigfigs : None or integer
#     If specified, round all numerical quantities being hashed prior to
#     serializing them, such that values that would evaluate to be equal
#     within that number of significant figures will hash to the same value.

def hash_obj(obj, hash_to='int'):
    """Return hash for an object. Object can be a numpy ndarray or matrix
    (which is serialized to a string), an open file (which has its contents
    read), or any pickle-able Python object.

    Parameters
    ----------
    obj : object
        Object to hash. Note that the larger the object, the longer it takes to
        hash.

    hash_to : string
        'i', 'int', or 'integer': Hash is derived from the first 8 bytes of the
            MD5 sum, interpreted as an integer.
        'b', 'bin', or 'binary': MD5 sum digest
        'h', 'x', 'hex': MD5 sum hexdigest

    Returns
    -------
    hash

    See also
    --------
    hash_file : hash a file on disk by filename

    """
    if hash_to is None:
        hash_to = 'int'
    hash_to = hash_to.lower()

    # Handle numpy arrays and matrices specially
    # TODO: is this still needed now that we use pickle?
    if isinstance(obj, np.ndarray) or isinstance(obj, np.matrix):
        return hash_obj(obj.tostring())
    # Handle e.g. an open file specially
    if hasattr(obj, 'read'):
        return hash_obj(obj.read())
    hash = hashlib.md5(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    if hash_to in ['i', 'int', 'integer']:
        hash_val, = struct.unpack('<q', hash.digest()[:8])
    elif hash_to in ['b', 'bin', 'binary']:
        hash_val = hash.digest()
    elif hash_to in ['h', 'x', 'hex', 'hexadecimal']:
        hash_val = hash.hexdigest()
    else:
        raise ValueError('Unrecognized `hash_to`: "%s"' % (hash_to,))
    return hash_val


def hash_file(fname, hash_to=None):
    """Return a hash for a file, passing contents through hash_obj function."""
    resource = find_resource(fname)
    with open(resource, 'rb') as f:
        return hash_obj(f, hash_to=hash_to)


def test_hash_obj():
    assert hash_obj('x') == -8438379708274508437
    assert hash_obj('x') == -8438379708274508437
    #assert hash_obj('x', hash_to='bin') == '\xfdn ]\xda\xe4\x8a\xde&\x80xNg+f'.encode,\
    #        (hash_obj('x', hash_to='bin')).decode('ascii')
    assert hash_obj('x', hash_to='hex') == '6bfd6e205ddae48ade2680784e672b66'
    assert hash_obj(object) == -591373952375362512
    assert hash_obj(object()) == -5704184814176152584


if __name__ == "__main__":
    test_hash_obj()
