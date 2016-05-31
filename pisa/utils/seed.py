# author: J.L. Lanfranchi
# date:   2016-05-01


import numpy as np


def n_bad_seeds(*args):
    """Set random state based upon some number of seeds which themselves
    needn't be "good" seeds.

    Parameters
    ----------
    *args : one or more integers in range [0, 2**32-1]

    Returns
    -------
    state : full numpy random number generator state for reference
    """
    np.random.seed(args[0])
    for n, badseed in enumerate(args):
        assert badseed >= 0 and badseed < 2**32
        next_seed_set = np.random.randint(0, 2**32, badseed+1)
        # init generator with bad seed
        np.random.seed(next_seed_set[badseed])
        # blow through some states to increase entropy
        np.random.randint(-1e9, 1e9, 1e5)
        # grab a good seed (the next randomly-generated integer)
        goodseed = np.random.randint(0, 2**32, 1)
        # seed the generator with the good seed
        np.random.seed(goodseed)
        # blow through some states to increase entropy
        np.random.randint(-1e9, 1e9, 1e5)
    return np.random.get_state()


def test_n_bad_seeds():
    n_bad_seeds(*[1, 2, 3])
    assert np.random.randint(low=-2**32, high=2**32) == 330226075
    print '<< PASSED : test_n_bad_seeds >>'


if __name__ == '__main__':
    test_n_bad_seeds()
