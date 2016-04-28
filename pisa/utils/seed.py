import numpy as np

def n_bad_seeds(*args):
    """Set random state based upon some number of seeds."""
    np.random.seed(args[0])
    for n, badseed in enumerate(args):
        next_seed_set = np.random.randint(0, 2**32, badseed+1)
        # init generator with bad seed
        np.random.seed(next_seed_set[badseed])
        # blow through some states to increase entropy
        np.random.randint(-1e9,1e9,1e5)
        # grab a good seed (the next randomly-generated integer)
        goodseed = np.random.randint(0, 2**32, 1)
        # seed the generator with the good seed
        np.random.seed(goodseed)
        # blow through some states to increase entropy
        np.random.randint(-1e9,1e9,1e5)
    return np.random.get_state()
