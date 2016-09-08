#! /usr/bin/env python
# author:  J.L. Lanfranchi
# email:   jll1062+pisa@phys.psu.edu
# date:    Sept 3, 2016


from collections import Sequence

import numpy as np


def get_random_state(random_state, jumpahead=0):
    """Derive a `numpy.random.RandomState` object (usable to generate random
    numbers and distributions) from a flexible specification..

    Parameters
    ----------
    random_state : RandomState, string, int, state vector, or sequence of int
        Note for all of the below cases, `jumpahead` is applied _after_ the
        RansomState is initialized using the `random_state` (except for
        `random_state` indicating a truly-random number, in which case
        `jumpahead` is ignored).
        * If instantiated RandomState object is passed, it is used directly
        * If string : must be either 'rand' or 'random'; random state is
          instantiated at random from either /dev/urandom or (if that is not
          present) the clock. This creates an irreproducibly-random number.
          `jumpahead` is ignored.
        * If int or sequence of lenth one: This is used as the `seed` value;
          must be in [0, 2**32).
        * If sequence of two integers: first must be in [0, 32768): 15
          most-significant bits. Second must be in [0, 131072): 17
          least-significant bits.
        * If sequence of three integers: first must be in [0, 32): 4
          most-significant bits. Second must be in [0, 2048): next 11
          less-significant bits. Third must be in [0, 131072): 17
          least-significant bits.
        * If a "state vector" (sequence of length five usable by
          `numpy.random.RandomState.set_state`), set the random state using
          this method.

    jumpahead : int >= 0
        Starting with the random state specified by `random_state`, produce
        `jumpahead` random numbers to move this many states forward in the
        random number generator's finite state machine. Note that this is
        ignored if `random_state`="random" since jumping ahead any number of
        states from a truly-random point merely yields another truly-random
        point, but takes additional computational time.

    Returns
    -------
    numpy.random.RandomState

    """
    if isinstance(random_state, np.random.RandomState):
        new_random_state = random_state

    elif isinstance(random_state, basestring):
        allowed_strings = ['rand', 'random']
        rs = random_state.lower().strip()
        if rs not in allowed_strings:
            raise ValueError(
                '`random_state`=%s not a valid string. Must be one of %s.'
                %(random_state, allowed_strings)
            )
        new_random_state = np.random.RandomState()
        jumpahead = 0

    elif isinstance(random_state, int):
        new_random_state = np.random.RandomState(seed=random_state)
        
    elif isinstance(random_state, Sequence):
        new_random_state = np.random.RandomState()
        if all([isinstance(x, int) for x in random_state]):
            if len(random_state) == 1:
                seed = random_state[0]
                assert seed >= 0 and seed_spec < 2**32
            elif len(seed_spec) == 2:
                s0, s1 = seed_spec
                assert s0 >= 0 and s0 < 2**15
                assert s1 >= 0 and s1 < 2**17
                seed = (s0 << 17) + s1
            elif len(seed_spec) == 3:
                s0, s1, s2 = seed_spec
                assert s0 >= 0 and s0 < 2**4
                assert s1 >= 0 and s1 < 2**11
                assert s2 >= 0 and s2 < 2**17
                seed = (s0 << 11+17) + (s1 << 17) + s2
            else:
                raise ValueError(
                    '`random_state` sequence of int must be length 1-3'
                )
            new_random_state.seed(seed)
        elif len(random_state) == 5:
            new_random_state.set_state(random_state)
        else:
            raise ValueError(
                'Do not know what to do with `random_state` Sequence %s'
                %(random_state,)
            )
        return new_random_state

    else:
        raise ValueError('Unhandled `random_state` of type %s: %s'
                         %(type(random_state), random_state))

    if jumpahead > 0:
        new_random_state.rand(jumpahead)

    return new_random_state


def test_get_random_state():
    raise NotImplementedError()


if __name__ == '__main__':
    from pisa.utils.log import logging, set_verbosity
    set_verbosity(3)
    test_get_random_state()