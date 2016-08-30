import time

from line_profiler import LineProfiler

from pisa.utils.log import logging, set_verbosity, tprofile


class log():
    """Class to redirect output into our logging stream."""
    def write(self, string):
        string = string.rstrip('\n')
        if string:
            tprofile.debug(string)
    def flush():
        pass


tlog = log()
"""Instance of a global timing logger"""

line_profiler = LineProfiler()
"""Instance of a global LineProfiler"""


def line_profile(func):
    """@line_profile decorator"""
    def profiled_func(*args, **kwargs):
        try:
            line_profiler.enable_by_count()
            line_profiler.add_function(func)
            return func(*args, **kwargs)
        finally:
            line_profiler.disable_by_count()
            # only print if it is the outermost function
            if line_profiler.functions[0] == func:
                line_profiler.print_stats(stream=tlog)
    return profiled_func


def profile(func):
    """@profile decorator"""
    def profiled_func(*args, **kwargs):
        try:
            start_t = time.time()
            return func(*args, **kwargs)
        finally:
            end_t = time.time()
            tprofile.debug('module %s, function %s: %.4f ms'
                           %(func.__module__, func.__name__,
                             (end_t - start_t) * 1000))
    return profiled_func


def test_profile():
    @profile
    def get_number():
        logging.debug('hello, i am get_number')
        for x in xrange(500000):
            yield x

    @profile
    def expensive_function():
        logging.debug('hello, i am expensive fun')
        for x in get_number():
            i = x ^ x ^ x
        return 'some result!'

    result = expensive_function()
    logging.info('<< PASSED : test_profile >>')


def test_line_profile():
    @line_profile
    def get_number():
        logging.debug('hello, i am get_number')
        for x in xrange(500000):
            yield x

    @line_profile
    def expensive_function():
        logging.debug('hello, i am expensive fun')
        for x in get_number():
            i = x ^ x ^ x
        return 'some result!'

    result = expensive_function()
    logging.info('<< PASSED : test_line_profile >>')


if __name__ == '__main__':
    set_verbosity(3)
    test_profile()
    test_line_profile()
