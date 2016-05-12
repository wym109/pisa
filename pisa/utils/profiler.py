from pisa.utils.log import logging, tprofile

try:
    # if LineProfiler available
    from line_profiler import LineProfiler
    profiler = LineProfiler()
    # to use simple timer add silly import that breaks
    #import sdfghj

    # to redirect into our logging stream 
    class log():
        def write(self, bla):
            bla = bla.rstrip('\n')
            if bla:
                tprofile.debug(bla)
        def flush():
            pass
    tlog = log()

    # @profile decorator
    def profile(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler.enable_by_count()
                profiler.add_function(func)
                return func(*args, **kwargs)
            finally:
                profiler.disable_by_count()
                # only print if it is the outermost function
                if profiler.functions[0] == func:
                    profiler.print_stats(stream=tlog)
        return profiled_func

except ImportError:
    # just use ordinary timer
    import time

    # @profile decorator
    def profile(func):
        def profiled_func(*args, **kwargs):
            try:
                start_t = time.time()
                return func(*args, **kwargs)
            finally:
                end_t = time.time()
                tprofile.debug('module %s, function %s: %.4f ms'%(func.__module__,func.__name__,(end_t - start_t) * 1000))
        return profiled_func

if __name__ == '__main__':

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
