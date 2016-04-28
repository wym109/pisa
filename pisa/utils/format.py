import numbers
import numpy as np
import itertools

def hrlist_formatter(start, end, step):
    """Format a range (sequence) in a simple and human-readable format.

    Parameters
    ----------
    start, end, step : numeric

    Notes
    -----
    If `start` and `end` are integers and `step` is 1, step size is omitted.

    The format does NOT follow Python's slicing syntax, in part because the
    interpretation is meant to differ; e.g.,
        '0-10:2' includes both 0 and 10 with step size of 2
    whereas
        0:10:2 (slicing syntax) excludes 10

    Numbers are converted to integers if they are equivalent for more compact
    display.

    Examples
    --------
    >>> hrlist_formatter(start=0, end=10, step=1)
    '0-10'
    >>>> hrlist_formatter(start=0, end=10, step=2)
    '0-10:2'
    >>>> hrlist_formatter(start=0, end=3, step=8)
    '0-3:8'
    >>>> hrlist_formatter(start=0.1, end=3.1, step=1.0)
    '0.1-3.1:1'
    """
    if int(start) == start:
        start = int(start)
    if int(end) == end:
        end = int(end)
    if int(step) == step:
        step = int(step)
    if int(start) == start and int(end) == end and step == 1:
        return '{}-{}'.format(start, end)
    return '{}-{}:{}'.format(start, end, step)


def list2hrlist(lst):
    """Convert a list of numbers to a compact and human-readable string.

    Adapted to make scientific notation work correctly from Scott B's
    adaptation to Python 2 of Rik Poggi's answer to his question
    stackoverflow.com/questions/9847601/convert-list-of-numbers-to-string-ranges

    Examples
    --------
    >>>> list2hrlist([0, 1])
    '0,1'
    >>>> list2hrlist([0, 1, 2])
    '0-2'
    >>>> utils.list2hrlist([0.1, 1.1, 2.1, 3.1])
    '0.1-3.1:1'
    """
    if isinstance(lst, numbers.Number):
        lst = [lst]
    lst = sorted(lst)
    rtol = np.finfo(float).resolution
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if not np.isclose(lst[scan + 2] - lst[scan + 1], step, rtol=rtol):
            result.append(str(lst[scan]))
            scan += 1
            continue
        for j in xrange(scan+2, n-1):
            if not np.isclose(lst[j+1] - lst[j], step, rtol=rtol):
                result.append(hrlist_formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(hrlist_formatter(lst[scan], lst[-1], step))
            return ','.join(result)
    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(itertools.imap(str, lst[scan:])))

    return ','.join(result)


if __name__ == '__main__':
    print hrlist_formatter(start=0, end=10, step=1)
    print hrlist_formatter(start=0, end=10, step=2)
    print hrlist_formatter(start=0, end=3, step=8)
    print hrlist_formatter(start=0.1, end=3.1, step=1.0)
    print list2hrlist([0, 1])
    print list2hrlist([0, 1, 2])
    print list2hrlist([0.1, 1.1, 2.1, 3.1])
