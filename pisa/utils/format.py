

def engfmt(n, units, sigfigs=3, decimals=None, sign_always=False):
    """Format number as string in engineering format (10^(multiples-of-three)),
    including the most common metric prefixes (from atto to Exa).

    Parameters
    ----------
    n : numeric
        Number to be formatted
    units : str
        A string that suffixes the output (separated by a space)
    sigfigs : int
        Number of significant figures to limit the result to
    decimals : int or None
        Number of decimals to display (zeros filled out as necessary)
    sign_always : bool
        Prefix the number with "+" sign if number is positive; otherwise,
        only negative numbers are prefixed with a sign ("-")
    """
    prefixes = {-18:'a', -15:'f', -12:'p', -9:'n', -6:'u', -3:'m', 0:'',
                3:'k', 6:'M', 9:'G', 12:'T', 15:'P', 18:'E'}
    # Logs don't like negative numbers...
    sign = np.sign(n)
    n *= sign

    mag = int(np.floor(np.log10(n)))
    pfx_mag = int(np.floor(np.log10(n)/3.0)*3)

    if decimals is None:
        decimals = sigfigs-1 - (mag-pfx_mag)

    round_to = decimals
    if sigfigs is not None:
        round_to = sigfigs-1 - (mag-pfx_mag)

    scaled_rounded = np.round(n/10.0**pfx_mag, round_to)

    sign_str = ''
    if sign_always and sign > 0:
        sign_str = '+'
    num_str = sign_str + format(sign*scaled_rounded, '.'+str(decimals)+'f')

    if pfx_mag not in prefixes:
        return num_str + 'e'+str(mag) + ' ' + units
    return  num_str + ' ' + prefixes[pfx_mag] + units


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



