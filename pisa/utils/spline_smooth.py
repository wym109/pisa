import numpy as np
from scipy.interpolate import splrep, splev

def spline_smooth(array, spline_binning, eval_binning, axis=0, smooth_factor=50, k=3,errors=None):
    '''
    smooth in slices along an axis
    assuming 2d arrays
    with binning `spline_binning` of axis
    return array with `eval_binning` of axis
    could be expanded to nd arrays to generalize it
    '''
    # only working for 2d right now!
    assert(array.ndim == 2, 'cannot do other dimensions than 2 right now, sorry')
    # points at which to evaluate splines
    spline_points = spline_binning.midpoints
    if axis == 1:
        # always smooth along axis=0
        array = array.T
        if errors is not None:
            errors[errors == 0] = errors[errors != 0].min()
            errors = 1./errors
            errors = errors.T

    smoothed_slices = []
    for index in range(array.shape[1]):
        # take slice
        h_slice = array[:,index]
        h_errors = None if errors is None else errors[:,index]
        # Fit spline to slices
        slice_spline = splrep(
            spline_points, h_slice, h_errors,
            k=k, s=smooth_factor,
        )
        # eval spline over midpoints
        smoothed_slice = splev(eval_binning.midpoints, slice_spline)
        # Assert that there are no nan or inf values in smoothed cz-slice
        assert np.all(np.isfinite(smoothed_slice))
        smoothed_slices.append(smoothed_slice)

    # Convert list of slices to array
    smoothed_array = np.array(smoothed_slices)
    # swap axes back if necessary
    if axis == 0:
        smoothed_array = smoothed_array.T

    return smoothed_array
