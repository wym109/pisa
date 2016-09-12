import numpy as np
from sys import exit

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def mergeDictionaries(genie_dict, nugen_dict, nugen_ccfactor, nugen_ncfactor, weight_keys, break_energy):
    # First correct NUGEN with a factor
    for weight_key in ['weight'] + weight_keys:
        nugen_dict['CC'][weight_key] /= nugen_ccfactor
        nugen_dict['NC'][weight_key] /= nugen_ncfactor

    skip_keys = ['GENIE_W']

    # Then merge the keys:
    new_dict = {}
    for interaction_key in genie_dict.keys():
        new_dict[interaction_key] = {}
        for event_info_key in genie_dict[interaction_key].keys():
            if event_info_key in skip_keys:
                continue
            genie_bool = genie_dict[interaction_key]['energy'] < break_energy
            nugen_bool = nugen_dict[interaction_key]['energy'] > break_energy
            new_dict[interaction_key][event_info_key]=\
                np.concatenate((genie_dict[interaction_key][event_info_key][genie_bool], 
                                nugen_dict[interaction_key][event_info_key][nugen_bool]),
                               axis = 0)
    
    return new_dict

def checkDictParams(default_parameters, user_parameters, warning_origin = None):
    wrong_keys = False
    user_keys = user_parameters.keys()
    for one_key in user_keys:
        if not default_parameters.has_key(one_key) and one_key!="sin_sq_th23":
            print warning_origin, ': Setting ', one_key, ' was given but is not used. Did you write it right?'
            wrong_keys = True

    if wrong_keys:
        print 'Check the input parameters before continuing!'
        exit()

    return    

def addStatisticalFluctuations(exp_histo):
    for bin_i in range(exp_histo.shape[0]):
        for bin_j in range(exp_histo.shape[1]):
            exp_histo[bin_i, bin_j] = np.random.poisson(exp_histo[bin_i, bin_j])
    return exp_histo

# def find_nearest(array,value):
#     idx = (abs(array-value)).argmin()
#     return idx

# def getOscTableIndices(nu_list, osc_table):
#     energy_indices = np.zeros(len(nu_list['energy']))
#     zenith_indices = np.zeros(len(nu_list['energy']))
#     for index in range(len(nu_list['energy'])):
#         energy_indices[index] = find_nearest(osc_table['energy_list'], nu_list['energy'][index])
#         zenith_indices[index] = find_nearest(osc_table['zenith_list'], nu_list['zenith'][index])

#     return {'oscTableBin_energy':energy_indices, 'oscTableBin_zenith':zenith_indices}
