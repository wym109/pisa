#
# Some utilities for putting llh/llr data into and out of pandas Data
# Frame objects
#

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pisa.utils.log import logging


def get_llr_data_frames(llh_data):
    """
    Loads data from llh_data, in a dict-like format into data frames
    that are more agile to work with.

    Returns two data frames: the 4 LLR distributions from the data
    sets at the MC true fiducial model parameters, as well as 4 LLR
    distributions from the data sets taken at the best fit for the
    alternative hierarchy hypothesis (if they exist).
    """

    true_h_name_layer2 = 'true_h_fiducial'
    false_h_name_layer2 = 'false_h_best_fit'

    true_h_df = []
    false_h_df = []
    layer1 = ['true_NH','true_IH']
    layer2 = [true_h_name_layer2, false_h_name_layer2]
    layer3 = ['hypo_NH', 'hypo_IH']
    for key1 in layer1:
        for key2 in layer2:
            # Check if layer2 key in keys:
            if key2 not in llh_data[key1].keys(): continue

            for key3 in layer3:

                # Shortcut for writing this all:
                curr_dict = llh_data[key1][key2][key3]
                final_keys = curr_dict.keys()
                entries = len(curr_dict[final_keys[0]])

                data = { key: np.array(curr_dict[key]) for key in final_keys }
                data['mctrue'] = np.empty_like(data[final_keys[0]], dtype='|S16')
                data['mctrue'][:] = key1
                data['hypo'] = np.empty_like(data[final_keys[0]], dtype='|S16')
                data['hypo'][:] = key3

                df = DataFrame(data)
                if key2 == true_h_name_layer2: true_h_df.append(df)
                elif key2 == false_h_name_layer2: false_h_df.append(df)
                else:
                    raise Exception("Can't find key2: %s in list of good keys: %s"
                                    %(key2,layer2))

    # It is possible for one of these lists to return empty, if the
    # llh_data dict did not contain one of the layer2 keys
    return true_h_df, false_h_df

def get_data_frames(llh_dict,top_keys=['true_NH','true_IH']):
    """
    Loads data from llh_dict data into a data frame for each
    combination of 'true_data | hypo'
    """

    data_frames = []
    for tFlag in top_keys:
        for hFlag in ['hypo_NH','hypo_IH']:

            keys = llh_dict[tFlag][hFlag].keys()
            entries = len(llh_dict[tFlag][hFlag][keys[0]])

            data = {key: np.array(llh_dict[tFlag][hFlag][key]) for key in keys }
            data['mctrue'] = np.empty_like(data[keys[0]],dtype='|S16')
            data['mctrue'][:] = tFlag
            data['hypo'] = np.empty_like(data[keys[0]],dtype='|S16')
            data['hypo'][:] = hFlag

            df = DataFrame(data)
            data_frames.append(df)

    return data_frames

def show_frame(df,nrows=20):
    """
    Shows all columns of data frame, no matter how large it is. Number
    of rows to show is configurable.
    """
    pd.set_option('display.max_columns', len(df))
    pd.set_option('expand_frame_repr', False)
    pd.set_option('max_rows',nrows)
    logging.debug("df:\n%s"%df)

    return

def get_llh_ratios(df_frames):
    """
    IMPORTANT: expects df_frames to be ordered according to the output
    of get_data_frames. That is:
    [true_NH/hypo_NH, true_NH/hypo_IH, true_IH/hypo_NH, true_IH/hypo_IH]
    """
    # Apply this function to columns of data frame to get LLR:
    get_llh_ratio = lambda hIMH,hNMH: -(hIMH['llh'] - hNMH['llh'])
    llr_dNMH = get_llh_ratio(df_frames[1], df_frames[0])
    llr_dIMH = get_llh_ratio(df_frames[3], df_frames[2])

    llr_dict = {'true_NH':llr_dNMH, 'true_IH': llr_dIMH}

    return llr_dict
