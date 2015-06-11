#
# params_nutau.py
#
# Functions augmenting those in params.py for accessing and formatting
# parameters taken from .json template settings file for analysis.
#
# author: Feifei Huang
#         fxh140@psu.edu
#
# date:   2015-06-11
#

import pisa.utils.params as PAR

def select_hierarchy_and_nutau_norm(params, normal_hierarchy, nutau_norm_value):
    ''' Select one hierarchy and change the value of nutau_norm (for the pseudo settings file)'''
    newparams = PAR.select_hierarchy(params, normal_hierarchy)
    newparams["nutau_norm"]["value"] = nutau_norm_value
    return newparams


def change_nutau_norm_settings(params, nutau_norm_val, nutau_norm_fix):
    ''' Change the value of nutau_norm["value"] and ["fixed"] (for the template settings file)'''
    if not isinstance(nutau_norm_fix, bool):
                raise ValueError('nutau_norm_fix must be boolean value')
    newparams = PAR.select_hierarchy(params, True)
    newparams["nutau_norm"]["value"] = nutau_norm_val
    newparams["nutau_norm"]["fixed"] = nutau_norm_fix
    return newparams
