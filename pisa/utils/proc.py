#
# proc.py
#
# A set of utilities for common processing tasks in
# the different stages.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-05-19

import inspect
from pisa.utils.log import logging

def get_params():
    '''
    Get the parameter names and values of the calling function
    Ignores any non-simple parameters (classes, dicts, etc...)
    '''
    #Get the frame of the calling function
    frame = inspect.stack()[1][0]
    #Get list of arguments and values dict, ignoring *varargs and **kwargs
    args, _, _, values = inspect.getargvalues(frame)
    #Select only types with primitive values
    ptypes =  (int, float, bool, str, type(None))
    args = [ arg for arg in args if type(values[arg]) in ptypes]
    values = [ values[arg] for arg in args]
    return dict(zip(args,values))

def report_params(params,units):
    '''
    Print the parameter values with units
    '''
    formatter = {int: '%i',
               float: '%.4e',
               str: '%s',
               bool: '%s',
               type(None): '%s'}
    #Print everything - must be sorted
    forms = [formatter[type(v)] for k,v in sorted(params.items())] 
    for key, form, unit in zip(sorted(params), forms, units):
        logging.debug(('%20s: '+form+' %s')%(key,params[key],unit))

def add_params(setA,setB):
    '''
    Join the parameters in setA and setB,
    making sure that no parameters are overwritten
    '''
    #check for overlap
    if any(p in setA for p in setB):
        pnames = set(setA.keys()) & set(setB.keys())
        logging.error('Trying to store parameter(s) %s twice'%pnames)
        raise KeyError('Trying to store parameter(s) %s twice'%pnames)

    #Otherwise append
    return dict(setA.items() + setB.items())

