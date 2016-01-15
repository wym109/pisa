#!/usr/bin/env python
# 
# DataProcParams class for importing, working with, and storing data processing
# parameters (e.g., PINGU's V5 processing).
#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#


import os
import h5py

import pisa.utils.jsons as jsons
import pisa.utils.flavInt as flavInt
import pisa.resources.resources as resources

# Note that the form of the numpy import is intentional, so that cuts -- which
# are exectuted with `eval` -- will have access to numpy's namespace without
# explicit reference to numpy. It's a hack, but it works well.
from numpy import *


class DataProcParams(dict):
    """Class for importing, working with, and storing data processing
    parameters.
    
    Implements cutting and particle identification (PID) functionality that can
    be applied to MC/data that have the specified verion of processing applied
    to it.

    Parameters
    ----------
    data_proc_params : string or dict
        If string: looks for the corresponding JSON resource (file) and loads
          the contents as a data_proc_params dict
        If dict: taken to be data_proc_params dict
        The data_proc_params dict must follow the format described below.
    detector : string
        Converted to lower-case string which must be a detector key in
        data_proc_params dict
    proc_ver
        Converted to lower-case string which must be a proc_ver key in
        data_proc_params dict

    Notes
    -----
    All information describing the processing version is loaded from a JSON
    file with the following defined format:
    
    Note that the following common cuts are defined in this class and so
    needn't be defined in the JSON file:
                      '1' : Select particles
                     '-1' : Select anti-particles
                     'cc' : Select charged-current (CC) interactions
                     'nc' : Select neutral-current (NC) interactions
       'true_upgoing_zen' : Select true-upgoing events by zenith angle
    'true_upgoing_coszen' : Select true-upgoing events by cos(zenith) angle

    data_proc_params dictionary format (and same for corresponding JSON file):

    {
      "<lower-case detector name>": {
        "<lower-case processing version>": {
          "field_map": {
            "run": "<HDF5 path to corresponding node>",
            "nu_code": "<HDF5 path to corresponding node>",
            "true_energy": "<HDF5 path to corresponding node>",
            "true_zenith": "<HDF5 path to corresponding node>",
            "reco_energy": "<HDF5 path to corresponding node>",
            "reco_zenith": "<HDF5 path to corresponding node>",
            "one_weight": "<HDF5 path to corresponding node>",
            "generator_volume": "<HDF5 path to corresponding node>",
            "generator_radius": "<HDF5 path to corresponding node>",
            "detection_length": "<HDF5 path to corresponding node>",
            "interaction_type": "<HDF5 path to corresponding node>",
            "azimuth_min": "<HDF5 path to corresponding node>",
            "azimuth_max": "<HDF5 path to corresponding node>",
            "zenith_min": "<HDF5 path to corresponding node>",
            "zenith_max": "<HDF5 path to corresponding node>",
            "energy_log_min": "<HDF5 path to corresponding node>",
            "energy_log_max": "<HDF5 path to corresponding node>",
            "num_events_per_file": "<HDF5 path to corresponding node>",
            "sim_spectral_index": "<HDF5 path to corresponding node>",
            "pid": "<HDF5 path to corresponding node>",
            "cuts_step_1": "<HDF5 path to corresponding node>",
            "cuts_step_2": "<HDF5 path to corresponding node>"
        },
        # OPTIONAL: mapping from file's nu code to PDG nu codes>
        "nu_code_to_pdg_map": {
          "<source nue code>":        12,
          "<source nue_bar code>":   -12,
          "<source numu code>":       14,
          "<source numu_bar code>":  -14,
          "<source nutau code>":      16,
          "<source nutau_bar code>": -16
        },
        "cuts": {
          "bgrej": {
            "fields": ["<field1>", "<field2>", ... ],
            "pass_if": "<expression>"
          },
          "analysis": {
            "fields": ["<field1>", "<field2>", ... ],
            "pass_if": "<expression>"
          },
          "<cut name>": {
            "fields": ["<field1>", "<field2>", ... ],
            "pass_if": "<expression>"
          }
        },
        "pid": {
          "<particle name 1>": {
            "field": [<field1>, <field2>, ...],
            "criteria": "<expression>"
          }
          "<particle name 2>": {
            "field": [<field1>, <field2>, ...],
            "criteria": "<expression>"
          }
        }
      }
    }

    Note that cuts "pass_if" and pid "criteria" expressions can make use of the
    numpy namespace and have access to any columns extracted from the source
    HDF5 file, by the standardized names given in the "field_map". For example,
    if the following "fields" are specified for a cut in the data_proc_params
    dict:
        ["cuts_step_1", "cuts_step_2"]
    then the following is a valid "pass_if" expression:
        "(reco_zenith > pi/2) & (cuts_step_1 == 1) & (cuts_step_2 == 1)"
    """
    def __init__(self, data_proc_params, detector, proc_ver):
        if isinstance(data_proc_params, basestring):
            ps = jsons.from_json(resources.find_resource(data_proc_params))
        elif isinstance(data_proc_params, dict):
            ps = data_proc_params
        else:
            raise TypeError('Unhandled data_proc_params type passed in arg: ' +
                            type(data_proc_params))
        self.detector = detector.lower()
        self.proc_ver = str(proc_ver).lower()
        ps = ps[self.detector][self.proc_ver]
        self.update(ps)

        self.trans_nu_code = False
        if self.has_key('nu_code_to_pdg_map'):
            self.trans_nu_code = True
            try:
                self.nu_code_to_pdg_map = {
                    int(code): pdg
                    for code,pdg in self['nu_code_to_pdg_map'].items()
                }
            except:
                self.nu_code_to_pdg_map = self['nu_code_to_pdg_map']

        # NOTE: the keys are strings so the particular string formatting is
        # important for indexing into the dict!
        
        # Add generic cuts
        self['cuts'].update({
            # Cut for particles only (no anti-particles)
            str(flavInt.NuFlav(12).barNoBar()).lower():
                {'fields': ['nu_code'], 'pass_if': 'nu_code > 0'},
            # Cut for anti-particles only (no particles)
            str(flavInt.NuFlav(-12).barNoBar()).lower():
                {'fields': ['nu_code'], 'pass_if': 'nu_code < 0'},
            # Cut for charged-current interactions only
            str(flavInt.IntType('cc')).lower():
                {'fields': ['interaction_type'],
                 'pass_if': 'interaction_type == 1'},
            # Cut for neutral-current interactions only
            str(flavInt.IntType('nc')).lower():
                {'fields': ['interaction_type'],
                 'pass_if': 'interaction_type == 2'},
            # True-upgoing cut usinng the zenith field
            'true_upgoing_zen':
                {'fields': ['true_zenith'], 'pass_if': 'true_zenith > pi/2'},
            # True-upgoing cut usinng the cosine-zenith field
            'true_upgoing_coszen':
                {'fields': ['true_zenith'], 'pass_if': 'true_coszen < 0'},
        })

        # Enforce rules on cuts:
        self.validateCutSpec(self['cuts'])

    @staticmethod
    def validateCutSpec(cuts):
        for cutname, cutspec in cuts.iteritems():
            # Cut names lower case
            assert isinstance(cutname, basestring)
            assert cutname == cutname.lower()
            # Has appropriate keys (and no extra)
            assert len(cutspec) == 2
            assert cutspec.has_key('fields')
            assert cutspec.has_key('pass_if')
            assert not isinstance(cutspec['fields'], basestring)
            # 'fields' contains a sequence
            assert hasattr(cutspec['fields'], '__iter__') and \
                    not isinstance(cutspec['fields'], basestring)
            # 'pass_if' contains a string
            assert isinstance(cutspec['pass_if'], basestring)

    @staticmethod
    def retrieveNodeData(h5group, address):
        sg = h5group
        for sub_addy in address.split('/'):
            sg = sg[sub_addy]
        return sg

    @staticmethod
    def populateGNS(h5group, field_map):
        for var, address in field_map.items():
            globals()[var] = retrieveNodeData(h5group, address)
    
    @staticmethod
    def cutBoolIdx(h5group, cut_fields, cut_pass_if):
        populateGNS(h5group, cut_fields)
        bool_idx = eval(cut_pass_if)
        return bool_idx

    def getData(self, h5, run_settings=None, flav=None):
        myfile = False
        try:
            if isinstance(h5, basestring):
                myfile = True
                h5 = h5py.File(os.path.expandvars(os.path.expanduser(h5)),
                               mode='r')
            data = {name:self.retrieveNodeData(h5,path)
                    for name, path in self['field_map'].items()}
        finally:
            if myfile and isinstance(h5, h5py.File):
                try:
                    h5.close()
                except:
                    pass
        self.interpretData(data)
        # TODO: enable consistency checks here & implement in run_settings
        #if run_settings is not None:
        #    run_settings.consistencyChecks(data, flav=flav)
        return data

    def interpretData(self, data):
        if self.trans_nu_code:
            data['nu_code'] = [
                self.nu_code_to_pdg_map[code] for code in data['nu_code']
            ]
        data['true_coszen'] = cos(data['true_zenith'])
        data['reco_coszen'] = cos(data['reco_zenith'])

    def applyCuts(self, data, cuts, boolean_op='&', return_fields=None):
        if isinstance(cuts, basestring) or isinstance(cuts, dict):
            cuts = [cuts]

        cut_strings = set()
        cut_fields = set()
        for cut in cuts:
            if isinstance(cut, dict):
                self.validateCutSpec(cut)
            elif self['cuts'].has_key(cut.lower()):
                cut = self['cuts'][cut.lower()]
            else:
                raise Exception('Unrecognized or invalid cut: "'+str(cut)+'"')
            cut_strings.add(cut['pass_if'])
            cut_fields.update(cut['fields'])

        # Combine cut criteria strings together with boolean operation
        cut_string = boolean_op.join(['('+cs+')' for cs in cut_strings])

        # Load the fields necessary for the cut into the global namespace
        for f in set(cut_fields):
            globals()[f] = data[f]

        # Evaluate cuts, returning a boolean array
        bool_idx = eval(cut_string)

        # Default is to return all fields
        if return_fields is None:
            return_fields = self['field_map'].keys()

        # Return specified (or all) fields, indexed by boolean array
        return {f:array(data[f])[bool_idx] for f in return_fields}

    # TODO: Implement similar method to `applyCuts` but for performing PID
