#!/usr/bin/env python
# 
# DataProcParams class for importing, working with, and storing data processing
# parameters (e.g., PINGU V5 processing)
#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#


import os
import h5py

from pisa.utils import jsons
import pisa.utils.flavInt as FI
from pisa.resources import resources as res

# Note that the form of the numpy import is intentional, so that cuts -- which
# are exectuted with `eval` -- will have access to numpy's namespace without
# explicit reference to numpy. It's a hack, but it works well.
from numpy import *


class DataProcParams(dict):
    def __init__(self, proc_settings, detector, proc_ver):
        if isinstance(proc_settings, basestring):
            ps = jsons.from_json(res.find_resource(proc_settings))
        elif isinstance(proc_settings, dict):
            ps = proc_settings
        else:
            raise TypeError('Unhandled proc_settings type passed in arg: ' +
                            type(proc_settings))
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
            str(FI.NuFlav(12).barNoBar()).lower():
                {'fields': ['nu_code'], 'pass_if': 'nu_code > 0'},
            # Cut for anti-particles only (no particles)
            str(FI.NuFlav(-12).barNoBar()).lower():
                {'fields': ['nu_code'], 'pass_if': 'nu_code < 0'},
            # Cut for charged-current interactions only
            str(FI.IntType('cc')).lower():
                {'fields': ['interaction_type'], 'pass_if': 'interaction_type == 1'},
            # Cut for neutral-current interactions only
            str(FI.IntType('nc')).lower():
                {'fields': ['interaction_type'], 'pass_if': 'interaction_type == 2'},
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
