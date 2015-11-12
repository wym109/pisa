#!/usr/bin/env python
#
# Handle flavors, interactions types, "kinds" (flav+int. type), and kind groups
# in a consistent and convenient manner.
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#


import pisa.utils.fileio as FIO
import pisa.utils.flavInt as FI
from pisa.resources import resources as RES

# Following import * is intentionally done so that `eval` called in
# translateSourceDict will execute with direct access to numpy namespace
from numpy import *


# TODO: make ability to serialize instantiated MCSimRunSettings and
# DetMCSimRunsSettings objects back to JSON format from which they were
# generated

# TODO: make sure user is forced to choose run & detector here

class MCSimRunSettings(dict):
    def __init__(self, run_settings, run=None, detector=None):
        # TODO: clean up this constructor!
        #if isinstance(run_settings, basestring):
        #    rsd = jsons.from_json(RES.find_resource(run_settings))
        if isinstance(run_settings, dict):
            rsd = run_settings
        else:
            raise TypeError('Unhandled run_settings type passed in arg: ' +
                            type(run_settings))
        #if detector is not None:
        #    try:
        #        rsd = rsd[detector]
        #    except:
        #        pass
        rsd = self.translateSourceDict(rsd) #{run:self.translateSourceDict(rs) for run,rs in rsd.iteritems()}
        if not detector is None:
            detector = str(detector).lower()
        self.detector = detector
        self.run = run
        self.update(rsd)

    @staticmethod
    def translateSourceDict(d):
        d['tot_gen'] = d['num_events_per_file'] * d['num_i3_files']
        d['kinds'] = FI.NuKindGroup(d['kinds'])

        # Numeric fields are allowed to be expressions that get evaluated
        numeric_fields = ['azimuth_max',
                          'azimuth_min',
                          'energy_max',
                          'energy_min',
                          'genie_physical_factor',
                          'genie_prescale_factor',
                          'nu_to_total_fract',
                          'num_events_per_file',
                          'num_i3_files',
                          'sim_spectral_index',
                          'zenith_max',
                          'zenith_min',]
        for f in numeric_fields:
            if isinstance(d[f], basestring):
                d[f] = eval(d[f])

        return d

    def consistencyChecks(self, data, flav=None):
        pass

    def barnobarfract(self, barnobar=None, is_particle=None,
                      flav_or_kind=None):
        nargs = sum([(not barnobar is None),
                     (not is_particle is None),
                     (not flav_or_kind is None)])
        if nargs != 1:
            raise ValueError('One and only one of barnobar, is_particle, and'
                             ' flav_or_kind must be specified. Got ' +
                             str(nargs) + ' args instead.')

        if not flav_or_kind is None:
            is_particle = FI.NuKind(flav_or_kind).isParticle()
        if is_particle:
            return self['nu_to_total_fract']
        return 1 - self['nu_to_total_fract']

    def totGen(self, barnobar=None, is_particle=None, flav_or_kind=None):
        nargs = sum([(not barnobar is None),
                     (not is_particle is None),
                     (not flav_or_kind is None)])
        if nargs == 0:
            fract = 1.0
        else:
            fract = self.barnobarfract(barnobar=barnobar,
                                       is_particle=is_particle,
                                       flav_or_kind=flav_or_kind)

        return fract * self['tot_gen']


class DetMCSimRunsSettings(dict):
    def __init__(self, run_settings, detector=None):
        if isinstance(run_settings, basestring):
            rsd = FIO.from_file(RES.find_resource(run_settings))
        elif isinstance(run_settings, dict):
            rsd = run_settings
        else:
            raise TypeError('Unhandled run_settings type passed in arg: ' +
                            type(run_settings))

        if not detector is None:
            detector = str(detector).lower()
        self.detector = detector

        # Determine how deeply nested runs are in the dict to allow for
        # user to specify a dict that has multiple detectors in it OR
        # a dict with just a single detector in it
        if rsd.values()[0].has_key('kinds'):
            runs_d = rsd
        elif rsd.values()[0].values()[0].has_key('kinds'):
            if self.detector is None:
                if len(rsd) == 1:
                    runs_d = rsd.values()[0]
                else:
                    raise ValueError('Must specify which detector; detectors '
                                     'found: ' + str(rsd.keys()))
            else:
                runs_d = rsd[self.detector.lower()]
        else:
            raise Exception('dict must either be 3 levels: '
                            '{DET:{RUN:{...}}}; or 2 levels: {RUN:{...}}')

        # Convert the run numbers to integers (JSON files cannot have an int as
        # a key, so it is a string upon import) and convert actual run settings
        # dict to MCSimRunSettings instances
        runs_d = {int(k): MCSimRunSettings(v) for k,v in runs_d.iteritems()}

        # Save the runs_d to this object instance, which behaves like a dict
        self.update(runs_d)

    def consistencyChecks(self, data, run, flav=None):
        pass

    def barnobarfract(self, run, *args, **kwargs):
        return self[run].barnobarfract(*args, **kwargs)

    def totGen(self, run, *args, **kwargs): 
        return self[run].totGen(*args, **kwargs)
