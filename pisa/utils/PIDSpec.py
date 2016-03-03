#!/usr/bin/env python
#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 2, 2016
"""
PIDSpec class for importing, working with, and storing PID specifications.
"""


import os
import collections
import h5py

import pisa.utils.jsons as jsons
import pisa.utils.flavInt as flavInt
import pisa.resources.resources as resources

# Note that the form of the numpy import is intentional, so that cuts -- which
# are exectuted with `eval` -- will have access to numpy's namespace without
# explicit reference to numpy. It's a hack, but it works well.
import numpy as np
from numpy import *


class PIDSpec(object):
    """Class for importing, applying, and storing particle identification (PID)
    specifications.

    Parameters
    ----------
    pid_specs : string or dict
        Specifications of PID categorization criteria and the fields required
        to apply those criteria
    detector, geom, proc_ver, pid_spec_ver : strings
        Detector name, version of detector's geometry, processing version
        applied, and PID specification version to use. These must all be found
        in the `pid_specs` file/dict.

    Notes
    -----
    All information describing the PID specifications is loaded from a
    file with the following defined format:

        {
          "<detector name>": {
            "<geometry version>": {
              "<processing version>": {
                "<pid version>": {
                  "field_map": {
                    "<field1>": "<path to field in I3 file>",
                    "<field2>": "<path to field in I3 file>",
                    ...
                  },
                  "criteria": {
                    "<signature 1>": "<criteria>",
                    "<signature 2>": "<criteria>",
                  }
                }
              }
            }
          }
        }
    """
    def __init__(self, pid_specs, detector, geom, proc_ver, pid_spec_ver=1):
        if isinstance(pid_specs, basestring):
            pid_specs = jsons.from_json(resources.find_resource(pid_specs))
        elif isinstance(pid_specs, collections.Mapping):
            pass
        else:
            raise TypeError('Unhandled `pid_specs` type: "%s"' %
                            type(data_proc_params))
        self.detector = detector
        self.proc_ver = proc_ver
        self.pid_spec_ver = str(pid_spec_ver)
        self.pid_spec = ps[detector][geom][proc_ver][pid_spec_ver]

        # Enforce rules on PID spec:
        self.validatePIDSpec(self.pid_spec)

    @staticmethod
    def validatePIDSpec(pids):
        """Validate a PID specification"""
        # TODO: implement validation
        #for signature, pidspec in pids.iteritems():
        #    # Particle names are lower-case strings with no surrounding
        #    # whitespace
        #    assert isinstance(signature, basestring)
        #    assert signature == signature.lower()
        #    assert signature == signature.strip()
        #    # Has appropriate keys (and no extra)
        #    assert len(pidspec) == 2
        #    assert pidspec.has_key('fields')
        #    assert pidspec.has_key('criteria')
        #    assert not isinstance(pidspec['fields'], basestring)
        #    # 'fields' contains a sequence
        #    assert hasattr(pidspec['fields'], '__iter__') and \
        #            not isinstance(pidspec['fields'], basestring)
        #    # 'criteria' contains a string
        #    assert isinstance(pidspec['criteria'], basestring)
        return

    def applyPID(self, events, signatures=None, return_fields=None):
        """Divide events in `events` that fall into specified PID signature(s).

        Parameters
        ----------
        events : Events object or convertible thereto
        signatures : string, sequence of strings, or None
            Signatures to return events for.
            If string or sequence of strings, return events for the specified
            signature(s); if None, use all signatures in the PID specification.
        return_fields : sequence of strings or None

        Returns
        -------
        Separated events FlavIntData object with each flavint leaf node
        populated by a dictionary formatted as follows:
        {
          '<signature1>': {
            '<return_field1>': array,
            '<return_field2>': array,
             ...
          },
          ...
        }
        """
        # Interpret `signatures`
        if isinstance(signatures, basestring):
            signatures = [signatures]
        elif signatures is None:
            signatures = self.pid_spec['criteria'].keys()

        # Validate signatures
        for signature in signatures:
            assert isinstance(signature, basestring), \
                    'Signature "%s" is not string.' % signature
            if signature not in self.pid_spec['criteria']:
                raise ValueError(
                    'Invalid signature "%s": Not avilable in PID specification;'
                    ' valid signatures are: %s' %
                    (signature, ', '.join(['"%s"'%vn for vn in valid_names]))
                )

        if isinstance(return_fields, basestring):
            return_fields = [return_fields]

        separated_events = flavInt.FlavIntData()
        # Outer loop is over flavint, so that same data is processed multiply
        # for different PID specs, rather than new data loaded for each PID
        # spec. (Theoretically faster.)
        for flavint in events.flavints():
            src = events[flavint]
            dest = separated_events[flavint] = {}
            if return_fields is None:
                fields_to_get = sorted(src.keys())
            else:
                fields_to_get = return_fields

            if len(fields_to_get) == 0:
                raise ValueError('No fields to get.')

            for signature in signatures:
                dest[signature] = {}
                pid_criteria = self.pid_spec['criteria'][signature]

                # Load the fields into global namespace that are necessary to
                # apply PID criteria
                for field in sorted(pid['field_map'].keys()):
                    globals()[field] = events[field]

                # Evaluate PID criteria, returning a boolean array
                bool_idx = eval(pid_criteria)

                # Store all requested fields that meet the criteria
                for field in return_fields:
                    dest[signature][field] = src[field][bool_idx]

        return separated_events
