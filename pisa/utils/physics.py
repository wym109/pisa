#
# physics.py
#
# A set of utilities for dealing with naming and actual physics.
# 
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   2014-10-01

def get_PDG_ID(name):
    """Return the Particle Data Group Particle ID for a given particle"""
    ptcl_dict = {'nue': 12, 'nue_bar': -12, 
                 'numu': 14, 'numu_bar': -14,
                 'nutau': 16, 'nutau_bar': -16}
    return ptcl_dict[name]
