#! /usr/bin/env python
#
# Classes for working with neutrino flavors (NuFlav), interactions types
# (IntType), "kinds" (a flavor and an interaction type) (NuKind), and kind
# groups (NuKindGroup) in a consistent and convenient manner.
#
# FIData class for working with data stored by kind (flavor & interaction
# type). This should replace the PISA convention of using raw doubly-nested
# dictionaries indexed as [<flavor>][<interaction type>]. For now, FIData
# objects can be drop-in replacements for such dictionaries (they can be
# accessed and written to in the same way since FIData subclasses dict) but
# this should be deprecated; eventually, all direct access of the data
# structure should be eliminated and disallowed by the FIData object.
#
# Define convenience tuples ALL_{x} for easy iteration
#
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#

# TODO: make simpleStr() method convertible back to NuKindGroup, either by
# increasing the intelligence of interpret(), by modifying what simpleStr()
# produces, or by adding another function to interpret simple strings. (I'm
# leaning towards the second option at the moment, since I don't see how to
# make the first interpret both a simplestr AND nue as nuecc+nuenc, and I
# don't think there's a way to know "this is a simple str" vs not easily.)

import sys, traceback
from itertools import product, combinations, izip
import numpy as np
from copy import deepcopy
import re
import collections

from pisa.utils.log import logging, set_verbosity
import pisa.utils.fileio as fileio
import pisa.utils.utils as utils


global __BAR_SSEP__
__BAR_SSEP__ = ''

class BarSep():
    def __init__(self, val):
        global __BAR_SSEP__
        self.old_val = __BAR_SSEP__
        self.new_val = val

    def __enter__(self):
        global __BAR_SSEP__
        __BAR_SSEP__ = self.new_val

    def __exit__(self, type, value, traceback):
        global __BAR_SSEP__
        __BAR_SSEP__ = self.old_val
        
def set_bar_ssep(val):
    global __BAR_SSEP__
    assert isinstance(val, basestring)
    __BAR_SSEP__ = val

def get_bar_ssep():
    global __BAR_SSEP__
    return __BAR_SSEP__


def tex(x, d=False):
    if d:
        return '$' + x.tex() + '$'
    return x.tex()


class NuFlav(object):
    PART_CODE = 1
    ANTIPART_CODE = -1
    NUE_CODE = 12
    NUMU_CODE = 14
    NUTAU_CODE = 16
    NUEBAR_CODE = -12
    NUMUBAR_CODE = -14
    NUTAUBAR_CODE = -16
    TOKENS = re.compile('(nu|e|mu|tau|bar)')
    F_RE = re.compile(
        r'(?P<fullflav>(?:nue|numu|nutau)(?P<barnobar>bar){0,1})'
    )
    def __init__(self, val):
        self.fstr2code = {
            'nue': self.NUE_CODE,
            'numu': self.NUMU_CODE,
            'nutau': self.NUTAU_CODE,
            'nuebar': self.NUEBAR_CODE,
            'numubar': self.NUMUBAR_CODE,
            'nutaubar': self.NUTAUBAR_CODE
        }
        self.barnobar2code = {
            None: self.PART_CODE,
            '': self.PART_CODE,
            'bar': self.ANTIPART_CODE,
        }
        self.f2tex = {
            self.NUE_CODE:  r'{\nu_e}',
            self.NUMU_CODE:  r'{\nu_\mu}',
            self.NUTAU_CODE:  r'{\nu_\tau}',
            self.NUEBAR_CODE: r'{\bar\nu_e}',
            self.NUMUBAR_CODE: r'{\bar\nu_\mu}',
            self.NUTAUBAR_CODE: r'{\bar\nu_\tau}',
        }
        # Instantiate this neutrino flavor object by interpreting val
        orig_val = val
        try:
            if isinstance(val, basestring):
                val = val.lower()
                val = ''.join(self.TOKENS.findall(val))
                flav_dict = self.F_RE.match(val).groupdict()
                self.__flav = self.fstr2code[flav_dict['fullflav']]
                self.__barnobar = self.barnobar2code[flav_dict['barnobar']]
            elif hasattr(val, 'flavCode'):
                self.__flav = val.flavCode()
                self.__barnobar = np.sign(self.__flav)
            elif hasattr(val, 'flav'):
                self.__flav = val.flav().flavCode()
                self.__barnobar = np.sign(self.__flav)
            else:
                if val in self.fstr2code.values():
                    self.__flav = int(val)
                    self.__barnobar = np.sign(self.__flav)
                else:
                    raise ValueError('Invalid neutrino flavor/code: "%s"' %
                                     str(val))
            # Double check than flav and barnobar codes are valid
            assert self.__flav in self.fstr2code.values()
            assert self.__barnobar in self.barnobar2code.values()
        except (AssertionError, ValueError, AttributeError):
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise ValueError('Could not interpret value "' + orig_val
                             + '":\n' + '\n'.join(
                                 traceback.format_exception(exc_type,
                                                            exc_value,
                                                            exc_traceback)
                             ))

    def __str__(self):
        global __BAR_SSEP__
        fstr = [s for s,code in self.fstr2code.items() if code == self.__flav]
        fstr = fstr[0]
        fstr = fstr.replace('bar', __BAR_SSEP__+'bar')
        return fstr

    # TODO: copy, deepcopy, and JSON serialization
    #def __copy__(self):
    #    return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__flav)

    def __cmp__(self, other):
        if not isinstance(other, NuFlav):
            return 1
        c0 = self.flavCode()
        c1 = other.flavCode()
        ac0 = abs(c0)
        ac1 = abs(c1)
        if ac0 < ac1:
            return -1
        if ac0 == ac1:
            return c1 - c0
        if ac0 > ac1:
            return +1

    def __neg__(self):
        return NuFlav(self.__flav*-1)

    def __add__(self, other):
        return NuKindGroup(self, other)

    def tex(self):
        return self.f2tex[self.__flav]

    def flavCode(self):
        return self.__flav

    def barNoBar(self):
        return self.__barnobar

    def isParticle(self):
        return self.__barnobar == self.PART_CODE

    def isAntiParticle(self):
        return self.__barnobar == self.ANTIPART_CODE

    def pidx(self, d, *args):
        with BarSep('_'):
            field = d[str(self)]
        for idx in args:
            field = field[idx]
        return field


ALL_PARTICLES = (NuFlav(12), NuFlav(14), NuFlav(16))
ALL_ANTIPARTICLES = (NuFlav(-12), NuFlav(-14), NuFlav(-16))
ALL_FLAVS = tuple(sorted(list(ALL_PARTICLES) + list(ALL_ANTIPARTICLES)))


class AllNu(object):
    def __init__(self):
        self.__flav = [p for p in ALL_PARTICLES]

    def flav(self):
        return self.__flav

    def __str__(self):
        return 'nuall'
    
    def tex(self):
        return r'{\nu_{\rm all}}'


class AllNuBar(object):
    def __init__(self):
        self.__flav = [p for p in ALL_ANTIPARTICLES]

    def flav(self):
        return self.__flav

    def __str__(self):
        return 'nuallbar'
    
    def tex(self):
        return r'{\bar\nu_{\rm all}}'


class IntType(object):
    '''
    Interaction type object.

    Instantiate via
      * Numerical code: 1=CC, 2=NC
      * String (case-insensitive; all characters besides valid tokens are
        ignored)
      * Instantiated IntType object (or any method implementing intTypeCode()
        which returns a valid interaction type code)
      * Instantiated NuKind object (or any object implementing intType()
        which returns a valid IntType object)

    The following, e.g., are all interpreted as charged-current IntTypes:
      IntType('cc')
      IntType('\n\t _cc \n')
      IntType('numubarcc')
      IntType(1)
      IntType(1.0)
      IntType(IntType('cc'))
      IntType(NuKind('numubarcc'))
    '''
    CC_CODE = 1
    NC_CODE = 2
    IT_RE = re.compile(r'(cc|nc)')
    def __init__(self, val):
        self.istr2code = {
            'cc': self.CC_CODE,
            'nc': self.NC_CODE,
        }
        self.i2tex = {
            self.CC_CODE: r'{\rm CC}',
            self.NC_CODE: r'{\rm NC}'
        }
        
        # Interpret `val`
        try:
            orig_val = val
            if isinstance(val, basestring):
                val = val.lower()
                int_type = self.IT_RE.findall(val)
                if len(int_type) != 1:
                    raise ValueError('Found %d interaction type tokens'
                                     ' (can' ' only handle one).' %
                                     len(int_type))
                self.__int_type = self.istr2code[int_type[0]]
            elif hasattr(val, 'intType'):
                self.__int_type = val.intType().intTypeCode()
            elif hasattr(val, 'intTypeCode'):
                self.__int_type = val.intTypeCode()
            else:
                if val in self.istr2code.values():
                    self.__int_type = int(val)
                else:
                    raise TypeError('Unhandled type: "%s"' %
                                    str(type(int_type)))
            # Double check that the interaction type code set is valid
            assert self.__int_type in self.istr2code.values()
        except (AssertionError, TypeError, ValueError, AttributeError):
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise ValueError('Could not interpret value "%s":  %s' % (
                str(orig_val),
                '\n'.join(traceback.format_exception(
                    exc_type, exc_value, exc_traceback))
            ))

    def __str__(self):
        return [s for s,code in self.istr2code.items()
                if code == self.__int_type][0]

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__int_type)

    def __cmp__(self, other):
        if not isinstance(other, IntType):
            return 1
        return self.intTypeCode() - other.intTypeCode()

    def isCC(self):
        return self.__int_type == self.CC_CODE

    def isNC(self):
        return self.__int_type == self.NC_CODE

    def intTypeCode(self):
        return self.__int_type

    def tex(self):
        return self.i2tex[self.__int_type]


ALL_INT_TYPES = (IntType('cc'), IntType('nc'))


class NuKind(object):
    '''A neutrino "kind" encompasses both the neutrino flavor and its
    interaction type.
    
    Instantiate via
      * String containing a single flavor and a single interaction type
        e.g.: 'numucc', 'nu_mu_cc', 'nu mu CC', 'numu_bar CC', etc.
      * Another instantiated NuKind object
      * Two separate objects that can be converted to a valid NuFlav
        and a valid IntType (in that order)
      * An iterable of length two which contains such objects
      * kwargs `flav` and `int_type` specifying such objects

    String specifications simply ignore all characters not recognized as a
    valid token.
    '''
    TOKENS = re.compile('(nu|e|mu|tau|bar|nc|cc)')
    FINT_RE = re.compile(
        r'(?P<fullflav>(?:nue|numu|nutau)'
        r'(?P<barnobar>bar){0,1})'
        r'(?P<int_type>cc|nc){0,1}'
    )
    FINT_SSEP = '_'
    FINT_TEXSEP = r' \, '
    # TODO: use multiple inheritance to clean up the below
    def __init__(self, *args, **kwargs):
        if kwargs:
            if args:
                raise TypeError('Either positional or keyword args may be'
                                ' provided, but not both')
            keys = kwargs.keys()
            if len(set(keys).difference(set(('flav', 'int_type')))) != 0:
                raise TypeError('Invalid kwarg(s) specified: %s' %
                                kwargs.keys())
            flav_int = (kwargs['flav'], kwargs['int_type'])
        elif args:
            if len(args) == 0:
                raise TypeError('No kind specification provided')
            elif len(args) == 1:
                flav_int = args[0]
            elif len(args) == 2:
                flav_int = args
            elif len(args) > 2:
                raise TypeError('More than two args')

        # Initialize with string
        if isinstance(flav_int, basestring):
            orig_flav_int = flav_int
            try:
                flav_int = ''.join(self.TOKENS.findall(flav_int.lower()))
                kind_dict = self.FINT_RE.match(flav_int).groupdict()
                self.__flav = NuFlav(kind_dict['fullflav'])
                self.__int_type = IntType(kind_dict['int_type'])
            except (UnboundLocalError, ValueError, AttributeError):
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise ValueError(
                    'Could not interpret value "%s" as valid kind: %s' %
                    (str(orig_flav_int),
                     '\n'.join(traceback.format_exception(exc_type, exc_value,
                                                          exc_traceback)))
                )
        elif hasattr(flav_int, '__len__'):
            self.__flav = NuFlav(flav_int[0])
            self.__int_type = IntType(flav_int[1])
        elif isinstance(flav_int, NuKind):
            self.__flav = NuFlav(flav_int.flav())
            self.__int_type = IntType(flav_int.intTypeCode())
        else:
            raise TypeError('Unhandled type: "' + str(type(flav_int)) +
                            '"; class: "' + str(flav_int.__class__) +
                            '; value: "' + str(flav_int) + '"')

    def __str__(self):
        return self.flavStr() + self.FINT_SSEP + self.intTypeStr()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash( (self.flavCode(), self.intTypeCode()) )

    def __cmp__(self, other):
        if not isinstance(other, NuKind):
            return 1
        return cmp(
            (self.flav(), self.intType()), (other.flav(), other.intType())
        )

    def __neg__(self):
        return NuKind(-self.__flav, self.__int_type)

    def __add__(self, other):
        return NuKindGroup(self, other)

    def pidx(self, d, *args):
        with BarSep('_'):
            field = d[str(self.flav())][str(self.intType())]
        for idx in args:
            field = field[idx]
        return field

    def flav(self):
        return self.__flav

    def flavCode(self):
        return self.__flav.flavCode()

    def barNoBar(self):
        return self.__flav.barNoBar()

    def isParticle(self):
        return self.__flav.isParticle()

    def isAntiParticle(self):
        return self.__flav.isAntiParticle()

    def isCC(self):
        return self.__int_type.isCC()

    def isNC(self):
        return self.__int_type.isNC()

    def intType(self):
        return self.__int_type

    def intTypeCode(self):
        return self.__int_type.intTypeCode()

    def flavStr(self):
        return str(self.__flav)

    def intTypeStr(self):
        return str(self.__int_type)

    def flavTex(self):
        return self.__flav.tex()

    def intTypeTex(self):
        return self.__int_type.tex()

    def tex(self):
        return '{%s%s%s}' % (self.flavTex(),
                             self.FINT_TEXSEP,
                             self.intTypeTex())


class NuKindGroup(collections.MutableSequence):
    '''
    Grouping of neutrino kinds. Specification can be via
      * A single `NuFlav` object; this gets promoted to include both
        interaction types
      * A single `NuKind` object
      * String:
        * Ignores anything besides valid tokens
        * A flavor with no interaction type specified will include both CC
          and NC interaction types
        * Multiple flavor/interaction-type specifications can be made;
          use of delimiters is optional
        * Interprets "nuall" as nue+numu+nutau and "nuallbar" as
          nuebar+numubar+nutaubar
      * Iterable containing any of the above (i.e., objects convertible to
        `NuKind` objects). Note that a valid iterable is another `NuKindGroup`
        object.
    '''
    TOKENS = re.compile('(nu|e|mu|tau|all|bar|nc|cc)')
    K_RE = re.compile(r'((?:nue|numu|nutau|nuall)(?:bar){0,1}(?:cc|nc){0,2})')
    F_RE = re.compile(r'(?P<fullflav>(?:nue|numu|nutau|nuall)(?:bar){0,1})')
    def __init__(self, *args):
        self.kind_ssep = '+'
        self.__kinds = []
        # Possibly a special case if len(args) == 2, so send as a single entity
        # if this is the case
        if len(args) == 2:
            args = [args]
        [self.__iadd__(a) for a in args]

    def __add__(self, val):
        kind_list = sorted(set(self.__kinds + self.interpret(val)))
        return NuKindGroup(kind_list)

    def __iadd__(self, val):
        self.__kinds = sorted(set(self.__kinds + self.interpret(val)))
        return self

    def __delitem__(self, idx):
        self.__kinds.__delitem__(idx)

    def remove(self, val):
        '''
        Remove a kind from this group.

        `val` must be valid for the interpret() method
        '''
        kind_list = sorted(set(self.interpret(val)))
        for k in kind_list:
            try:
                idx = self.__kinds.index(k)
            except ValueError:
                pass
            else:
                del self.__kinds[idx]

    def __sub__(self, val):
        cp = deepcopy(self)
        cp.remove(val)
        return cp

    def __isub__(self, val):
        self.remove(val)
        return self

    def __setitem__(self, idx, val):
        self.__kinds[idx] = val

    def insert(self, idx, val):
        self.__kinds.insert(idx, val)

    def __cmp__(self, other):
        if not isinstance(other, NuKindGroup):
            return 1
        if len(other) != len(self):
            return len(self) - len(other)
        cmps = [cmp(mine, other[n]) for n,mine in enumerate(self.__kinds)]
        if all([c==0 for c in cmps]):
            return 0
        return [c for c in cmps if c != 0][0]

    def __contains__(self, val):
        return all([(k in self.__kinds) for k in self.interpret(val)])

    def __len__(self):
        return len(self.__kinds)

    def __getitem__(self, idx):
        return self.__kinds[idx]

    def __str__(self):
        allkg = set(self.kinds())

        # Check if nuall or nuallbar CC, NC, or both
        nuallcc, nuallbarcc, nuallnc, nuallbarnc = False, False, False, False
        ccKinds = NuKindGroup(self.ccKinds())
        ncKinds = NuKindGroup(self.ncKinds())
        if len(ccKinds.particles()) == 3:
            nuallcc = True
        if len(ccKinds.antiParticles()) == 3:
            nuallbarcc = True
        if len(ncKinds.particles()) == 3:
            nuallnc = True
        if len(ncKinds.antiParticles()) == 3:
            nuallbarnc = True

        # Construct nuall(bar) part(s) of string
        strs = []
        if nuallcc and nuallnc:
            strs.append('nuall')
            [allkg.remove(NuKind(k, 'cc')) for k in ALL_PARTICLES]
            [allkg.remove(NuKind(k, 'nc')) for k in ALL_PARTICLES]
        elif nuallcc:
            strs.append('nuall' + NuKind.FINT_SSEP + str(IntType('cc')))
            [allkg.remove(NuKind(k, 'cc')) for k in ALL_PARTICLES]
        elif nuallnc:
            strs.append('nuall' + NuKind.FINT_SSEP + str(IntType('nc')))
            [allkg.remove(NuKind(k, 'nc')) for k in ALL_PARTICLES]

        if nuallbarcc and nuallbarnc:
            strs.append('nuallbar')
            [allkg.remove(NuKind(k, 'cc')) for k in ALL_ANTIPARTICLES]
            [allkg.remove(NuKind(k, 'nc')) for k in ALL_ANTIPARTICLES]
        elif nuallbarcc:
            strs.append('nuallbar' + NuKind.FINT_SSEP + str(IntType('cc')))
            [allkg.remove(NuKind(k, 'cc')) for k in ALL_ANTIPARTICLES]
        elif nuallbarnc:
            strs.append('nuallbar' + NuKind.FINT_SSEP + str(IntType('nc')))
            [allkg.remove(NuKind(k, 'nc')) for k in ALL_ANTIPARTICLES]

        # Among remaining kinds, group by flavor and combine if both CC and NC
        # are present for individual flavors (i.e., eliminate the intType
        # string altogether)
        for flav in ALL_PARTICLES + ALL_ANTIPARTICLES:
            if flav in [k.flav() for k in allkg]:
                cc, nc = False, False
                if NuKind(flav, 'cc') in allkg:
                    cc = True
                if NuKind(flav, 'nc') in allkg:
                    nc = True
                if cc and nc:
                    strs.append(str(flav))
                    allkg.remove(NuKind(flav, 'cc'))
                    allkg.remove(NuKind(flav, 'nc'))
                elif cc:
                    strs.append(str(NuKind(flav, 'cc')))
                    allkg.remove(NuKind(flav, 'cc'))
                elif nc:
                    strs.append(str(NuKind(flav, 'nc')))
                    allkg.remove(NuKind(flav, 'nc'))
        return self.kind_ssep.join(strs)

    def __repr__(self):
        return self.__str__()

    # TODO:
    # Technically, since this is a mutable type, the __hash__ method shouldn't
    # be implemented as this will allow for "illegal" behavior, like using
    # a NuKindGroup as a key in a dict. So this should be fixed, maybe.
    #__hash__ = None
    def __hash__(self):
        return hash(tuple(self.__kinds))

    @staticmethod
    def interpret(val):
        '''
        Interpret a NuKindGroup arg
        '''
        if isinstance(val, basestring):
            orig_val = val
            try:
                kinds = []
                orig_val = val
                val = val.lower()

                # Eliminate anything besides valid tokens
                val = ''.join(NuKindGroup.TOKENS.findall(val))

                # Find all kinds specified
                allkinds_str = NuKindGroup.K_RE.findall(val)

                for kind_str in allkinds_str:
                    match = NuKindGroup.F_RE.match(kind_str)
                    flav = match.groupdict()['fullflav']

                    # A kind found above can include 'all' which is actually
                    # three different flavors
                    if 'all' in flav:
                        flavs = [flav.replace('all', x)
                                 for x in ('e','mu','tau')]
                    else:
                        flavs = [flav]

                    ints = sorted(set(IntType.IT_RE.findall(kind_str)))

                    # If kind_str does not include 'cc' or 'nc', include both
                    if len(ints) == 0:
                        ints = ['cc', 'nc']

                    # Add all combinations of (flav, int) found in this
                    # kind_str
                    kinds.extend([''.join(fi) for fi in product(flavs, ints)])

            except (ValueError, AttributeError):
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise ValueError('Could not interpret value "' + orig_val
                                 + '":\n' + '\n'.join(
                                     traceback.format_exception(exc_type,
                                                                exc_value,
                                                                exc_traceback)
                                 ))
        elif isinstance(val, NuFlav):
            kinds = [NuKind((val,'cc')), NuKind((val,'nc'))]
        elif isinstance(val, NuKind):
            kinds = [val]
        elif isinstance(val, NuKindGroup):
            kinds = list(val.kinds())
        elif np.isscalar(val):
            kinds = [val]
        elif val is None:
            kinds = []
        elif hasattr(val, '__len__'):
            kinds = []
            # Treat length-2 iterables as special case, in case the two
            # elements can form a single NuKind.            if len(val) == 2:
            if len(val) == 2:
                try_again = True
                try:
                    # Start with counter-hypothesis: that the two elements of
                    # `val` can form two valid, independent NuKinds...
                    k1 = NuKindGroup.interpret(val[0])
                    k2 = NuKindGroup.interpret(val[1])
                    if k1 and k2:
                        # Success: Two independent NuKinds were created
                        try_again = False
                        kinds.extend(k1)
                        kinds.extend(k2)
                except (UnboundLocalError, ValueError, AssertionError,
                        TypeError):
                    pass
                if try_again:
                    # If the two elements of the iterable did not form two
                    # NuKinds, try forming a single NuKind with `val`
                    kinds = [NuKind(val)]
            else: 
                # If 1 or >2 elements in `val`, make a kind out of each
                [kinds.extend(NuKindGroup.interpret(x)) for x in val]
        else:
            raise Exception('Unhandled val: ' + str(val) + ', class '
                            + str(val.__class__) + ' type ' + str(val))

        kind_list = []
        for k in kinds:
            try:
                nk = NuKind(k)
                kind_list.append(nk)
            except TypeError:
                # If NuKind failed, try NuFlav; if this fails, give up.
                flav = NuFlav(k)
                kind_list.append(NuKind((flav, 'cc')))
                kind_list.append(NuKind((flav, 'nc')))
        return kind_list

    def kinds(self):
        return tuple(self.__kinds)

    def flavs(self):
        return tuple([k.flav() for k in self.__kinds])

    def ccKinds(self):
        return tuple([k for k in self.__kinds if k.intType() == IntType('cc')])
    
    def ncKinds(self):
        return tuple([k for k in self.__kinds if k.intType() == IntType('nc')])

    def particles(self):
        return tuple([k for k in self.__kinds if k.isParticle()])
    
    def antiParticles(self):
        return tuple([k for k in self.__kinds if k.isAntiParticle()])

    def ccFlavs(self):
        return tuple([k.flav() for k in self.__kinds
                      if k.intType() == IntType('cc')])
    
    def ncFlavs(self):
        return tuple([k.flav() for k in self.__kinds
                      if k.intType() == IntType('nc')])

    def uniqueFlavs(self):
        return tuple(sorted(set([k.flav() for k in self.__kinds])))

    def groupFlavsByIntType(self):
        uniqueF = self.uniqueFlavs()
        fint_d = {f:set() for f in uniqueF}
        {fint_d[k.flav()].add(k.intType()) for k in self.kinds()}
        grouped = {
            'all_int_type_flavs': [],
            'cc_only_flavs' : [],
            'nc_only_flavs': []
        }
        for f in uniqueF:
            if len(fint_d[f]) == 2:
                grouped['all_int_type_flavs'].append(f)
            elif list(fint_d[f])[0] == IntType('cc'):
                grouped['cc_only_flavs'].append(f)
            else:
                grouped['nc_only_flavs'].append(f)
        return grouped

    def __simpleStr(self, flavsep, flavintsep, kindsep, addsep, func):
        grouped = self.groupFlavsByIntType()
        all_nu = AllNu()
        all_nubar = AllNuBar()
        for k,v in grouped.items():
            if all([f in v for f in all_nubar.flav()]):
                [grouped[k].remove(f) for f in all_nubar.flav()]
                grouped[k].insert(0, all_nubar)
            if all([f in v for f in all_nu.flav()]):
                [grouped[k].remove(f) for f in all_nu.flav()]
                grouped[k].insert(0, all_nu)
        all_s = flavsep.join([func(f) for f in grouped['all_int_type_flavs']])
        cc_only_s = flavsep.join([func(f) for f in grouped['cc_only_flavs']])
        nc_only_s = flavsep.join([func(f) for f in grouped['nc_only_flavs']])
        strs = []
        if len(all_s) > 0:
            strs.append(all_s + flavintsep + func(IntType('cc')) + addsep +
                        func(IntType('nc')))
        if len(cc_only_s) > 0:
            strs.append(cc_only_s + flavintsep + func(IntType('cc')))
        if len(nc_only_s) > 0:
            strs.append(nc_only_s + flavintsep + func(IntType('nc')))
        return kindsep.join(strs)

    def simpleStr(self, flavsep='+', flavintsep=' ', kindsep=', ', addsep='+'):
        return self.__simpleStr(flavsep=flavsep, flavintsep=flavintsep,
                                kindsep=kindsep, addsep=addsep, func=str)

    def fileStr(self, flavsep='_', flavintsep='_', kindsep='__', addsep=''):
        return self.__simpleStr(flavsep=flavsep, flavintsep=flavintsep,
                                kindsep=kindsep, addsep=addsep, func=str)

    def simpleTex(self, flavsep=r', \, ', flavintsep=r' \, ', kindsep=r'; \; ',
                  addsep=r'+'):
        return self.__simpleStr(flavsep=flavsep, flavintsep=flavintsep,
                                kindsep=kindsep, addsep=addsep, func=tex)

    def tex(self, *args, **kwargs):
        return self.simpleTex(*args, **kwargs)

    def uniqueFlavsTex(self, flavsep=r', \, '):
        return flavsep.join([f.tex() for f in self.uniqueFlavs()])


ALL_KINDS = NuKindGroup('nuall,nuallbar')


class FIData(dict):
    def __init__(self, val=None):
        if isinstance(val, basestring):
            d = self.__load(val)
        elif isinstance(val, dict):
            d = val
        elif val is None:
            # Instantiate empty FIData
            with BarSep('_'):
                d = {str(f): {str(it):None for it in ALL_INT_TYPES}
                     for f in ALL_FLAVS}
        else:
            raise TypeError('Unrecognized `val` type %s' % type(val))
        self.validate(d)
        self.update(d)

    def __eq__(self, other):
        return utils.recEq(self, other)

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        return utils.recAllclose(self, other, rtol=rtol, atol=atol)

    def set(self, *args):
        all_keys = list(args[:-1])
        top_lvl_key = all_keys[0]
        subkeys = all_keys[1:]
        new_val = deepcopy(args[-1])

        try:
            kind = NuKind(all_keys[0])
            with BarSep('_'):
                f = str(kind.flav())
                it = str(kind.intType())
            all_keys[0] = f
            all_keys.insert(1, it)

        except (ValueError, TypeError):
            flav = NuFlav(all_keys[0])
            with BarSep('_'):
                all_keys[0] = str(flav)
                try:
                    it = str(IntType(all_keys[1]))
                except (IndexError, AssertionError, ValueError, TypeError):
                    pass
                else:
                    all_keys[1] = it

        skstr = '[' + ']['.join([str(k) for k in all_keys]) + ']'
        logging.trace('setting self%s = "%s" (%s)' % 
                      (skstr, new_val, type(new_val)))

        branch_keys = all_keys[:-1]
        node_key = all_keys[-1]
        lvl = self
        for key in branch_keys:
            lvl = lvl[key]
        old_val = lvl[node_key]
        lvl[node_key] = new_val
        try:
            self.validate(self)
        except:
            lvl[node_key] = old_val
            raise

    def get(self, *args):
        all_keys = list(args)
        top_lvl_key = all_keys[0]
        subkeys = all_keys[1:]

        try:
            kind = NuKind(all_keys[0])
            with BarSep('_'):
                f = str(kind.flav())
                it = str(kind.intType())
            all_keys[0] = f
            all_keys.insert(1, it)

        except (ValueError, TypeError):
            flav = NuFlav(all_keys[0])
            with BarSep('_'):
                all_keys[0] = str(flav)
                try:
                    it = str(IntType(all_keys[1]))
                except (IndexError, AssertionError, ValueError, TypeError):
                    pass
                else:
                    all_keys[1] = it

        skstr = '[' + ']['.join([str(k) for k in all_keys]) + ']'
        logging.trace('getting self%s' % (skstr,))

        branch_keys = all_keys[:-1]
        node_key = all_keys[-1]
        lvl = self
        for key in branch_keys:
            lvl = lvl[key]
        return deepcopy(lvl[node_key])

    def __basic_validate(self, fi_container):
         for kind in ALL_KINDS:
            with BarSep('_'):
                f = str(kind.flav())
                it = str(kind.intType())
            assert isinstance(fi_container, dict), "container must be of" \
                    " type 'dict'; instead got %s" % type(fi_container)
            assert fi_container.has_key(f), "container missing flavor '%s'" % f
            assert isinstance(fi_container[f], dict), \
                    "Child of flavor '%s': must be type 'dict' but" \
                    " got %s instead" % (f,type(fi_container[f]))
            assert fi_container[f].has_key(it), \
                    "Flavor '%s' sub-dict must contain a both interaction" \
                    " types, but missing (at least) intType '%s'" % (f, it)

    def validate(self, fi_container):
        self.__basic_validate(fi_container)

    def save(self, fname):
        fileio.to_file(self, fname)

    def __load(self, fname):
        d = fileio.from_file(fname)
        self.validate(d)
        return d

    def idDupes(self, close_rtol=0):
        '''Group kinds according to those that have duplicated data.

        close_rtol
            Set to a positive value to use np.allclose(a, b, rtol=close_rtol)
        '''
        cmpfunc = utils.recEq
        if close_rtol != 0:
            def cmpfunc(x, y):
                return utils.recAllclose(x, y, rtol=close_rtol)
        dupe_kindgroups = []
        dupe_kindgroups_data = []
        for knum, kind in enumerate(ALL_KINDS):
            this_datum = self.get(kind)
            match = False
            for n, group_datum in enumerate(dupe_kindgroups_data):
                if len(this_datum) != len(group_datum):
                    continue
                if cmpfunc(this_datum, group_datum):
                    dupe_kindgroups[n] += kind
                    match = True
                    break
            if not match:
                dupe_kindgroups.append(NuKindGroup(kind))
                dupe_kindgroups_data.append(this_datum)
        #indices = np.argsort(dupe_kindgroups)[0]
        #dupe_kindgroups = [dupe_kindgroups[idx] for idx in indices]
        #dupe_kindgroups_data = [dupe_kindgroups_data[idx] for idx in indices]
        return dupe_kindgroups, dupe_kindgroups_data


class CombinedFIData(FIData):
    '''Container class for storing data that is combined for some
    NuKinds (as opposed to FIData, which stores one datum for each kind
    separately).

    val
        Data with which to populate the hierarchy.
    kind_groupings
        User-defined groupings of kinds. One set of data is shared among all
        kinds in a kind grouping, and therefore a change to one kind in the
        group also affects all others.

        None
            if val == None, no kinds are grouped together
            if val != None, kind_groupings are deduced from the data
        string
            is a string, it is expected to be ";"-delimited whose
            fields describe kind_groups
        iterable
            the members are converted to strings and interpreted as in the case
            that kind_groupings is a string
    dedup : bool (default: False)
        If True, after populating all data according to any `val` and
        `kind_groupings` arguments, re-groups the data according to those kinds
        whose data is considered equal (see `dedup_close_rtol`).
        (Default: False)
    dedup_close_rtol : numeric (default: 0)
        If set to 0, dedup requires exact equality among kinds' datasets to
        group them together. If non-zero, this gives the relative tol (rtol)
        parameter passed to numpy.allclose for determining close-enough
        equality. This parameter has no effect if `dedup` is False.
    '''
    def __init__(self, val=None, kind_groupings=None, dedup=False, dedup_close_rtol=False):
        # Interpret the kind_groupings arg
        if kind_groupings is None:
            grouped = []
            ungrouped = list(ALL_KINDS)
        elif isinstance(kind_groupings, basestring):
            grouped, ungrouped = self.xlateGroupsStr(kind_groupings)
        elif hasattr(kind_groupings, '__iter__'):
            strkgs = ';'.join([str(x) for x in kind_groupings])
            grouped, ungrouped = self.xlateGroupsStr(strkgs)
        else:
            raise TypeError('Incomprehensible `kind_groupings`: "%s"' %
                            str(kind_groupings))

        # Interpret the val arg
        named_g = None
        named_ung = None
        if isinstance(val, basestring):
            val = self.__load(val)

        if isinstance(val, dict):
            d = val
            named_g, named_ung = self.xlateGroupsStr(';'.join(d.keys()))
            # Force keys to standard naming convention (be liberal on input,
            # strict on output)
            for key in d.keys():
                for g in named_g + named_ung:
                    if (NuKindGroup(key) == g) and not (key == str(g)):
                        d[str(g)] = d.pop(key)
        elif val is None:
            if kind_groupings is None:
                logging.warn('CombinedFIData object instantiated without'
                             ' kind groupings specified; might as well use'
                             ' FIData')
            named_g = grouped
            named_ung = ungrouped

            # Instantiate empty dict with groupings as keys
            d = {str(k):None for k in named_g+named_ung}
        else:
            raise TypeError('Unrecognized `val`: "%s", type %s' %
                            (str(val), type(val)))
        if named_g and kind_groupings:
            assert named_g == grouped
            assert named_ung == ungrouped
      
        self.validate(d)
        self.grouped, self.ungrouped = named_g, named_ung
        self.kinds_to_keys = [(ks, str(ks)) for ks in named_g+named_ung]
        self.update(d)

        if dedup:
            self.deduplicate(close_rtol=dedup_close_rtol)

    def deduplicate(self, close_rtol=False):
        '''Identify duplicate datasets and combine the associated kinds
        together, elinimating redundancy in the data.

        This forces any kinds with identical data to be tied to one another
        into the future
        '''
        dupe_kgs, dupe_kgs_data = self.idDupes(close_rtol=close_rtol)
        d = {str(kg):dat for kg,dat in izip(dupe_kgs, dupe_kgs_data)}
        self.validate(d)
        self.grouped = [kg for kg in dupe_kgs if len(kg) > 1]
        self.ungrouped = [kg for kg in dupe_kgs if len(kg) == 1]
        self.kinds_to_keys = [(kg, str(kg)) for kg in dupe_kgs]
        self.clear()
        self.update(d)

    def __basic_validate(self, cfid):
        assert isinstance(cfid, dict), 'container must be of' \
                ' type `dict`; instead got %s' % str(type(cfid))
        keys = cfid.keys()
        key_grps = [NuKindGroup(k) for k in keys]
        for kind in ALL_KINDS:
            found = 0
            for grp in key_grps:
                if kind in grp:
                    found += 1
            assert found > 0, 'container missing kind %s' % str(kind)

    def validate(self, cfid):
        self.__basic_validate(cfid)

    def __eq__(self, other):
        return utils.recEq(self, other)

    def set(self, *args):
        all_keys = list(args[:-1])
        new_val = deepcopy(args[-1])
        tgt_grp = NuKindGroup(all_keys[0])
        for (kinds, key) in self.kinds_to_keys:
            match = False
            # Identical
            if tgt_grp == kinds:
                all_keys[0] = key
                match = True
            # Requested kinds are strict subset
            elif len(tgt_grp - kinds) == 0:
                all_keys[0] = key
                match = True
                logging.warning('Setting data for subset (%s) of'
                                ' grouping %s' % (str(tgt_grp), str(kinds)))
            # Set it
            if match:
                branch_keys = all_keys[:-1]
                node_key = all_keys[-1]
                lvl = self
                for k in branch_keys:
                    lvl = lvl[k]
                old_val = lvl[node_key]
                lvl[node_key] = new_val
                try:
                    self.validate(self)
                except:
                    lvl[node_key] = old_val
                    raise
                return
        # If you get this far, no match was found
        raise ValueError('Could not set data for group %s' % str(tgt_grp))

    def get(self, *args):
        all_keys = list(args)
        tgt_grp = NuKindGroup(all_keys[0])
        for (kinds, key) in self.kinds_to_keys:
            match = False
            # Identical
            if tgt_grp == kinds:
                all_keys[0] = key
                match = True
            # Requested kinds are strict subset
            elif len(tgt_grp - kinds) == 0:
                all_keys[0] = key
                match = True
                logging.warning('Requesting data for subset (%s) of'
                                ' grouping %s' % (str(tgt_grp), str(kinds)))
            # Get it
            if match:
                branch_keys = all_keys[:-1]
                node_key = all_keys[-1]
                lvl = self
                for k in branch_keys:
                    lvl = lvl[k]
                return deepcopy(lvl[node_key])
        # If you get this far, no match was found
        raise ValueError('Could not locate data for group %s' % str(tgt_grp))

    def __load(self, fname):
        d = fileio.from_file(fname)
        self.validate(d)
        return d

    def save(self, fname):
        fileio.to_file(self, fname)

    @staticmethod
    def xlateGroupsStr(val):
        '''Translate a ";"-separated string into separate `NuKindGroup`s.
    
        val
            ";"-delimited list of valid NuKindGroup strings, e.g.:
                "nuall_nc;nue;numu_cc+numubar_cc"
            Note that specifying NO interaction type results in both interaction
            types being selected, e.g. "nue" implies "nue_cc+nue_nc". For other
            details of how the substrings are interpreted, see docs for
            NuKindGroup.
    
        returns:
            grouped, ungrouped
    
        grouped, ungrouped
            lists of NuKindGroups; the first will have more than one kind in each
            NuKindGroup whereas the second will have just one kind in each
            NuKindGroup. Either list can be of 0-length.
    
        This function does not enforce mutual-exclusion on kinds in the various
        kind groupings, but does list any kinds not grouped together in the
        `ungrouped` return arg. Mutual exclusion can be enforced through set
        operations upon return.
        '''
        # What kinds to group together
        grouped = [NuKindGroup(s) for s in val.split(';')]
    
        # Find any kinds not included in the above groupings
        all_kinds = set(ALL_KINDS)
        all_grouped_kinds = set(NuKindGroup(grouped))
        ungrouped = [NuKindGroup(k)
                     for k in sorted(all_kinds.difference(all_grouped_kinds))]
    
        return grouped, ungrouped


def test_IntType():
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #==========================================================================
    # Test IntType
    #==========================================================================
    ref = IntType('cc')
    assert IntType('\n\t _cc \n') == ref
    assert IntType('numubarcc') == ref
    assert IntType(1) == ref
    assert IntType(1.0) == ref
    assert IntType(IntType('cc')) == ref
    assert IntType(NuKind('numubarcc')) == ref
    for i in all_i_codes:
        IntType(i)
        IntType(float(i))
    logging.info('<< PASS >> : IntType checks')


def test_NuFlav():
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #==========================================================================
    # Test NuFlav
    #==========================================================================
    ref = NuFlav('numu')
    assert ref.flavCode() == 14
    assert (-ref).flavCode() == -14
    assert ref.barNoBar() == 1
    assert (-ref).barNoBar() == -1
    assert ref.isParticle()
    assert not (-ref).isParticle()
    assert not ref.isAntiParticle()
    assert (-ref).isAntiParticle()

    assert NuFlav('\n\t _ nu_ mu_ cc\n\t\r') == ref
    assert NuFlav('numucc') == ref
    assert NuFlav(14) == ref
    assert NuFlav(14.0) == ref
    assert NuFlav(NuFlav('numu')) == ref
    assert NuFlav(NuKind('numucc')) == ref
    assert NuFlav(NuKind('numunc')) == ref

    for f in all_f_codes:
        NuFlav(f)
        NuFlav(float(f))
    for (f,bnb) in product(['e','mu','tau'], ['', 'bar']):
        NuFlav('nu_' + f + '_' + bnb)

    logging.info('<< PASS >> : NuFlav checks')


def test_NuKind():
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #==========================================================================
    # Test NuKind
    #==========================================================================
    try:
        NuKind('numu')
    except ValueError:
        pass

    # Equality
    fi_comb = [fic for fic in product(all_f_codes, all_i_codes)]
    for (fi0,fi1) in product(fi_comb, fi_comb):
        if fi0 == fi1:
            assert NuKind(fi0) == NuKind(fi1)
        else:
            assert NuKind(fi0) != NuKind(fi1)
    assert NuKind((12,1)) != 'xyz'
    # Sorting: this is my desired sort order
    nfl0 = [NuKind(fic) for fic in fi_comb]
    nfl1 = [NuKind(fic) for fic in fi_comb]
    np.random.shuffle(nfl1)
    nfl_sorted = sorted(nfl1)
    assert all([ v0 == nfl_sorted[n] for n,v0 in enumerate(nfl0) ])
    assert len(nfl0) == len(nfl_sorted)
    
    # Test NuKind instantiation
    nue = NuFlav('nue')
    cc = IntType('cc')
    nc = IntType('nc')
    nuebar = NuFlav('nuebar')
    flavs = list(ALL_FLAVS)
    flavs.extend(['nue', 'numu', 'nutau', 'nu_e', 'nu e', 'Nu E', 'nuebar', 'nu e bar'])
    flavs.extend(all_f_codes)
    ints = [cc, nc, 'cc', 'nc', 'CC', 'NC', 1, 2]
    nuecc = NuKind('nuecc')
    nuebarnc = NuKind('nuebarnc')

    # Instantiate with combinations of flavs and int types
    for f,i in product(flavs, ints):
        ref = NuKind(f, i)
        assert NuKind((f, i)) == ref
        assert NuKind(flav=f, int_type=i) == ref
        if isinstance(f, basestring) and isinstance(i, basestring):
            assert NuKind(f+i) == ref
            assert NuKind(f + '_' + i) == ref
            assert NuKind(f + ' ' + i) == ref

    # Instantiate with already-instantiated `NuKind`s
    assert NuKind(nuecc) == NuKind('nuecc')
    assert NuKind(nuebarnc) == NuKind('nuebarnc')

    # test negating kind
    nk = NuKind('numucc')
    assert -nk == NuKind('numubarcc')

    logging.info('<< PASS >> : NuKind checks')


def test_NuKindGroup():
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #==========================================================================
    # Test NuKindGroup
    #==========================================================================
    fi_comb = [fic for fic in product(all_f_codes, all_i_codes)]
    nfl0 = [NuKind(fic) for fic in fi_comb]
    nfl1 = [NuKind(fic) for fic in fi_comb]
    nfl_sorted = sorted(nfl1)
    nkg0 = NuKindGroup(nfl0)
    nkg1 = NuKindGroup(nfl_sorted)
    assert nkg0 == nkg1
    assert nkg0 != 'xyz'
    assert nkg0 != 'xyz'

    # Test inputs
    assert NuKindGroup('nuall,nuallbar').uniqueFlavs() == \
            tuple([NuFlav(c) for c in all_f_codes])

    #
    # Test NuKindGroup instantiation
    #
    nue = NuFlav('nue')
    numu = NuFlav('numu')
    nue_cc = NuKind('nue_cc')
    nue_nc = NuKind('nue_nc')

    # Empty args
    NuKindGroup()
    NuKindGroup([])

    # String flavor promoted to CC+NC
    assert set(NuKindGroup('nue').kinds()) == set((nue_cc, nue_nc))
    # NuFlav promoted to CC+NC
    assert set(NuKindGroup(nue).kinds()) == set((nue_cc, nue_nc))
    # List of single flav str same as above
    assert set(NuKindGroup(['nue']).kinds()) == set((nue_cc, nue_nc))
    # List of single flav same as above
    assert set(NuKindGroup([nue]).kinds()) == set((nue_cc, nue_nc))

    # Single kind spec
    assert set(NuKindGroup(nue_cc).kinds()) == set((nue_cc,))
    # Str with single kind spec
    assert set(NuKindGroup('nue_cc').kinds()) == set((nue_cc,))
    # List of single str containing single kind spec
    assert set(NuKindGroup(['nue_cc']).kinds()) == set((nue_cc,))

    # Multiple kinds as *args
    assert set(NuKindGroup(nue_cc, nue_nc).kinds()) == set((nue_cc, nue_nc))
    # List of kinds
    assert set(NuKindGroup([nue_cc, nue_nc]).kinds()) == set((nue_cc, nue_nc))
    # List of single str containing multiple kinds spec
    assert set(NuKindGroup(['nue_cc,nue_nc']).kinds()) == set((nue_cc, nue_nc))
    # List of str containing kinds spec
    assert set(NuKindGroup(['nue_cc','nue_nc']).kinds()) == set((nue_cc, nue_nc))

    # Another NuKindGroup
    assert set(NuKindGroup(NuKindGroup(nue_cc, nue_nc)).kinds()) == set((nue_cc, nue_nc))

    # Addition of kinds promoted to NuKindGroup
    assert nue_cc + nue_nc == NuKindGroup(nue)
    # Addition of flavs promoted to NuKindGroup including both CC & NC
    assert nue + numu == NuKindGroup(nue, numu)

    # Test remove
    nkg = NuKindGroup('nue_cc+numucc')
    nkg.remove(NuKind((12,1)))
    assert nkg == NuKindGroup('numucc')

    # Test del
    nkg = NuKindGroup('nue_cc+numucc')
    del nkg[0]
    assert nkg == NuKindGroup('numucc')

    # Equivalent object when converting to string and back to NuKindGroup from
    # that string
    for n in range(1, len(ALL_KINDS)+1):
        logging.debug('NuKindGroup --> str --> NuKindGroup, n = %d' % n)
        for comb in combinations(ALL_KINDS, n):
            ref = NuKindGroup(comb)
            assert ref == NuKindGroup(str(ref))

    # test TeX strings
    nkg = NuKindGroup('nuall,nuallbar')
    logging.info(str(nkg))
    logging.info(tex(nkg))
    logging.info(nkg.simpleStr())
    logging.info(nkg.simpleTex())
    logging.info(nkg.uniqueFlavsTex())

    logging.info('<< ???? >> : NuKindGroup checks pass upon inspection of'
                 ' above outputs and generated file(s).')


def test_FIData():
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #==========================================================================
    # Test FIData
    #==========================================================================
    # Excercise the "standard" PISA nested-python-dict features, where this
    # dict uses an '_' to separate 'bar' in key names, and the nested dict
    # levels are [flavor][interaction type].
    
    # Force separator to something weird before starting, to ensure everything
    # still works and this separator is still set when we're done
    oddball_sep = 'xyz'
    set_bar_ssep(oddball_sep)
    ref_pisa_dict = {f:{it:None for it in ['cc','nc']} for f in
                     ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']}
    fi_cont = FIData()
    for f in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        for it in ['cc','nc']:
            assert fi_cont[f][it] == ref_pisa_dict[f][it]
            kind = NuKind(f, it)
            assert kind.pidx(ref_pisa_dict) == ref_pisa_dict[f][it]
            assert fi_cont.get(kind) == fi_cont[f][it]
            assert fi_cont.get(f)[it] == fi_cont[f][it]
    assert get_bar_ssep() == oddball_sep
    set_bar_ssep('')

    # These should fail because they invalidate the data
    try:
        fi_cont.set(NuFlav('numu'), 'xyz')
    except AssertionError:
        pass
    else:
        raise Exception('Test failed, exception should have been raised')
    # The previously-valid fi_cont should *still* be valid, as `set` should
    # revert to the original (valid) values rather than keep the invalid values
    fi_cont.validate(fi_cont)

    # Test setting, getting, and JSON serialization of FIData
    fi_cont.set('nue_cc', 'this is a string blah blah blah')
    fi_cont.get(NuKind('nue_cc'))
    fi_cont.set(NuKind('nue_nc'), np.pi)
    fi_cont.get(NuKind('nue_nc'))
    fi_cont.set(NuKind('numu_cc'), [0,1,2,3])
    fi_cont.get(NuKind('numu_cc'))
    fi_cont.set(NuKind('numu_nc'), {'new':{'nested':{'dict':'xyz'}}})
    fi_cont.get(NuKind('numu_nc'))
    fi_cont.set(NuKind('nutau_cc'), 1)
    fi_cont.get(NuKind('nutau_cc'))
    fi_cont.set(NuKind('nutaubar_cc'), np.array([0,1,2,3]))
    fi_cont.get(NuKind('nutaubar_cc'))
    fileio.to_file(fi_cont, '/tmp/test_FIData.json')

    logging.info('<< ???? >> : FIData checks pass upon inspection of'
                 ' above outputs and generated file(s).')



def test_CombinedFIData():
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #==========================================================================
    # Test CombinedFIData.xlateGroupsStr function
    #==========================================================================
    # Test string parsing for flavor groupings
    gp1, ug1 = CombinedFIData.xlateGroupsStr('nuall_nc; nuallbar_nc; nue;'
                              'numu_cc+numubar_cc; nutau_cc')
    logging.info(str(([kg.simpleStr() for kg in gp1], ug1)))
    gp2, ug2 = CombinedFIData.xlateGroupsStr('nue,numu')
    logging.info(str(([kg.simpleStr() for kg in gp2], ug2)))
    gp3, ug3 = CombinedFIData.xlateGroupsStr('nuall_nc')
    logging.info(str(([kg.simpleStr() for kg in gp3], ug3)))
    gp4, ug4 = CombinedFIData.xlateGroupsStr('nuall_nc+nuallbar_nc;nuall_cc+nuallbar_cc')
    logging.info(str(([kg.simpleStr() for kg in gp4], ug4)))

    logging.info('<< PASS >> : CombinedFIData.xlateGroupsStr')

    #==========================================================================
    # Test CombinedFIData class
    #==========================================================================
    # Empty container with no groupings
    CombinedFIData()
    # Empty container with groupings
    CombinedFIData(kind_groupings='nuall;nuallbar')
    # Instantiate with non-standard key names
    cfid = CombinedFIData(val={'nuall':np.arange(0,100),
                               'nu all bar CC':np.arange(100,200),
                               'nuallbarnc':np.arange(200,300)})
    assert set(cfid.keys()) == set(('nuall', 'nuallbar_cc', 'nuallbar_nc'))
    cfid.save('/tmp/cfid.json')
    cfid.save('/tmp/cfid.hdf5')
    cfid2 = CombinedFIData('/tmp/cfid.json')
    cfid3 = CombinedFIData('/tmp/cfid.hdf5')
    assert cfid2 == cfid
    assert cfid3 == cfid

    # Test deduplication
    d1 = {
        'nuecc':        np.arange(0,10),
        'nuebarcc':     np.arange(0,10),
        'numucc':       np.arange(1,11),
        'numubarcc':    np.arange(1,11),
        'nutaucc':      np.arange(2,12),
        'nutaubarcc':   np.arange(2,12),
        'nuallnc':      np.arange(20,30),
        'nuallbarnc':   np.arange(10,20),
    }
    d2 = {
        'nuecc':        np.arange(0,10)*(1+1e-10),
        'nuebarcc':     np.arange(0,10),
        'numucc':       np.arange(1,11)*(1+1e-10),
        'numubarcc':    np.arange(1,11),
        'nutaucc':      np.arange(2,12)*(1+1e-8),
        'nutaubarcc':   np.arange(2,12),
        'nuallnc':      np.arange(20,30),
        'nuallbarnc':   np.arange(10,20),
    }

    # Require exact equality, fields are exactly equal
    cfid1 = CombinedFIData(val=d1, dedup=True, dedup_close_rtol=0)

    # Loose rtol spec, fields are not exactly equal but all within rtol (should
    # match 1)
    rtol = 1e-7
    cfid2 = CombinedFIData(val=d2, dedup=True, dedup_close_rtol=rtol)
    assert cfid2.allclose(cfid1, rtol=rtol)

    # Tight rtol spec, fields are not exactly equal and some CC groupings are
    # now outside rtol
    rtol = 1e-9
    cfid3 = CombinedFIData(val=d2, dedup=True, dedup_close_rtol=rtol)
    assert not cfid1.allclose(cfid3, rtol=rtol)

    # Tight rtol spec, fields are not exactly equal and all CC groupings are
    # outside rtol
    rtol = 1e-11
    cfid4 = CombinedFIData(val=d2, dedup=True, dedup_close_rtol=rtol)
    assert not cfid1.allclose(cfid4, rtol=rtol)

    # Loose rtol, fields are exactly equal
    cfid5 = CombinedFIData(val=d1, dedup=True, dedup_close_rtol=1e-7)
    assert cfid1 == cfid5

    # Tighter rtol, fields are exactly equal
    cfid6 = CombinedFIData(val=d1, dedup=True, dedup_close_rtol=1e-9)
    assert cfid1 == cfid6

    # Very tight rtol, fields are exactly equal
    cfid7 = CombinedFIData(val=d1, dedup=True, dedup_close_rtol=1e-14)
    assert cfid1 == cfid7

    cfidat = CombinedFIData(
        kind_groupings='nuecc+nuebarcc;'
                       'numucc+numubarcc;'
                       'nutaucc+nutaubarcc;'
                       'nuallnc;'
                       'nuallbarnc'
    )
    for k in ALL_KINDS:
        cfidat.set(k, np.arange(10))
    cfidat.deduplicate()
    assert len(cfidat.kinds_to_keys) == 1
    assert cfidat.kinds_to_keys[0][0] == NuKindGroup('nuall+nuallbar')

    logging.info(str([NuKindGroup(k) for k in cfid1.keys()]))
    logging.info('<< ???? >> : CombinedFIData')


if __name__ == "__main__":
    #test_IntType()
    #test_NuFlav()
    #test_NuKind()
    #test_NuKindGroup()
    #test_FIData()
    test_CombinedFIData()
