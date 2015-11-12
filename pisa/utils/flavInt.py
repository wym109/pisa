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


import sys, traceback
import numpy as np
from copy import deepcopy
import re
import collections

from pisa.utils.log import logging, set_verbosity
import pisa.utils.fileio as fileio


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
    def __init__(self, flav):
        self.ignore = re.compile('[-_. ]')
        self.f_re = re.compile(
            r'(?P<fullflav>(?:nue|numu|nutau)(?P<barnobar>bar){0,1})'
        )
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
        # Instantiate this neutrino flavor object
        if isinstance(flav, basestring):
            orig_flav = flav
            flav = flav.lower()
            flav = flav.strip()
            flav,_ = self.ignore.subn('', flav)
            flav = flav.strip()
            try:
                flav_dict = self.f_re.match(flav).groupdict()
            except:
                raise ValueError('Could not interpret input "'+orig_flav+'"')
            self.__flav = self.fstr2code[flav_dict['fullflav']]
            self.__barnobar = self.barnobar2code[flav_dict['barnobar']]
        elif isinstance(flav, NuFlav):
            self.__flav = flav.flavCode()
            self.__barnobar = flav.barNoBar()
        else:
            try:
                int(flav) == float(flav)
            except:
                raise TypeError('Unhandled type: "'+str(type(flav))+'"')
            else:
                if (int(flav) == float(flav)) and int(flav) in self.fstr2code.values():
                    self.__flav = int(flav)
                    self.__barnobar = np.sign(self.__flav)
                else:
                    raise ValueError('Invalid neutrino flavor/code: "'+str(flav)+'"')

        assert self.__flav in self.fstr2code.values()
        assert self.__barnobar in self.barnobar2code.values()

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

    #def set_bar_ssep(self, val):
    #    '''Set separator between "base" flavor ("nue", etc.) and "bar" when
    #    stringifying the flavor'''
    #    assert isinstance(val, basestring)
    #    self.BAR_SSEP = val

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
    CC_CODE = 1
    NC_CODE = 2
    def __init__(self, int_type):
        self.ignore = re.compile('[-_. ]')
        self.int_re = re.compile(r'.*(cc|nc)$')
        self.istr2code = {
            'cc': self.CC_CODE,
            'nc': self.NC_CODE,
        }
        self.i2tex = {
            self.CC_CODE: r'{\rm CC}',
            self.NC_CODE: r'{\rm NC}'
        }
        # Instantiate
        # TODO: clean up logic
        if isinstance(int_type, basestring):
            orig_int_type = int_type
            int_type = int_type.lower()
            int_type,_ = self.ignore.subn('', int_type)
            int_type = int_type.strip()
            try:
                int_type_match = self.int_re.match(int_type).groups()[0]
            except:
                raise ValueError('Could not interpret input "'+orig_int_type+'"')
            self.__int_type = self.istr2code[int_type_match]
        elif isinstance(int_type, IntType):
            self.__int_type = int_type.intTypeCode()
        else:
            try:
                int(int_type) == float(int_type)
            except:
                raise TypeError('Unhandled type: "'+str(type(int_type))+'"')
            else:
                if (int(int_type) == float(int_type) and
                        int(int_type) in self.istr2code.values()):
                    self.__int_type = int(int_type)
        assert self.__int_type in self.istr2code.values()

    def __str__(self):
        return [s for s,code in self.istr2code.items() if code == self.__int_type][0]

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
    '''A neutrino "kind" encompasses both the neutrino flavor and its interaction type.'''
    # TODO: use multiple inheritance to clean up the below
    def __init__(self, *args):
        self.ignore = re.compile('[-_. ]')
        self.all_fint_sep = re.compile('[+,; ]')
        self.fint_re = re.compile(r'(?P<fullflav>(?:nue|numu|nutau)(?P<barnobar>bar){0,1})(?P<int_type>cc|nc){0,1}')
        self.fint_ssep = '_'
        self.fint_tsep = r' \, '

        if len(args) == 1:
            flav_int = args[0]
        else:
            flav_int = args

        # Initialize with string
        if isinstance(flav_int, basestring):
            flav_int,_ = self.ignore.subn('', flav_int)
            flav_int = flav_int.strip()
            kind_dict = self.fint_re.match(flav_int).groupdict()
            self.__flav = NuFlav(kind_dict['fullflav'])
            self.__int_type = IntType(kind_dict['int_type'])
        elif hasattr(flav_int, '__len__') and len(flav_int) == 2:
            self.__flav = NuFlav(flav_int[0])
            self.__int_type = IntType(flav_int[1])
        elif isinstance(flav_int, NuKind):
            self.__flav = NuFlav(flav_int.flav())
            self.__int_type = IntType(flav_int.intTypeCode())
        else:
            raise TypeError('Unhandled type: "'+str(type(flav_int))+'"; class: "'+str(flav_int.__class__)+'; value: "' + str(flav_int) + '"')

    def __str__(self):
        return self.flavStr() + self.fint_ssep + self.intTypeStr()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash( (self.flavCode(), self.intTypeCode()) )

    def __cmp__(self, other):
        if not isinstance(other, NuKind):
            return 1
        return cmp( (self.flav(), self.intType()), (other.flav(), other.intType()) )

    def __neg__(self):
        return NuKind(-self.__flav, self.__int_type)

    #def set_bar_ssep(self, val):
    #    self.__flav.set_bar_ssep(val)

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
        return r'{' + self.flavTex() + self.fint_tsep + self.intTypeTex() + r'}'


class NuKindGroup(collections.MutableSequence):
    def __init__(self, *args):
        self.ignore = re.compile('[-_.]')
        self.all_kind_sep = re.compile('[+, ]')
        self.kind_sep = '+'
        self.__kinds = []
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
        kind_list = sorted(set(self.interpret(val)))
        for k in kind_list:
            idx = self.__kinds.index(k)
            del self.__kinds[idx]

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
        return '+'.join([str(k) for k in self.__kinds])

    def __repr__(self):
        return self.__str__()

    # TODO:
    # Technically, since this is a mutable type, the __hash__ method shouldn't
    # be implemented as this will allow for "illegal" behavior, like using
    # a NuKindGroup as a key in a dict. So this should be fixed, maybe.
    #__hash__ = None
    def __hash__(self):
        return hash(tuple(self.__kinds))

    #def set_bar_ssep(self, val):
    #    [k.set_bar_ssep(val) for k in self.__kinds]

    def interpret(self, val):
        kind_list = []
        if isinstance(val, basestring):
            val = val.lower()
            val.strip()
            val,_ = self.ignore.subn('', val)
            kinds = []
            tmp_kinds = [kstr for kstr in self.all_kind_sep.split(val) if kstr != '']
            if len(tmp_kinds) == 0:
                tmp_kinds.append(val)
            for k in tmp_kinds:
                if 'all' in k:
                    for fs in ['e','mu','tau']:
                        kinds.append(k.replace('all', fs))
                else:
                    kinds.append(k)
        elif isinstance(val, NuFlav):
            kinds = [NuKind((val,'cc')), NuKind((val,'nc'))]
        elif isinstance(val, NuKind):
            kinds = [val]
        elif isinstance(val, NuKindGroup):
            kinds = list(val.kinds())
        elif hasattr(val, '__len__'):
            if len(val) == 0:
                return []
            if isinstance(val[0], NuKindGroup):
                kinds = []
                [kinds.extend(v.kinds()) for v in val]
            elif len(val) == 2 and np.isscalar(val[0]):
                try:
                    f = NuFlav(val[0])
                    i = IntType(val[1])
                    kinds = [val]
                except:
                    kinds = val
            else:
                kinds = list(val)
        elif np.isscalar(val):
            kinds = [val]
        elif val is None:
            return []
        else:
            raise Exception('all checks failed: ' + str(val) + ', class '
                            + str(val.__class__) + ' type ' + str(val))

        for k in kinds:
            try:
                nk = NuKind(k)
                kind_list.append(nk)
            except TypeError:
                #exc_type, exc_value, exc_traceback = sys.exc_info()
                #lines = traceback.format_exception(exc_type, exc_value,
                #                                   exc_traceback)
                #logging.trace('\n'.join(lines))
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
        return tuple([k.flav() for k in self.__kinds if k.intType() == IntType('cc')])
    
    def ncFlavs(self):
        return tuple([k.flav() for k in self.__kinds if k.intType() == IntType('nc')])

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

    def simpleStr(self, flavsep=',', flavintsep=' ', kindsep=';', addsep='+'):
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
            d = self.load(val)
        elif isinstance(val, dict):
            d = val
        elif val is None:
            # Instantiate empty FIData
            with BarSep('_'):
                d = {str(f):{str(it):None for it in ALL_INT_TYPES} for f in ALL_FLAVS}
        else:
            raise TypeError('Unrecognized `val` type %s' % type(val))
        self.validate(d)
        self.update(d)

    def __eq__(self, other):
        for k in ALL_KINDS:
            if not np.all(self.get(k) == other.get(k)):
                return False
        return True

    def allclose(self, other, rtol=1e-05, atol=1e-08):
        for k in ALL_KINDS:
            if not np.allclose(self.get(k), other.get(k), rtol=rtol, atol=atol):
                return False
        return True

    def basic_get(self, key, *subkeys):
        try:
            kind = NuKind(key)
            with BarSep('_'):
                f = str(kind.flav())
                it = str(kind.intType())
            val = self[f][it]
        except TypeError:
            flav = NuFlav(key)
            with BarSep('_'):
                f = str(flav)
            val = self[f]
        # Recurse into object if subkeys are provided
        for subkey in subkeys:
            val = val[subkey]
        return deepcopy(val)

    def basic_set(self, key, val):
        logging.trace('setting key = "%s" (%s) with val = "%s" (%s)' % 
                      (key, type(key), val, type(val)))
        try:
            kind = NuKind(key)
            with BarSep('_'):
                f = str(kind.flav())
                it = str(kind.intType())
            old_val = self[f][it]
            self[f][it] = deepcopy(val)
            try:
                self.validate(self)
            except AssertionError:
                self[f][it] = old_val
                raise
        except TypeError:
            flav = NuFlav(key)
            with BarSep('_'):
                f = str(flav)
            old_val = self[f]
            self[f] = deepcopy(val)
            try:
                self.validate(self)
            except AssertionError:
                self[f] = old_val
                raise

    @staticmethod
    def basic_validate(fi_container):
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

    @staticmethod
    def validate(fi_container):
        FIData.basic_validate(fi_container)

    def get(self, val, *args):
        return self.basic_get(val, *args)

    def set(self, key, val):
        self.basic_set(key, val)

    def save(self, fname):
        fileio.to_file(self, fname)

    @staticmethod
    def load(fname):
        d = fileio.from_file(self, fname)
        FIData.validate(d)
        return d


def test():
    import itertools
    set_verbosity(2)
    all_f_codes = [12,-12,14,-14,16,-16]
    all_i_codes = [1,2]

    #============================================================================
    # Test NuKind
    #============================================================================
    # Equality
    fi_comb = [fic for fic in itertools.product(all_f_codes, all_i_codes)]
    for (fi0,fi1) in itertools.product(fi_comb, fi_comb):
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

    #============================================================================
    # Test NuKindGroup
    #============================================================================
    nkg0 = NuKindGroup(nfl0)
    nkg1 = NuKindGroup(nfl_sorted)
    assert nkg0 == nkg1
    assert nkg0 != 'xyz'
    assert nkg0 != 'xyz'

    #fi_comb = [fic for fic in itertools.product(all_f_codes, all_i_codes)]
    #k = NuKind(1)
    #[NuKind(f) for f in all_f_codes]
    #[NuKind(f) for f in all_f_codes]
    #k = NuKind()

    assert NuKindGroup('nuall,nuallbar').uniqueFlavs() == \
            tuple([NuFlav(c) for c in all_f_codes])

    # Test remove
    nkg = NuKindGroup('nuecc+numucc')
    nkg.remove(NuKind((12,1)))
    assert nkg == NuKindGroup('numucc')

    # Test del
    nkg = NuKindGroup('nuecc+numucc')
    del nkg[0]
    assert nkg == NuKindGroup('numucc')

    # test negating flavor / kind
    nf = NuFlav('numu')
    assert -nf == NuFlav('numubar')
    nk = NuKind('numucc')
    assert -nk == NuKind('numubarcc')

    # test TeX strings
    nkg = NuKindGroup('nuall,nuallbar')
    logging.info(str(nkg))
    logging.info(tex(nkg))
    logging.info(nkg.simpleStr())
    logging.info(nkg.simpleTex())
    logging.info(nkg.uniqueFlavsTex())

    # Excercise the "standard" PISA nested-python-dict features, where this
    # dict uses an '_' to separate 'bar' in key names, and the nested dict
    # levels are [flavor][interaction type].
    
    # Force separator to something weird before starting, to ensure everything
    # still works and this separator is still set when we're done
    oddball_sep = 'xyz'
    set_bar_ssep(oddball_sep)
    ref_pisa_dict = {f:{it:None for it in ['cc','nc']} for f in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']}
    fi_cont = FIData()
    for f in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
        for it in ['cc','nc']:
            assert fi_cont[f][it] == ref_pisa_dict[f][it]
            kind = NuKind(f, it)
            assert kind.pidx(ref_pisa_dict) == kind.pidx(fi_cont)
    assert get_bar_ssep() == oddball_sep

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


if __name__ == "__main__":
    test()
