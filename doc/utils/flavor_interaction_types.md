# Flavor and interaction types

***NuFlav***, ***IntType***, ***NuFlavInt***, and ***NuFlavIntGroup*** are objects designed to easily work with neutrino flavors, interaction types, and combinations thereof. Source code: *pisa/utils/flavInt.py*

***FlavIntData*** and ***CombinedFlavIntData*** are container objects for data particular to neutrino flavors+interaction types. The former stores one datum for each flavor/interaction type combination, whereas the latter allows one to store a single data set for each grouping of flavor/interaction types. Source code: *pisa/utils/flavInt.py*

***Events*** subclasses *FlavIntData* but also contains standard metatadata pertinent to events. Source code: *pisa/utils/events.py*

## Examples using flavor/interaction type objects
```python
import pisa.utils.flavInt as flavInt
import pisa.utils.events as events

# Create a flavor
numu = flavInt.NuFlav(14) # from code
numu = flavInt.NuFlav('nu_mu') # from str
numubar = -numu # flip parity

# Interaction type
cc = flavInt.IntType('Cc') # permissive
nc = flavInt.IntType(2) # from code

# flavInt = flavor + int. type
numucc = flavInt.NuFlavInt('numu_bar_cc')
numucc = flavInt.NuFlavInt(numu, cc)
numubarcc = -numucc

numubarcc.flav() # -> 'numubar'
numubarcc.intType() # -> 'cc'
numubar.isParticle() # -> False
numubarcc.isCC() # -> True

# TeX string (nice for plots!)
numubarcc.tex() # -> '{{\bar\nu_\mu} \, {\rm CC}}'

# NuFlavIntGroup
nkg = flavInt.NuFlavIntGroup('numucc,numubarcc')
nkg = numucc + numubarcc # works!
nkg.kinds() # -> (numu_cc, numubar_cc)
nkg.particles() # -> (numu_cc,)
nkg -= numucc # remove a kind
nkg.kinds() # -> (numubar_cc,)

# No intType spec=BOTH intTypes
nkg = flavInt.NuFlavIntGroup('numu')
nkg.kinds() # -> (numu_cc, numu_nc)
nkg.ccFlavInts() # -> (numu_cc,)

# Loop over all particle CC kinds
for k in flavInt.NuFlavIntGroup('nuallcc'):

# String, TeX
nkg = flavInt.NuFlavIntGroup('nuallcc'):
print nkg # -> 'nuall_cc'
nkg.tex() # -> '{\nu_{\rm all}} \, {\rm CC}'

```

## Examples using container objects
***Old way***: Instantiate an empty flav/int-type nested dict
```python
oldfidat = {}
flavs = ['nue', 'numu', 'nutau']
int_types = ['cc', 'nc']
for baseflav, int_type in itertools.product(flavs, int_types):
    for mID in ['', '_bar']:
        flav = baseflav + mID
        if not flav in oldfidat:
            oldfidat[flav] = {}
        oldfidat[flav][int_type] = None
```

***New way***: Using FlavIntData
```python
from pisa.utils.flavInt import FlavIntData, ALL_KINDS
fidat = FlavIntData()
```

```python
# Loop through the old way
for baseflav, int_type in itertools.product(flavs, int_types):
    for mID in ['', '_bar']:
        flav = baseflav + mID
        oldfidat[flav][int_type]['map'] = np.arange(0,100)
        m = fidat[flav][int_type]['map']

# The new way
for kind in ALL_KINDS:
    fidat.set(kind, np.arange(0,100))
    m = fidat.get(kind, 'map')

# But if you really like nested dictionaries, you can still do:
fidat['nue_bar']['cc']['map']

# ...easier to use set() and get() though! Can use strings, but the
# interface is awesome now: you don't have to know that you need
# lower-case, '_' infix between 'numu' and 'bar', or structure of
# data structure being used!
fidat.get('numu bar CC')

# This gets the entire branch at 'numu' 
fidat.get('numu') # -> {'cc': None, 'nc': None}

# Save it to a JSON file
fidat.save('data.json')

# Load, instantiate new object
fidat2 = FlavIntData('data.json')

# Compare, recursing into any sub-structs (lists, dicts)
print fidat == fidat2 # -> True

# There is a function for doing this in utils.Utils.py:
from utils.Utils import recEq, recAllclose
print recEq(fidat, fidat2) # -> True
```

```python
from pisa.utils.events import Events

ev = Events('events/events__pingu__v36__runs_388-390__proc_v5__joined_G_nuall_nc_G_nuallbar_nc.hdf5')

print ev.metadata
```
```
{'cuts': array(['analysis'], 
       dtype='|S8'),
 'detector': 'pingu',
 'geom': 'v36',
 'kinds_joined': array(['nuall_nc', 'nuallbar_nc'], 
       dtype='|S11'),
 'proc_ver': '5',
 'runs': array([388, 389, 390])}
```

```python
from pisa.utils.flavInt import CombinedFlavIntData as CFID

cfidat = CFID(kind_groupings='nuecc+nuebarcc; numucc+numubarcc;'
              'nutaucc+nutaubarcc; nuallnc; nuallbarnc')

[cfidat.set(grp, np.random.rand(10)) for grp in cfidat.keys()]
cfidat.save('/tmp/cfidat.json') # save to JSON file
cfidat2 = CFID('/tmp/cfidat.json') # load from file
cfidat3 = CFID(cfidat2) # make another copy

# Force all data to be equal
[cfidat2.set(k, np.arange(10)) for k in cfidat2.keys()]

# Eliminate redundancy (should be just one unique dataset!)
cfidat2.dedupicate()
len(cfidat2) # -> 1
cfidat2.keys() # -> ['nuall+nuallbar']
```
