# Stage 2: Oscillations

The purpose of this stage is to modify the modeled neutrino
atmospheric flux by applying the appropriate oscillation probability
in each energy and zenith bin through the earth, for each flavour. At
the end of the stage, the oscillated neutrino flux for each flavour is
given.

## Main Module

The primary module for this stage is `Oscillation.py`, which provides
a function the takes the atmospheric flux maps of stage 1 as an input
and returns the oscillated flux maps as an output (in the same units).
```
def get_osc_flux(flux_maps,osc_service=None,deltam21=None,deltam31=None,
                 energy_scale=None, theta12=None,theta13=None,theta23=None,
                 deltacp=None,**kwargs):
```
The total event __flux__ in each bin is then given by

### Parameters
* `flux_maps`: A set of atmospheric flux maps in the format, with one
  map for each neutrino flavour and parity (i.e `nue, nue_bar, numu,
  numu_bar, nutau, nutau_bar`).
* `osc_service`: An _oscillation probability service_ that provides
  oscillation probabilities as a function of energy, zenith, and all
  oscillation parameters (deltam21, deltam31, theta12, theta13,
  theta23, deltacp).
* `deltam21`: Value of "solar mass splitting" oscillation parameter,
  \Delta m_{21}^{2} in units of [eV^2].
* `deltam31`: Value of "atmospheric mass splitting" oscillation
  parameter, \Delta m_{31}^{2} in units of [eV^2].
* `theta12`: Value of theta_{12} parameter, units of [rad]
* `theta13`: Value of theta_{13} parameter, units of [rad]
* `theta23`: Value of theta_{23} parameter, units of [rad]
* `deltacp`: Value of cp-violating phase, \delta_{cp}, units of [rad]

  areas as a function of energy and cos(zenith) and interaction
  type. See below for a choice of services.

### Output

This function returns maps of oscillated flux for each flavour and
interaction type.

```
  { ‘nue’: { ‘cc’: map,  ‘nc’: map},
     ‘nue_bar’: { ‘cc’: map,  ‘nc’: map},
     ‘numu’: { ‘cc’: map,  ‘nc’: map},
     ‘numu_bar’: { ‘cc’: map,  ‘nc’: map},
     ‘nutau’: { ‘cc’: map,  ‘nc’: map},
     ‘nutau_bar’: { ‘cc’: map,  ‘nc’: map},
     ‘params’: params}
```

## Services

Three oscillation services are provided, one for using the `Prob3`
code to generate oscillation probabilities, a second for using the
`NuCraft` code, and a third to use a pre-tabulated oscillation
probability table. The methods common to all OscillationServices
(which derive from `OscillationServiceBase`) are:

* `get_osc_prob_maps`: This method returns an oscillation probability
  map dictionary calculated at a particular set of oscillation
  parameters (deltam21, deltam31, theta12, etc.). The output
  dictionary is formatted as

```
{'nue_maps': {'nue':map,'numu':map,'nutau':map},
 'numu_maps': {...},
 'nue\_bar\_maps': {...},
 'numu\_bar\_maps': {...}
 }
```

(and if _nutau_ and _nutau\_bar_ exist in the input flux, then they
will appear here as well).

* `get_osc_probLT_dict`: Creates the oscillation probability lookup
  tables (LT) corresponding to atmospheric neturinos oscillating
  through the earth, and will return a dictionary of oscillation
  probability maps. This function is called by `get_osc_prob_maps` and
  it is here that the oversampling of energy/coszen is configured.

* `fill_osc_prob`: Method that does the heavy lifting of actually
  calculating the oscillaiton probabilities, oversampling them in each
  bin, then averaging them and storing them in the final oscillation
  probability map binning. This method is implemented separately in
  each derived Service.

### Prob3OscillationService

```
class Prob3OscillationService(OscillationServiceBase):
     def __init__(self, ebins, czbins, detector_depth=None, earth_model=None,
                  prop_height=None, **kwargs):
```

This service is initialized with the following parameters:

* `ebins`: Energy bin edges [GeV]
* `czbins`: cos(zenith) bin edges
* `earth_model`: Earth density model used for matter oscillations.
* `detector_depth`: Detector depth in km.
* `prop_height`: Height in the atmosphere where the neutrino
  interactions begin in km.

This service calculates the neutrino oscillation probabilities using
the `Prob3` code, which at its core, relies on a 3-flavour analytic
solution to the neutrino oscillation propagation over constant matter
density. For a realistic earth model, small enough constant density
layers are chosen to accuratly describe the matter density through the
earth. The `Prob3` code, initially written by the Super-Kamiokande
collaboration in C/C++, is publicly available here:

http://www.phy.duke.edu/~raw22/public/Prob3++/

and has been given PyBindings as well as a few additional
modifications to be optimized for the IceCube detector, and for use in
PISA.

### NuCraftOscillationService

```
def __init__(self, ebins, czbins, detector_depth=None, earth_model=None,
             prop_height=None, osc_precision=None,
             **kwargs):
```

This service is initialized with the following parameters:
* `ebins`: Energy bin edges [GeV]
* `czbins`: cos(zenith) bin edges
* `earth_model`: Earth density model used for matter oscillations.
* `detector_depth`: Detector depth in km.
* `prop_height`: Height in the atmosphere to begin in km.
    Default: 'sample', samples from a parametrization to the
    atmospheric interaction model presented in"Path length
    distributions of atmospheric neutrinos", Gaisser and Stanev,
    PhysRevD.57.1977
* `osc_precision`: Numerical precision for oscillation probabilities

This service uses the `NuCraft` oscillation code to calculate the
oscillation probabilities. `NuCraft` is used in the IceCube
collaboration and is publicly available from:

http://nucraft.hepforge.org

Because it was written to handle the general case of an arbitrary
number of flavors (>3) of neutrinos propagating through non-constant
matter density, it solves the 1D schroedinger equation at small step
sizes, but takes far longer to generate the oscillation probabilities
than the `Prob3` code.

### TableOscillationService

```
def __init__(self,ebins,czbins,datadir=None, **kwargs):
```

This service is initialized with the following parameters:

* `ebins`: Energy bin edges [GeV]
* `czbins`: cos(zenith) bin edges
* `datadir`: directory where oscillation probability tables are stored.

This service has not been fully implemented yet for arbirary
oscillation parameter values as inputs and should not be used at the
time of the writing of this document.
