# Stage 3: Effective Area

The purpose of this stage is the conversion of the incoming neutrino flux at the detector into an event count. This includes the interaction of the neutrinos, as well as the triggering and event selection criteria (filters), so that the resulting flux is at _analysis level_. 

## Main module

The main module for this stage is `Aeff.py`, which provides a function that takes the flux maps as an input and returns event count maps.
```
def get_event_rates(osc_flux_maps,aeff_service,livetime=None,nu_xsec_scale=None,
                    nubar_xsec_scale=None,aeff_scale=None,**kwargs):
```
The total event __counts__ in each bin is simply calculated as the product of 

![Events](images/events.png)

### Parameters
* `osc_flux_maps`: A set of oscillated flux maps in the format, with one map for each neutrino flavour and parity (i.e `nue, nue_bar, numu, numu_bar, nutau, nutau_bar`).

* `aeff_service`: An _effective area service_ that provides effective areas as a function of energy and cos(zenith) and interaction type. See below for a choice of services.

* `livetime`: the livetime in Julian years (**not** seconds)
* `nu_xsec_scale` : a scale factor for all **neutrino** event rates
* `nubar_xsec_scale` : a scale factor for all **anti-neutrino** event rates
* `aeff_scale` : an overall scale factor (independent of flavour or parity)

### Output
This function returns maps of events counts for each flavour and interaction type.
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
So far, two effective area services are surported, one for __parametrized__ effective areas, the other one builds effective areas directly from MC events.

###AeffServiceMC

```
class AeffServiceMC:
    def __init__(self,ebins,czbins,aeff_weight_file=None,**kwargs):
````

This service takes the energy and cos(zenith) bins as well as a data file (`aeff_weight_file`) in HDF5 format in the constructor. I then reads the events weights for each flavour and interaction type from the datafile and creates histogram of the effective area. The structure of the datafile is
```
flavour / int_type / value
```
where
  *  `flavour` is one of `nue, nue_bar, numu, numu_bar, nutau, nutau_bar`
  *  `int_type` is one of `cc` or `nc`
  *  `values` is one of 
    * `weighted_aeff`: the effective area weight per event (see below)
    * `true_energy` : the true energy of the event
    * `true_coszen` : the true cos(zenith) of the event

**NOTE:** the `weighted_aeff` is directly obtained from the `OneWeight` in `IceTray`; specifically it is calculated as
 
![Weights](images/weight.png)

where 
  * `OneWeight` is the `OneWeight` stored per event in `.i3` data files
  * `N_events` is the number of events **per data file**
  * `N_files` is the total number of data files included
  * the factor 2 reflects that in IceCube simulation both `nu` and `nubar` are simulated in the same runs, while we have seperate effective areas for each of them.

To obtain the effective area, these weights are histrogrammed in the given binning as a function of cos(zenith) and energy. To obtain the effective area in each bin _i_, the sum of weights in this bin is divided with the solid angle and energy range covered by the bin. 

![AeffMC](images/aeffmc.png)

### AeffServicePar
```
class AeffServicePar:
    def __init__(self,ebins,czbins,aeff_egy_par,aeff_coszen_par,**kwargs):
```
This service uses pre-made datatables to describe the energy dependence of the effective area, while the cos(zenith) dependence is described as a functional form (i.e parametrization).
* **Energy dependence**:  `aeff_egy_par` is a dictionary that lists datatables for the 1D energy dependence for each flavour for charged-current interactions  (`nue`,`nue_bar`,`numu`,`numu_bar`,`nutau`,`nutau_bar`), while neutral-current interactions are modelled as flavour-independent (`NC`,`NC_bar`). A simple text file is used for the datatables with the format

    ```
    energy aeff
    ```

   where `energy` is in GeV and `aeff` is in m^2. These files are parsed using `numpy.loadtxt`, and interpolated using 1D linear interpolation with all ranges outside the interpolation range set to 0. 
   
* **cos(zenith) dependence**: `aeff_coszen_par` is a dictionary with one entry per flavour for charged-current and one for neutral-current events (`nue`,`numu`,`nutau`,`NC`). Each entry holds a function definition (typically a `lambda` function) that takes the cos(zenith) angle and returns a modification function for the energy-dependent functions.
    ```
    { 'nue' : 'lambda cz: 0.882 * np.abs(cz)**0.508 + 0.415', ...}
    ```
    The function strings are evaluated using `eval` to return python function objects.

The total effective area is calculated for each bin _i_ by evaluating the both parametrization functions at the bin centers in energy and cos(zenith) and multiplying them, where the cos(zenith) functions are also normalized to unity over the energy range. 

![AeffPar](images/aeffpar.png)


