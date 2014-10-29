# Stage 4: Reconstruction

The purpose of this stage is to apply an assumed detector resolution to 
the incoming flux to convert the events' "true" energies and directions 
into the "reconstructed" ones.

## Main module

The main module for this stage is `Reco.py`, which takes a true, 
triggered event rate file (output of `Aeff.py`) as input and produces a 
set of reconstructed templates of nue CC, numu CC, nutau CC, and NC 
events. Note that in this stage, the counts of neutrinos and corresponding 
anti-neutrinos are summed.

### Parameters

* `event_rate_maps`: Event rate input file (`JSON`) with the following fields:
```
{"nue": {'cc':{'czbins':[], 'ebins':[], 'map':[]},'nc':...},
 "numu": {...},
 "nutau": {...},
 "nue_bar": {...},
 "numu_bar": {...},
 "nutau_bar": {...} }
```
* `mode`: Defines which service to use for the reconstruction. One of 
 [`MC`, `param`, `stored`]. Details below.
* Depending on the chosen reco mode, one of [`mc_file` (HDF5), 
 `param_file` (JSON), `kernel_file` (JSON)] has to be specified, providing 
 the actual reco information.
* `e_reco_scale`, `cz_reco_scale`: Scales the width of the energy (zenith) 
 reco by a given factor (default: 1.0). Currently only supported by the 
 parametrized reconstruction service.

### Output

This stage returns maps of reconstructed event counts for each "flavour":

```
{"nue_cc": {'czbins':[], 'ebins':[], 'map':[]},
 "numu_cc": {...},
 "nutau_cc": {...},
 "nuall_nc": {...}}
```

## Services

The base service class `RecoServiceBase` contains everything that is 
related to actually applying the reconstruction kernels to the data, as 
well as sanity checks of the kernels. The methods common to all reco 
services (which are all implemented here) are:
