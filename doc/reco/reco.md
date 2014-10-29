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
* Depending on the chosen reco mode, one of [`mc_file` (`HDF5`), `param_file` (`JSON`), `kernel_file` (`JSON`)] 
 has to be specified, providing the actual reco information.
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

* `get_reco_kernels`: This method is called to construct the reco kernels, 
 i.e. a 4D histogram of true (1st and 2nd axis) vs. reconstructed (3rd and
 4th axis) energy (1st and 3rd axis) and cos(zenith) (2nd and 4th axis). 
 It individually implemented in the derived classes, since the way the reco 
 kernels are generated is the depends on the reco method. 
* `check_kernels`: Test whether the reco kernels have the correct shape
 (see above).
* `normalize_kernels`: Ensure that all reco kernels are normalized.
* `get_reco_maps(true_event_maps, recalculate=False, **kwargs)`: Apply the 
 reconstruction kernels to the "true" event histograms and return the 
 "reconstructed" histograms. If `recalculate` is `True`, call `recalculate_kernels` 
 before and pass `kwargs` to it (might be needed when scanning a systematic 
 parameter related to reconstruction).
* `recalculate_kernels(**kwargs)`: Call `get_reco_kernels` again, passing `kwargs` 
 and do all necessary checks. If new kernels are corrupted, stick with the old ones.
* `store_kernels(filename)`: Store reconstruction kernels in `JSON` format 
 to re-use them later.

The different derived reconstruction services (i.e. different implementations 
of `get_reco_kernels`) are:

### MCRecoService

Argument(s):
* `simfile`: MC file in `HDF5` format containing data from all flavours for a 
 particular instumental geometry

Creates the 4D reco kernel (see above) from histogramming `simfile`.
Expects the file format to be:
```
{'nue': {
   'cc': {
     'true_energy': np.array,
     'true_coszen': np.array,
     'reco_energy': np.array,
     'reco_coszen': np.array
     },
   'nc': {...
     }
   },
 'nue_bar' {...},
 ...
}
```

### ParamRecoService

Argument(s):
* `paramfile`: `JSON` file containing the parametrizations as strings defining 
 python lambda functions. `numpy` can be used as `np`, also `scipy.norm`.
* `e_reco_scale`, `cz_reco_scale`: `double`, scales the width of the energy (zenith) 
 reco by a given factor (default: 1.0).

Assumes all resolutions can be described by double gaussians whose parameters depend 
on the true neutrino energy. The input file should have the format
```
{'nue': {
   'cc': {
     'energy': {
       'loc1': 'lambda E: some_func(E)',
       'loc2': '...',
       'width1': '...',
       'width2': '...',
       'fraction': '...'
     },
     'coszen': {...}
   }
   'nc': {...
   }
 },
 'nue_bar' {...},
 ...
}
```
such that the `'lambda E: ...'` definitions can be read via `eval()`. The double 
gaussians then take the form
```
   F(E_true) = (1-fraction)*exp((E_true-loc1)^2/width1^2)
                 + fraction*exp((E_true-loc2)^2/width2^2
```
with `fraction`, `loc1`, `loc2`, `width1`, and `width2` all being functions of 
the true energy as defined in the parametrization file.

### KernelFileRecoService

Argument(s):
* `kernelfile`: `JSON` file containing a previously calculated reconstruction 
 kernel as produced by the `store_kernels` method.

Loads the kernels from disk and uses them for reconstruction.
