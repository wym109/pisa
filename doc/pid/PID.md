# Stage 5: Particle ID

The purpose of this stage is to simulate the event classification of 
PINGU, sorting the reconstructed nue CC, numu CC, nutau CC, and NC 
events into the track and cascade channels.

## Main module

The main module for this stage is `PID.py`, which takes reconstructed 
event rate file (output of `Reco.py`) as an input and sorts it into the 
event classes PINGU can actually distinguish: track-like (`'trck'`) and 
shower-like (`'cscd'`). There is the possibility to open a third channel 
(`'unkn'`) for events that passed neither track nor shower selection cuts.

Therefore, it provides the function `get_pid_maps`, which is called as
```
get_pid_maps(reco_events, pid_service=None, recalculate=False, 
             return_unknown=False, **kwargs)
```
where
* `reco_events`: dict holding reconstructed event rates with the 
following fields:
```
{"nue_cc": {'czbins':[], 'ebins':[], 'map':[]},
 "numu_cc": {...},
 "nutau_cc": {...},
 "nuall_nc": {...} }
```
* `pid_service`: a pid service to use
* `recalculate`: whether to re-calculate the PID kernels before doing the 
 identification
* `return_unknown`: whether to return a channel of 'unknown' signature

### Parameters

* `reco_event_maps`: Reconstructed event rate input file (`JSON`) whose 
 content will be passed to `get_pid_maps`.
* `mode`: Defines which service to use for the particle identification, 
 one of [`param`, `stored`]. Details below.
* Depending on the chosen PID mode, one of [`param_file` (`JSON`), `kernel_file` (`JSON`)] 
 has to be specified, providing the actual PID information.

### Output

This stage returns maps of event counts for both signatures:
```
{"cscd": {'czbins':[], 'ebins':[], 'map':[]},
 "trck": {'czbins':[], 'ebins':[], 'map':[]}}
```

## Services

The base service class `PIDServiceBase` contains everything that is 
related to actually applying the PID kernels to the data, as 
well as sanity checks of the kernels. The methods common to all PID 
services (which are all implemented here) are:

* `get_pid_kernels`: This method is called to construct the PID kernels, 
 i.e. two 2D histograms of ID probability as track and cascade as function
 of energy and cos(zenith) for each flavour. It is individually 
 implemented in the derived classes, since the way the PID kernels are 
 generated depends on the PID method.
* `check_kernels`: Test whether the PID kernels have the correct shape
 (see above) and the sum of the track and cascade ID probabilities is not 
 larger than 1.
* `recalculate_kernels(**kwargs)`: Call `get_pid_kernels` again, passing 
 `kwargs` and do all necessary checks. If new kernels are corrupted, 
 stick with the old ones.
* `store_pid_kernels(filename)`: Store PID kernels in `JSON`  format to 
 re-use them later.

The different derived PID services (i.e. different implementations 
of `get_pid_kernels`) are:

### PIDServiceParam
Argument(s):
* `pid_paramfile`: `JSON` file containing the PID functions as strings for 
all flavours. This should look like
```
{"nue_cc": {
   "cscd": "lambda E: some_func_of(E)",
   "trck": "lambda E: some_other_func_of(E)"
 },
 "numu_cc": {...},
 "nutau_cc": {...},
 "nuall_nc": {...}
}
```
such that it can be evaluated via pythons `eval()` function. In the function 
definitions, `numpy` (as `np`) and `scipy.stats` can be used.
* `PID_scale` (default: 1): Systematic parameter, scales *all* PID functions 
 by this factor. Reflects the fact that particle ID might be more or less 
 effective than assumed. Note that `PID_scale != 1` will lead to losing or 
 generating additional  events if the `unkn` channel is not enabled!
* `PID_offset` in GeV (default: 0): Systematic parameter, shifts all PID 
 functions in energy. Stands for the possibility that particle ID, which can 
 usually be described by a step-like function, might become effecive at higher 
 (lower) energies than assumed.

### PIDServiceKernelFile

Argument(s):
* `pid_kernelfile`: `JSON` file containing a previously calculated PID 
 kernel as produced by the `store_pid_kernels` method.

Loads the kernels from disk and uses them for particle ID.
