# Stage 5: Particle ID

The purpose of this stage is to simulate the event classification of
PINGU, sorting the reconstructed nue CC, numu CC, nutau CC, and NC
events into the track and cascade channels.

## Services

There exists three services for this particular stage: `hist`, `param` and
`kernel`.

### hist
This service takes in events from a **joined** PISA HDF5 file. The current
implementation of this service requires that the nodes on these file
match a certain flavour/interaction combination or "particle signature", which
is `nue_cc, numu_cc, nutau_cc, nuall_nc`. Thus, only the HDF5 files with the
naming convention
```
events__*__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5
```
should be used as input. The structure of the datafile is
```
flavour / int_type / value
```
where
  *  `flavour` is one of `nue, nue_bar, numu, numu_bar, nutau, nutau_bar`
  *  `int_type` is one of `cc` or `nc`
  *  `values` is one of
    * `pid` : the pid score per event
    * `reco_energy` : the reco energy of the event
    * `reco_coszen` : the reco cos(zenith) of the event
    * `weighted_aeff`: the effective area weight per event (see Stage 3, Effective Area)

For the 'joined' event files, the charged current components for the particle
and antiparticle of a specific neutrino flavour are summed so that, for
example, the data in the nodes `nue/cc` and `nue_bar/cc` both contain their
own and each others events. The combined neutral current interaction for all
neutrino flavours is also summed in the same way, so that any `nc` node
contains the data of all neutrino flavours.

Once the file has been read in, for each particle signature, a histogram in the
input binning dimensions and pid score is created and then normalised to one
with respect to the particle signature to give the PID probabilities in each
bin. The input maps are then transformed according to these probabilities to
provide an output containing a map for track-like events `trck` and
shower-like events `cscd`, which is then returned.

Arguments:
  * `params` : `ParamSet` or sequence with which to instantiate a ParamSet.

    Parameters which set everything besides the binning.

    Parameters required by this service are:
      - `pid_events` : `Events` or filepath

        Events object or file path to HDF5 file containing events

      - `pid_ver` : `string`

        Version of PID to use (as defined for this detector/geometry/processing)

      - `pid_remove_true_downgoing` : `bool`

        Remove MC-true-downgoing events

      - `pid_spec` : `PIDSpec`

        PIDSpec object which specifies the PID specifications.
        Either `pid_spec` or `pid_spec_source` can be used to define the PID specifications

      - `pid_spec_source` : filepath

        Resource for loading PID specifications

      - `compute_error` : `bool`

        Compute histogram errors

  * `input_binning` : `MultiDimBinning`

    Arbitrary number of dimensions accepted. Contents of the input `pid_events`
    parameter defines the possible binning dimensions. Name(s) of given
    binning(s) must match to a reco variable in `pid_events`.

  * `output_binning` : `MultiDimBinning`
  * `transforms_cache_depth` : `int` >= 0
  * `outputs_cache_depth` : `int` >= 0
