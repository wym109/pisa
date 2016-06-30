# Stage 4: Reconstruction

The purpose of this stage is to apply an assumed detector resolution to
the incoming flux to convert the events' "true" energies and directions
into the "reconstructed" ones.

## Services

Only one service is currently supported in PISA for dealing with converting from truth to reconstructed variables: `hist`.

### hist

The `hist` service (and indeed all `hist` services in PISA) takes Monte Carlo events as inputs and directly histograms them to create the transform. For the reconstruction stage this transform takes the form of a 2N-dimensional histogram (N is the number of dimensions in your analysis) which matches from truth to reconstructed variables. In it, each bin along the truth direction is a distribution over all of the reconstructed space.

Consider the 1-dimensional case for simplicity where we have a reconstruction kernel which transforms from true cosZenith to reconstructed cosZenith. Say we have 10 bins in between -1 and 0 GeV at truth level and an equivalent 10 bins at reco level. The first bin of the reconstruction kernel will contain a map over -1 to 0 in reconstructed cosZenith that is all of the events from the truth bin -1 to -0.9. Thus it tells us the contribution to the reconstructed distribution from every truth bin. These maps in reconstructed space are normalised to the total number of events from that truth bin.

For this service, one must specify where to pull these Monte Carlo events from. Typically this should be one of the files in the `resources/events/` directory. One can also specify the choice of weights to be used in constructing these histograms. Note that this must be one of the columns in the events file, or you can set it to `None` to produce the histograms unweighted.

