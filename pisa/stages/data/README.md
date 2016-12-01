# Stage: data

loading data events from files and provide the outputs in the form of a PISA MapSet

## Services

Two services are available

### data

Load data events and just histogram them (no scaling or systematics applied)

### icc

Load data events to be used to model the atmospheric muon background. The events are scaled by `atm_muon_scale` and `livetime`, and additional uncertainties are provided given the finite statistics and also an alternative icc definition to generate a shape uncertainty term.
