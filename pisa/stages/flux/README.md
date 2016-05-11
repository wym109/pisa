# Stage 1: Flux

The purpose of this stage is to get the expected flux produced in the atmosphere for different particles at different energies and angles. It is therefore possible to reweight a dataset using the effective area to investigate uncertainties from using different flux models. The neutrino files come from several different measurements made at different positions on the earth and currently can only use azimuth-averaged data.

## Main Module

The main module for this stage is `Flux.py`, which gets a flux service and then produces splined flux maps for all angles and energies from binned data files. The flux service is implemented with the function:

```flux_service = HondaFluxService(data_file)```

And then the splined flux maps are produced using the function:

```flux_maps = get_flux_maps(flux_service, energy_bins, angle_bins)```

### Parameters

*  `ebins`: Edges of the energy bins in units of GeV, default is 40 edges (39 bins) from 1.0 to 80 GeV in logarithmic spacing.
*  `czbins`: Edges of the cos(zenith) bins, default is 21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.
*  `flux_file`: Input flux file in Honda format.

#### Output

This function returns splined maps of flux (`map`) in bins of energy (`ebins`) and cos(zenith) (`czbins`).
```
{
  "nue" {
        "czbins": []
        "ebins": []
        "map": []
  }
  "nue_bar": {
        "czbins": []
        "ebins": []
        "map": []
  }
  "numu": {
        "czbins": []
        "ebins": []
        "map": []
  }
  "numu_bar": {
        "czbins": []
        "ebins": []
        "map": []
  }
}
```

## Services

### HondaFluxService

```
class HondaFluxService():
    def __init__(self, flux_file=None, smooth=0.05, **params):
        logging.info("Loading atmospheric flux table %s" %flux_file)
```

This class loads a neutrino flux from Honda-styles flux tables in units of [GeV^-1 m^-2 s^-1 sr^-1] and returns a 2D spline interpolated function for each flavour. For now only supports azimuth-averaged input files.

* `get_flux(self, ebins, czbins, prim)`: Get the flux in units [m^-2 s^-1] for the given bin edges in energy(`ebins`) and cos(zenith)(`czbins`) and the primary(`prim`).

## References
"Atmospheric neutrino flux at INO, South Pole and Pyhäsalmi" , Physics Letters B Volume 718, Issues 4–5, 29 January 2013, Pages 1375–1380
"Calculation of atmospheric neutrino flux using the interaction model calibrated with atmospheric muon data", Phys. Rev. D 75, 043006
