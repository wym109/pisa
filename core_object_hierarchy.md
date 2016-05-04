# Core object hierarchy
This is a summary of the hierarchy of the objects instantiated by the pisa.core
classes. Indentation indicates that the object lives in the class above.

At the top level is `Analysis`, which is where the actual 'fitting' or 'scanning' etc
happens.

Some examples are given to facilitate understanding.

* Analysis
  * TemplateMaker A (e.g. to produce pseudo data)
  * TemplateMaker B (that may be fitted to a template from TemplateMaker A, for example)
    * Pipeline a (e.gi. muons from icc data)
    * Pipeline b (e.g. neutrino MC)
    * Pipeline ...
      * Stage 1 (e.g. flux) : service x (e.g. integral_preserving)
      * Stage 2 (e.g. osc) : service y (e.g. prob3)
      * Stage 3 (e.g. aeff) : service z (e.g. MC)
      * Stage ...
        * ParamSet
          * Param foo (e.g. energy_scale)
            * Prior (e.g. a gaussian prior with given mu and sigma) 
          * Param bar (e.g. honda_flux_file)
          * Param ...
        * TransformSet (if applicable)
          * Transform t1
          * Transform t2
          * Transform ...
        * MapSet as (in)/output
          * Map m1 (e.g. numu_cc)
          * Map m2 (e.g. numu_nc)
          * Map ...
            * MultiDimBinning
              * OneDimBinning d1 (e.g. energy bins)
              * OneDimBinning d2 (e.g. coszen bins)
              * OneDimBinning ...
