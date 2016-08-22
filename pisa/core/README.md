# Core object hierarchy
This is a summary of the hierarchy of the objects instantiated by the pisa.core
classes. Indentation indicates that the object lives in the class above.

At the top level is `Analysis`, which is where the actual 'fitting' or 'scanning' etc
happens.

Some examples are given to facilitate understanding.

* [Analysis](/pisa/core/analysis.py)
  * [DistributionMaker](/pisa/core/distribution_maker.py) A (e.g. to produce pseudo data distribution)
  * [DistributionMaker](/pisa/core/distribution_maker.py) B (that may be fitted to a distribution from DistributionMaker A, for example)
    * [ParamSet](/pisa/core/param.py)
      * [Param](/pisa/core/param.py)
      * [Param](/pisa/core/param.py) ...
    * [Pipeline](/pisa/core/pipeline.py) a (e.g. muons from data (inv. corridor cut))
    * [Pipeline](/pisa/core/pipeline.py) b (e.g. neutrino MC)
    * [Pipeline](/pisa/core/pipeline.py) ...
      * [ParamSet](/pisa/core/param.py)
      * [Stage](/pisa/core/stage.py) s0: flux / service honda (inherits from Stage class)
      * [Stage](/pisa/core/stage.py) s1: osc / service prob3cpu (inherits from Stage)
      * [Stage](/pisa/core/stage.py) s2: aeff / service hist (inherits from Stage)
      * [Stage](/pisa/core/stage.py) ...
        * [ParamSet](/pisa/core/param.py)
          * [Param](/pisa/core/param.py) foo (e.g. energy_scale)
          * [Param](/pisa/core/param.py) bar (e.g. honda_flux_file)
          * [Param](/pisa/core/param.py) ...
            * [Prior](/pisa/core/prior.py) (e.g. a gaussian prior with given mu and sigma) 
        * [Events](/pisa/core/events.py) (if used by stage)
        * [TransformSet](/pisa/core/transform.py) (if applicable)
          * [Transform](/pisa/core/transform.py) t0 : BinnedTensorTransform (inherits from Transform)
          * [Transform](/pisa/core/transform.py) t1 : BinnedTensorTransform (inherits from Transform)
            * [MultiDimBinning](/pisa/core/binning.py) input_binning
              * [OneDimBinning](/pisa/core/binning.py) d0 (e.g. true/reco_energy)
              * [OneDimBinning](/pisa/core/binning.py) d1 (e.g. true/reco_coszen)
            * [MultiDimBinning](/pisa/core/binning.py) output_binning
        * [MapSet](/pisa/core/map.py) as input to / output from stage
          * [Map](/pisa/core/map.py) m0 (e.g. numu_cc)
          * [Map](/pisa/core/map.py) m1 (e.g. numu_nc)
          * [Map](/pisa/core/map.py) ...
            * [MultiDimBinning](/pisa/core/binning.py)
              * [OneDimBinning](/pisa/core/binning.py) d0 (e.g. true/reco_energy)
              * [OneDimBinning](/pisa/core/binning.py) d1 (e.g. true/reco_coszen)
              * [OneDimBinning](/pisa/core/binning.py) ...
