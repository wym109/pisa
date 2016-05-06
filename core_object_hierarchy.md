# Core object hierarchy
This is a summary of the hierarchy of the objects instantiated by the pisa.core
classes. Indentation indicates that the object lives in the class above.

At the top level is `Analysis`, which is where the actual 'fitting' or 'scanning' etc
happens.

Some examples are given to facilitate understanding.

* [Analysis](pisa/core/analysis.py)
  * [TemplateMaker](pisa/core/template_maker.py) A (e.g. to produce pseudo data)
  * [TemplateMaker](pisa/core/template_maker.py) B (that may be fitted to a template from TemplateMaker A, for example)
    * [Pipeline](pisa/core/pipeline.py) a (e.gi. muons from icc data)
    * [Pipeline](pisa/core/pipeline.py) b (e.g. neutrino MC)
    * [Pipeline](pisa/core/pipeline.py) ...
      * [Stage](pisa/core/stage.py) 1 (e.g. flux) : service x (e.g. integral_preserving)
      * [Stage](pisa/core/stage.py) 2 (e.g. osc) : service y (e.g. prob3)
      * [Stage](pisa/core/stage.py) 3 (e.g. aeff) : service z (e.g. MC)
      * [Stage](pisa/core/stage.py) ...
        * [ParamSet](pisa/core/param.py)
          * [Param](pisa/core/param.py) foo (e.g. energy_scale)
            * Prior (e.g. a gaussian prior with given mu and sigma) 
          * [Param](pisa/core/param.py) bar (e.g. honda_flux_file)
          * [Param](pisa/core/param.py) ...
        * [TransformSet](pisa/core/transform.py) (if applicable)
          * [Transform](pisa/core/transform.py) t1
          * [Transform](pisa/core/transform.py) t2
          * [Transform](pisa/core/transform.py) ...
        * [MapSet](pisa/core/map.py) as (in)/output
          * [Map](pisa/core/map.py) m1 (e.g. numu_cc)
          * [Map](pisa/core/map.py) m2 (e.g. numu_nc)
          * [Map](pisa/core/map.py) ...
            * [MultiDimBinning](pisa/core/binning.py)
              * [OneDimBinning](pisa/core/binning.py) d1 (e.g. energy bins)
              * [OneDimBinning](pisa/core/binning.py) d2 (e.g. coszen bins)
              * [OneDimBinning](pisa/core/binning.py) ...
