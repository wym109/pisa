# Core object hierarchy
This is a summary of the hierarchy of the objects instantiated by the pisa.core
classes. Indentation indicates that the object lives in the class above.

At the top level is `Analysis`, which is where the actual 'fitting' or 'scanning' etc
happens.

Some examples are given to facilitate understanding.

* [Analysis](/pisa/core/analysis.py)
  * [TemplateMaker](template_maker.py) A (e.g. to produce pseudo data)
  * [TemplateMaker](template_maker.py) B (that may be fitted to a template from TemplateMaker A, for example)
    * [Pipeline](pipeline.py) a (e.gi. muons from icc data)
    * [Pipeline](pipeline.py) b (e.g. neutrino MC)
    * [Pipeline](pipeline.py) ...
      * [Stage](stage.py) 1 (e.g. flux) : service x (e.g. integral_preserving)
      * [Stage](stage.py) 2 (e.g. osc) : service y (e.g. prob3)
      * [Stage](stage.py) 3 (e.g. aeff) : service z (e.g. MC)
      * [Stage](stage.py) ...
        * [ParamSet](param.py)
          * [Param](param.py) foo (e.g. energy_scale)
            * [Prior](prior.py) (e.g. a gaussian prior with given mu and sigma) 
          * [Param](param.py) bar (e.g. honda_flux_file)
          * [Param](param.py) ...
        * [TransformSet](transform.py) (if applicable)
          * [Transform](transform.py) t1
          * [Transform](transform.py) t2
          * [Transform](transform.py) ...
        * [MapSet](map.py) as (in)/output
          * [Map](map.py) m1 (e.g. numu_cc)
          * [Map](map.py) m2 (e.g. numu_nc)
          * [Map](map.py) ...
            * [MultiDimBinning](binning.py)
              * [OneDimBinning](binning.py) d1 (e.g. energy bins)
              * [OneDimBinning](binning.py) d2 (e.g. coszen bins)
              * [OneDimBinning](binning.py) ...
