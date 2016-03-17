pisa
====

PINGU Simulation and Analysis (PISA) is software written for performing predictive analyses with the proposed PINGU upgrade to the IceCube neutrino observatory. PISA is more broadly applicable, however, and can be used for analyses employing the existing IceCube detector as well as other similar detectors.

## Simulation chain

![Simulation chain](doc/PINGUSimulationChain.png "Simulation chain")

The original drawing is [here](https://docs.google.com/drawings/edit?id=1RxQj8rPndwFygxw3BUf4bx5B35GAMk0Gsos_BiJIN34).

## PISA Terminology
* **Map**: Dictionary-like object with key `'map'` containing a 2D histogram in energy and the cosine of the zenith angle (coszen); other keys in the map contain metadata about the histogram (e.g. `'ebins'`, `'czbins'`, etc.). This can be in terms of *true* or *reconstructed* neutrino energy & coszen, depending upon the situation. Sometimes a collection of maps is casually (but misleadingly) called a "map". Note that the histogram is a 2D numpy array with first dimension energy (increasing index 

* **Map set**: Dictionary-like object whose items are the maps used as inputs to or produced as outputs from the stages. Each map is stored in the collection by a key, where the key names vary by service and the dict can be nested and painful to access. Smarter objects than vanilla dicts should be used to handle sets of maps to make users/developers lives less painful.

* **Stage**: Each stage is a critical part of the process by which we detect neutrinos. For example, atmospheric neutrinos that pass through the earth will oscillate partially into different flavors prior to reaching the IceCube/PINGU detector. This part of the process is called the **oscillations** stage.

* **Service**: A particular *implementation* of a stage is termed a **service**. Using the oscillations stage as an example, a service that implements that stage is `pisa.oscillcations.Prob3GPUOscillationService.Prob3GPUOscillationService` (WeAreMastersOfConcision is our middle name).

* **Resource**: A file with settings, simulated events, parameterizations, metadata, or etc. that is used by one of the services, the template maker, the minimizer, .... Resources are found in the `$PISA/pisa/resources` directory, where a subdirectory exists for each stage (and several directories exist for resources used for other purposes).

* **Template settings**: The collection of all parameters required to instantiate all services in a simulation chain. Can be a PISA `template_settings.ini` file or a nested dictionary; formats for these are defined below.

## Implementation Details

![Stage architecture](doc/stage_architecture.png "Stage architecture")

### `pisa.analysis.TemplateMaker`
Instantiates and contains services implementing the simulation chain's stages

The functionality of the template maker is best described through its key methods:
* Can load parameters en masse from a ***template settings*** file or dictionary
* `generate_template` method produces a template based upon all services contained in the template maker (and the state of their parameters at the time of the method call).
* `match_to_data` method invokes a minimizer to adjust contained stages' free parameters to best match (either via LLH or chi2 criteria) a reference template 
* `scan` method for scanning over a parameter or parameters
* `set_params`, `get_params`, `get_free_params` methods for working with parameters on a lower level

### Stages

* There is one base class for all stages: `pisa.stage.Stage` which implements the most basic functionality of a stage, including instantiaton of the two caches pictured above
  * `set_params`, `get_params`, `get_free_params` methods for working with parameters
* Each stage has its own base class, e.g. FluxServiceBase, RecoServiceBase, etc.
  * `apply()` must be aware of all possible systematics. Their implementations might logically be via other methods within the base class to keep `apply` succinct, but in order to produce a meaningful hash for a transform, `apply` needs to account for *all* the ways that the transform might be modified.
    * Each of which should be called from within the `apply()` method (see below).
  * Implements a method called `apply(<input map>, **kwargs)` (except FluxServiceBase.apply() does *not* take `<input map>`)

* Each particular implementation ("mode") for a stage derives from the stage's base class
  * Each mode must determine how to (efficiently) compute a unique transform hash for its produced data
    * E.g., all of the parameters used for generating the transform should uniquely describe the transform

* Variations
  * The flux stage does not take an input map
  * The oscillation stage does not first produce a "nominal-systematics" transform that then gets computed and then transformed by systematics; this is all one step.

### Caching
Caching all services' transforms and results can speed up the template-making process by a factor of 2-3.

* Transform and result caches are memory-based, least-recently-used caches
* Disk storage for the nominal-systematics (aka no-systematics) transform defaults to `pingu/resources/<stage_type>/.cache/<service_name>.nominal_transform.hdf5`, unless a stage implements a different naming convention

#### Hashes
Caching requires the *fast* generation of unique identifiers for each item stored in the cache. Any hash collisions will cause corruption of the results, so hashing needs to be done carefully so as to make the odds of a collision vanishingly small.

* **Hashes for transforms**: Each service is respoinsible for generating a unique hash for its transform (e.g., based upon parameters used to produce the transform).
* **Hashes for sets-of-maps**: The service that produces a set of maps is also responsible for producing the map-set's hash. The hash is derived from a tuple of the input maps' hash and the transform hash. As this logic is consistent across all stages & services (so long as a flux input hash is used), it is implementated in the generic `pisa.stage.Stage` base class.
* The class `pisa.utils.utils.DictWithHash` is provided for conveniently passing transforms and map sets around with hashes attached. Note that it is the user's responsibility to ensure that the `hash` attribute of those objects is not out of sync with respect to the data contained within them. This can done manually after updating the data by calling the `DictWithHash` method `update_hash` with a simple object or hash as its argument (see help for that method for more details). To ensure such consistency between data and hash, it is recommended to modify the data contents in the `try` clause and update the hash in the `else` clause of a `try-except-else` code block.

## Installation
### Requirements

To install this package, you'll need to have the following requirements
installed

* [pip](https://pip.pypa.io/) -- version > 1.2 recommended
* [swig](http://www.swig.org/) -- install with `--universal` option
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/) -- version > 0.12 recommended
* [hdf5](http://www.hdfgroup.org/HDF5/) -- install with `--enable-cxx` option
* [h5py](http://www.h5py.org/) -- install via pip
* [cython](http://cython.org/) -- install via pip

If you are working on OSX, we suggest [homebrew](brew.sh/) as a package manager, which supports all of the non-python packages above. 

### Obtaining `pisa`

**User mode:**

Use this if you just want to run `pisa`, but don't want to edit it. First pick a revision from [this github page](https://github.com/tarlen5/pisa/releases). Then run this command in your shell, to directly install pisa from github.
```
pip install git+https://github.com/tarlen5/pisa@<release>#egg=pisa
```

where

* `<release>` is the release number, e.g. `2.0.0`

**Developer mode:**

Also in developer mode, you can directly install via `pip` from github. In order to contribute, you'll first need your own fork of the `pisa` repository.

1. Create your own [github account](https://github.com/)
1. Navigate to the [pisa github page](https://github.com/tarlen5/pisa) and fork the repository by clicking on the ![fork](doc/ForkButton.png) button
1. Now go to your terminal and install `pisa` from your fork using the following commands
```
pip install [ --src <your/source/dir> --editable ] git+https://github.com/<user>/pisa@<branch>#egg=pisa 
cd <your/source/dir>/pisa && git checkout <branch>
```

where

* `<user>` is the user name you picked on github.
* `<branch>` is the branch you'd like to install. This could be either one of
  the releases, or `master`
* `--editable` tells `pip` to make an editable installation, i.e instead of
  installing directories in your `<site-packages>` directory, `pip` will install
  the source in `<your/source/dir>` and link this from the `<site-packages>`.
  This way, if you change the source the changes will be automatically reflected
  when you run the code. __NOTE__: a subdirectory `pisa` that holds your files will be created within `<your/source/dir>`.
* As for now (`pip <= 1.5.6`) the additional `git checkout <branch>` is required as `pip`
  will checkout the specific latest commit in the branch you give, rather than
  the `HEAD` of that branch. You are thus left with a _detached HEAD_, and can
  not commit to the branch you check out.
  
__Notes:__

* You can work with your installation using the usual git commands (pull,
push, etc.). Note however, that these won't rebuild any of the extension (i.e.
_C/C++_) libraries. If you want to recompile these libraries, simply run
<br>```cd <your/source/dir>/pisa && python setup.py build_ext --inplace```

* If you did not install `pisa` in a virtual environment, then the package will
  be installed alongside with your other python packages. This typically means
  that you'll need super-user priviledges to install, i.e.<br>
  ```sudo pip install ...```<br>
  If your are using above with the `--editable` option, the source files will
  also be installed by the super-user, which means you might not be able to edit
  them. In this case, just<br>
  ```cd <your/source/dir> && sudo chown -R <user> pisa```<br>
  where `<user>` obviously just is your user name.

### Updating `pisa`

**Developer mode:**

To upgrade to new version of pisa, just run the install command again with a new version number and the `--upgrade` flag. 

**Developer mode:**

The simplest way to update pisa is just to checkout the version you want in git. However, this will not update the version number for `pip`, and it also won't recompile the `prob3` oscillation package. In order to get those updated, the best way is to

1. Make sure your _fork_ of pisa on github has the right version
2. Run the install command again
```
pip install --src <your/source/dir> --editable git+https://github.com/<user>/pisa@<branch>#egg=pisa 
```
Git will automatically realize that there is already a version of `pisa` in `<your/source/dir>`, so it will just update, but won't delete any of the files you have in there. 

## Data formats

This [working document](https://docs.google.com/document/d/1qPVrtECZUDHVVJz_CncCemqmeHk5nOgPlceIU7-jNGc/edit#) describes some of the data formats that will be used in the different steps.
