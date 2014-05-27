pisa
====

The common PINGU simulation and analysis code for the neutrino mass hierarchy.

## Simulation chain

![Simulation chain](doc/PINGUSimulationChain.png "Simulation chain")

The original drawing is [here](https://docs.google.com/drawings/edit?id=1RxQj8rPndwFygxw3BUf4bx5B35GAMk0Gsos_BiJIN34).

## Installation
### Requirements

To install this package, you'll need to have the following requirements
installed

* [pip](https://pip.pypa.io/)
* [swig](http://www.swig.org/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [hdf5](http://www.hdfgroup.org/HDF5/) -- install with `--enable-cxx` option
* [h5py](http://www.h5py.org/) -- install via pip

If you are working on OSX, we suggest [homebrew](brew.sh/) as a package manager, which supports all of the non-python packages above. 

### Obtaining `pisa`
You can directly install via `pip` from github, using the following command
```
pip install [ --src <your/source/dir> --editable ] git+https://github.com/sboeser/pisa@<branch>#egg=pisa 
```

where

* `<branch>` is the branch you'd like to install. This could be either one of
  the releases, or `master`
* `--editable` tells `pip` to make an editable installation, i.e instead of
  installing directories in your `<site-packages>` directory, `pip` will install
  the source in `<your/source/dir>` and link this from the `<site-packages>`.
  This way, if you change the source the changes will be automatically reflected
  when you run the code. __NOTE__: a subdirectory `pisa` that holds your files will be created within `<your/source/dir>`.
  
 


To update your installation to a later version, just run

```
pip install pisa --upgrade
```

## Data formats

This [working document](https://docs.google.com/document/d/1qPVrtECZUDHVVJz_CncCemqmeHk5nOgPlceIU7-jNGc/edit#) describes some of the data formats that will be used in the different steps.
