# PISA

PINGU Simulation and Analysis (PISA) is software written for performing analyses based upon Monte Carlo simulations of the [IceCube neutrino observatory](https://icecube.wisc.edu/), including the [DeepCore](https://arxiv.org/abs/1109.6096) and proposed [PINGU](https://arxiv.org/abs/1401.2046) low-energy in-fill arrays (as well as other similar neutrino detectors).

PISA was originally developed to cope with low-statistics Monte Carlo (MC) for PINGU using parameterizations of the MC, but its methods should apply equally as well to high-MC situations, and the PISA architecture is general enough to easily accomoodate traditional reweighted-MC-style analyses.

## Index

* [More about PISA](#more-about-pisa)
   * [Analysis types](#analysis-types)
   * [How an analysis is structured](#how-an-analysis-is-structured)
   * [An example parameterized-MC analysis pipeline](#an-example-parameterized-mc-analysis-pipeline)
   * [More information about analysis](#more-information-about-analysis)
* [Installation](#installation)
  * [Requirements](#requirements)
  * [Install Python](#install-python)
  * [Set up your environment](#set-up-your-environment)
  * [Github setup](#github-setup)
    * [SSH vs. HTTPS access to repository](#ssh-vs-https-access-to-repository)
      * [Set up password-less access (SSH)](#set-up-password-less-access-ssh)
      * [Set up password caching (SSH or HTTPS)](#set-up-password-caching-ssh-or-https)
  * [Obtain PISA sourcecode](#obtain-pisa-sourcecode)
    * [Developing PISA: Forking](#developing-pisa-forking)
    * [Using but not developing PISA: Cloning](#using-but-not-developing-pisa-cloning)
  * [Install PISA](#install-pisa)
* [Memetic description of PISA](#memetic-description-of-pisa)

Visit the project Wiki for the latest documentation (not yet folded into the codebase):
* [PISA Cake wiki](https://github.com/jllanfranchi/pisa/wiki)


## More about PISA

PISA implements a modular architecture wherein users can combine one or more analysis pipelines into distribution makers to make "data" and—separately—template distributions.
Within each pipeline, users can choose among several implementations for each of the stages they choose to include.

Finally, multiple types of analysis can be performed using the generated distributions to ultimately characterize the ability of the detector to make a measurement.

### Analysis types

PISA implements both what we call ***parameterized-Monte Carlo (MC) stages*** and ***reweighted-MC stages***.
In the former, distributions (and not individual event weights) are modified to reflect the effects of each analysis stage.
In the latter, the individual events' weights and properties (such as reconstructed energy) are modified directly to reflect the effects of the detector, and only in the end are the events histogrammed to characterize their distribution.

See the analysis guide for more explanaton of the difference between the two.

### How an analysis is structured

All of the analyses possible utilize a "data" distribution.
This can come from an actual measurement or by *injecting* a set of assumed-true values for the various parameters into the analysis pipeline(s) and producing what is called ***Asimov data***—the expected distribution given those parameter values—or ***pseudo data***, which is Asimov data but with random (Poisson) fluctuations applied.
A minimizer attempts to match the "data" distribution with a template by varying the parameters used to generate the template.
The "closeness" of match between the generated template and the "data" distribution is measured by a criterion such as chi-squared or log likelihood.

An important question is the significance of the experiment to measure one or more of the above parameters (the *measured parameters*).
To do this, the measured parameters are fixed successively to a range of values and, at each value, the matching process above is repeated with all other parameters—the *nuissance parameters*—allowed to vary.
This shows sensitivity of the criterion to the measured parameters, and hence the ability for the experiment to measure those parameters.
Put another way: The more the closeness creiterion varies with a change in the measurement parameters (after the nuissance parameters have done their best to try to make the templates all look like the data), the better able the experiment is to distinguish between values of the measured parameters.

### An example parameterized-MC analysis pipeline

![Parameterized-MC analysis pipeline](doc/PINGUSimulationChain.png "Parameterized-MC analysis pipeline")

The original drawing is [here](https://docs.google.com/drawings/edit?id=1RxQj8rPndwFygxw3BUf4bx5B35GAMk0Gsos_BiJIN34).

### More information about analysis

An excellent (and far more detailed) description of the analysis process is maintained by Elim Cheung with particular application to IceCube/DeepCore atmospheric neutrino measurements [here](http://umdgrb.umd.edu/~elims/Fitter/Basics).
She wrote her own fitter to perform these tasks. You can evaluate her ezFit analysis software as an alternative to (or as a complementary tool for comparing results with) PISA [here](http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/elims/ezfit).


## Installation

### Requirements

To install this package, you'll need to have the following non-python requirements
* [git](https://git-scm.com/)
* [swig](http://www.swig.org/)
* [hdf5](http://www.hdfgroup.org/HDF5/) — install with `--enable-cxx` option

In Ubuntu Linux, you can install these via
```bash
sudo apt-get install git hdf5
```
although you can also obtain `hdf5` by installing the Anaconda Python distribution (see below).

The Python requirements are
* [python](http://www.python.org) — version 2.7.x required
* [pip](https://pip.pypa.io/) — version > 1.2 recommended
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/) — version > 0.12 recommended
* [h5py](http://www.h5py.org/)
* [cython](http://cython.org/)
* [uncertainties](https://pythonhosted.org/uncertainties/)
* [pint](https://pint.readthedocs.org/en/0.7.2/)

Optional dependencies to enable add-on features are
* [PyCUDA](https://mathema.tician.de/software/pycuda)
* [openmp](http://www.openmp.org)

Obtaining packages and handling interdependencies is easiest if you use a Python distribution, such as [Anaconda](https://www.continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy).
Although the selection of maintained packages is smaller than if you use the `pip` command to obtain packages from the Python Package Index (PyPi), you can stil use `pip` even if you use a Python distribution.

### Install Python
There are many ways of obtaining Python and many ways of installing it.
Here we'll present two options, but this is by no means a complete list.

* Install Python 2.7.x from the Python website [https://www.python.org/downloads](https://www.python.org/downloads/)
* Install Python 2.7.x from the Anaconda distribution following instructions [here](https://docs.continuum.io/anaconda/install)

### Set up your environment
* Create a "parent" directory (the directory into which you wish for the PISA sourcecode to live).
Note that subsequent steps will create a directory named `pisa` within the parent directory you've chosen, so you don't need to create the actual `pisa` directory yourself.
```
mkdir -p <parent dir>
```

* To make life easier in the future (and to make these instructons easy to follow), define the environment variable `PISA`.
E.g., for the bash shell, edit your `.bashrc` file and add the line
```
export PISA=<parent dir>/pisa
```
Load this variable into your *current* environment by sourcing your `.bashrc` file:
```bash
. ~/.bashrc
```
(it will be loaded autmatically for all new shells).

### Github setup
1. Create your own [github account](https://github.com/)
1. Obtain access to the `WIPACrepo/pisa` repository by emailing Sebastian Böeser [sboeser@uni-mainz.de](mailto:sboeser@uni-mainz.de?subject=Access to WIPACrepo/pisa github repository)

#### SSH vs. HTTPS access to repository
You can interact with Github repositories either via SSH (which allows password-less operation) or HTTPS (which gets through firewalls that don't allow for SSH).
To choose one or the other just requires a different form of the repsitory's URL (the URL can be modified later to change method of access if you change your mind).

##### Set up password-less access (SSH)
If you use the SSH URL, you can avoid passwords altogether by uploading your public key to Github:

1. [Check for an existing SSH key](https://help.github.com/articles/checking-for-existing-ssh-keys/)
1. [Generate a new SSH key if none already exists](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
1. [Add your SSH key to github account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account)
1. [Test the ssh connection](https://help.github.com/articles/testing-your-ssh-connection)

##### Set up password caching (SSH or HTTPS)
Git 1.7.10 and later allows storing your password for some time in memory so you aren't asked every time you interact with Github via the command line.
Follow instructions [here](https://help.github.com/articles/caching-your-github-password-in-git/).
This is particularly useful for HTTPS or if you use SSH but do not wish to store a key pair on the computer/server you use.

### Obtain PISA sourcecode

#### Developing PISA: Forking
If you wish to modify PISA and contribute your code changes back to the PISA project (*highly recommended!*), fork `WIPACrepo/pisa` from Github.
*(How to work with the `cake` branch of PISA will be detailed below.)*

Forking creates your own version of PISA within your Github account.
You can freely create your own *branch*, modify the code, and then *add* and *commit* changes to that branch within your fork of PISA.
When you want to share your changes with `WIPACrepo/pisa`, you can then submit a *pull request* to `WIPACrepo/pisa` which can be merged by the PISA administrator (after the code is reviewed and tested, of course).

* Navigate to the [PISA github page](https://github.com/wipacrepo/pisa) and fork the repository by clicking on the ![fork](doc/ForkButton.png) button.
* From a terminal, change into the "parent" directory.<br>
`cd <parent dir>`
* Clone the repository via one of the following commands (`<github username>` is your Github username):
  * either SSH access to repo:<br>
`git clone git@github.com:<github username>/pisa.git`
  * or HTTPS access to repo:<br>
`git clone https://github.com/<github username>/pisa.git`

#### Using but not developing PISA: Cloning
If you just wish to pull changes from github (and not submit any changes back), you can just clone the sourcecode without creating a fork of the project.

* From a terminal, change into the "parent" directory.<br>
`cd <parent dir>`
* Clone the WIPACrepo/pisa repository via one of the following commands:
  * either SSH access to repo:<br>
`git clone git@github.com:wipacrepo/pisa.git`
  * or HTTPS access to repo:<br>
`git clone https://github.com/wipacrepo/pisa.git`

### Install PISA
```bash
pip install --src $PISA --requirement $PISA/requirements.txt --editable
```
Explanation of the above command:
* `--src $PISA`: Installs PISA from the sourcecode you just cloned in the directory pointed to by the environment variable `$PISA`.
* `--requirement $PISA/requirements.txt`: Specifies the file containing PISA's dependencies for `pip` to install prior to installing PISA.
This file lives at `$PISA/requirements.txt`.
* `--editable`: Allows for changes to the source code to be immediately propagated to your Python installation.
Basically, within your Python source tree, PISA is just a series of links to your source directory, so changes within your source tree are seen directly by your Python installation.

__Notes:__

* You can work with your installation using the usual git commands (pull, push, etc.).
However, these won't recompile any of the extension (i.e. _C/C++_) libraries.
If you want to do so, simply run<br>
`cd $PISA && python setup.py build_ext --inplace`

* If your Python installation was done by an administrator, if you have administrative access, preface the `pip install` command with `sudo`:<br>
`sudo pip install ...`<br>
If you do not have administrative access, you can install PISA as a user module via the `--user` flag:<br>
`pip install --user ...`

## Memetic description of PISA
![Wow](doge.png?raw=true "Wow")
