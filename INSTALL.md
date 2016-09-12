## Installation Guide

Obtaining packages and handling interdependencies is easiest if you use a Python distribution, such as [Anaconda](https://www.continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy).
Although the selection of maintained packages is smaller than if you use the `pip` command to obtain packages from the Python Package Index (PyPi), you can still use `pip` with these distributions (but always prefer to use the distribution's install mechanism over `pip`, since `pip` does not handle interdependencies well).

The other advantage to these distributions is that they easily install without system administrative privileges (and install in a user directory) and come with the non-Python binary libraries upon which many Python modules rely, making them ideal for setup on clusters, for example.

### Requirements

To install PISA, you'll need to have the following non-python requirements.
Note that these are not installed automatically, and you must install them yourself prior to installing PISA.
Also note that both SWIG and HDF5 support come pre-packaged in the Anaconda and Canopy Python distributions.
* [git](https://git-scm.com/)
* [swig](http://www.swig.org/)
* [hdf5](http://www.hdfgroup.org/HDF5/) — install with `--enable-cxx` option

If you are *not* using Anaconda or Canopy, you can install the above via:
* In Ubuntu,<br>
  'sudo apt-get install git swig hdf5`

The Python requirements that you must install manually:
* [python](http://www.python.org) — version 2.7.x required (tested with 2.7.11)
  * Anaconda: built in
  * Otherwise, if on Linux:<br>
    `sudo apt-get install python-2.7`
* [pip](https://pip.pypa.io/)
  * Anaconda:<br>
    `conda install pip`
  * Otherwise, if on Linux:<br>
    `sudo apt-get install pip`
* [numpy](http://www.numpy.org/)
  * Anaconda:<br>
    `conda install numpy`
  * Otherwise:<br>
    `pip install numpy`
* [cython](http://cython.org/)
  * Anaconda:<br>
    `conda install cython`
  * Otherwise:<br>
    `pip install cython`
* [line_profiler](https://pypi.python.org/pypi/line_profiler/): detailed profiling output<br>
  * Anaconda:<br>
    `conda install line_profiler`
  * pip:<br>
    `pip install line_profiler`

Required Python modules that are installed automatically when you use the pip command detailed later:
* [h5py](http://www.h5py.org/)
* [matplotlib](http://matplotlib.org/)
* [pint](https://pint.readthedocs.org/en/0.7.2/)
* [scipy](http://www.scipy.org/) — version > 0.12 recommended
* [simplejson](https://github.com/simplejson/simplejson)
* [tables](http://www.pytables.org/)
* [uncertainties](https://pythonhosted.org/uncertainties/)

### Optional dependencies

Optional dependencies, on the other hand, ***must be installed manually prior to installing PISA*** if you want the functionality they provide.
The features enabled by and the relevant install commands for optional dependencies are listed below.
* [openmp](http://www.openmp.org): parallelize certain routines on multi-core computers<br>
  *available from your compiler; gcc supports OpenMP, while Clang (OS X) might not*
* [PyCUDA](https://mathema.tician.de/software/pycuda): run certain routines on Nvidia CUDA GPUs (must have compute 2.0 or greater capability)<br>
  `pip install pycuda` (but note that support for CUDA 8 and Pascal architecture GPUs is currently only available in unreleased github version of pycuda)
* [PyROOT](https://root.cern.ch/pyroot): read ROOT cross sections files; must install ROOT on your system in addition to PyROOT. Instructions here work for Ubuntu 15.x and 16.04.
  `sudo apt-get install root-system libroot-bindings-python*`
* [numba](http://numba.pydata.org): just-in-time compilation via LLVM to accelerate certain routines<br>
  * Anaconda:<br>
    `conda install numba`
  * pip:<br>
    `pip install numba`
* [Sphinx](http://www.sphinx-doc.org/en/stable/) - version > 1.3 and [recommonmark](http://recommonmark.readthedocs.io/en/latest/) are required to compile PISA's documentation<br>
    * Anaconda:<br>
      `conda install sphinx`<br>
      `pip install recommonmark`
    * pip:<br>
      `pip install sphinx`<br>
      `pip install recommonmark`

### Install Python
There are many ways of obtaining Python and many ways of installing it.
Here we'll present two options, but this is by no means a complete list.

* Install the [Anaconda](https://docs.continuum.io/anaconda/install) or [Canopy](https://www.enthought.com/products/canopy) Python distribution; 
  * **Note**: Make sure that your `PATH` variable points to e.g. `<anaconda_install_dr>/bin` and *not* your system Python directory. To check this, type: `echo $PATH`; to udpate it, add `export PATH=<anaconda_install_dir>/bin:$PIATH` to your .bashrc file.
* Install Python 2.7.x from the Python website [https://www.python.org/downloads](https://www.python.org/downloads/)

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
1. Obtain access to the `jllanfranchi/pisa` repository by emailing (as a verifiable IceCube member) your **Github username** to Sebastian Böeser (sboeser@uni-mainz.de), cc: John Kelley (jkelley@icecube.wisc.edu)

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
If you wish to modify PISA and contribute your code changes back to the PISA project (*highly recommended!*), fork `jllanfranchi/pisa` from Github.
*(How to work with the `cake` branch of PISA will be detailed below.)*

Forking creates your own version of PISA within your Github account.
You can freely create your own *branch*, modify the code, and then *add* and *commit* changes to that branch within your fork of PISA.
When you want to share your changes with `jllanfranchi/pisa`, you can then submit a *pull request* to `jllanfranchi/pisa` which can be merged by the PISA administrator (after the code is reviewed and tested, of course).

* Navigate to the [PISA github page](https://github.com/jllanfranchi/pisa) and fork the repository by clicking on the ![fork](images/ForkButton.png) button.
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
* Clone the jllanfranchi/pisa repository via one of the following commands:
  * either SSH access to repo:<br>
`git clone git@github.com:jllanfranchi/pisa.git`
  * or HTTPS access to repo:<br>
`git clone https://github.com/jllanfranchi/pisa.git`

### Install PISA
```bash
pip install --editable $PISA -r $PISA/requirements.txt
```
Explanation of the above command:
* First, note that this is ***not** run as administrator. It is discouraged to do so (and has not been tested this way).
* `--editable <dir>`: Installs from `<dir>` and  allows for changes to the source code within to be immediately propagated to your Python installation.
Basically, within your Python source tree, PISA is just a link to your source directory, so changes within your source tree are seen directly by your Python installation.
* `-r $PISA/requirements.txt`: Specifies the file containing PISA's dependencies for `pip` to install prior to installing PISA.
This file lives at `$PISA/requirements.txt`.
* If a specific compiler is set under the `CC` env variable, it will be used,
    otherwise `cc` will be invoked. Note that there have been some problems
    using `clang` under OSX, since it does not support `openmp`. Using `gcc`
    instead works well.

__Notes:__
* You can work with your installation using the usual git commands (pull, push, etc.). However, these ***won't recompile*** any of the extension (i.e. pyx, _C/C++_) libraries. See below for how to handle this case.
* To test if your system compiled the gaussins.pyx Cython file with OpenMP threading support (and to see the speedup you can get using multiple cores for this module), you can run the following test script:
  `python $PISA/pisa/utils/test_gaussians.py --speed`
  The output should show speedups for increasing numbers of threads; if not, then it's likely that OpenMP did not compile on your system.

### Reinstall PISA
* To remove any compiled bits to ensure they get recompiled:
  ```bash
  cd $PISA
  python setup.py clean --all
  pip install --editable $PISA -r $PISA/requirements.txt
  ```
* To just reinstall the Python bits (and only build binaries if they don't already exist)
  ```bash
  pip install --editable $PISA -r $PISA/requirements.txt
  ```

### Compiling the Documentation

To compile a new version of the documentation to html (pdf and other formats are available by replacing `html` with `pdf`, etc.):
```bash
cd $PISA/docs && make html
```

In case code structure has changed, rebuild the apidoc by executing
```bash
cd $PISA && sphinx-apidoc -f -o docs/source pisa
```
