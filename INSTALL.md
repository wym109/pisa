## Installation Guide

### Requirements

To install this package, you'll need to have the following non-python requirements
* [git](https://git-scm.com/)
* [swig](http://www.swig.org/)
* [hdf5](http://www.hdfgroup.org/HDF5/) — install with `--enable-cxx` option

In Ubuntu Linux, you can install these via
```bash
sudo apt-get install git swig hdf5
```
although you can also obtain `hdf5` and `swig` (and ensure their compatibility with your Python installation) by installing a Python distribution like Anaconda.

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

* Navigate to the [PISA github page](https://github.com/wipacrepo/pisa) and fork the repository by clicking on the ![fork](images/ForkButton.png) button.
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
