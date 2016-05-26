# Coding conventions
* General Python conventions should be followed
  * [PEP8](https://www.python.org/dev/peps/pep-0008/)
  * Run [pylint](https://www.pylint.org/) to see how egregious your code is (and make it less so) before submitting a pull request (no hard requirement on making it pass 100%, but at least you'll learn how to make it cleaner in the future). Also, static analysis of code (i.e., running pylint) can find bugs you miss in other ways. (Note that there are some nice GUI interfaces for pylint, including e.g. [Spyder IDE](https://github.com/spyder-ide/spyder).)
  * Call functions and methods using full keyword names. E.g.:<br>`my_function(x=1, y=2, z='blahblah')`
  * Avoid positional arguments wherever possible and reasonable (both for Python methods/functions and for command-line interfaces)
  * Always give the user double-dash (`--three-word-arg`) command-line keyword options. Separate words in the option with single-dashes, *not* underscores. Optionally provide single-dash/single-letter options (`-a`) in addition to the double-sash keyword options.

# Naming conventions
* Stage/service naming: see [creating a service](creating_a_service). (Note that the all-lower-case class naming scheme for services is one of the few exceptions we make to the general Python conventions above.)
* Use the following syntax to refer to neutrino signatures `*flavour*_*interaction*_*pid*` where:
  * `flavour` is one of `nue, nuebar, numu, numubar, nutau, nutaubar`
  * `interaction` is one of `cc, nc`
  * `pid` is one of `trck, cscd`

  Omit suffixed fields as necessary to refer to a more general signature for the same flavour, e.g. `nue` refers to all interactions/pids involving electron neutrinos. To refer to multiple signatures separate each signature with a `+` symbol, e.g. `nue_cc+nuebar_cc`

# Documentation conventions
## Docstrings
Docstrings should be used *extensively*, and follow the NumPy/SciPy convention
There are a couple of good alternatives, but in order to make Sphinx auto-generate documentation, we need to stick to just one. NumPy docstrings are visually easy to read for detailed documentation, hence the choice.
* [PEP257: General Python docstring conventions](https://www.python.org/dev/peps/pep-0257/)
* [NumPy/SciPy documentation style guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
* [Example NumPy docstrings in code](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
* Since docstrings will be run through Sphinx using the Napoleon extension, the final arbiter on whether a docstring is acceptable is the output generated using [Napoleon](http://sphinxcontrib-napoleon.readthedocs.io/)

## Non-docstring documentation
Again, we'll be using [Sphinx](http://www.sphinx-doc.org/en/stable/) to compile documentation into well-structured, easily-browsable HTML and PDF. Therefore, all documentation must be compatible with Sphinx for it to be maximally useful to users/developers. The convention in PISA up to now was to use simple Markdown syntax, which is displayed nicely on Github but not natively interpretable by Sphinx (which uses [reStructuredText](http://docutils.sourceforge.net/rst.html) with Sphinx-specific directives). Fortunately, it appears we can use [recommonmark](https://github.com/rtfd/recommonmark) to allow Sphinx to at least understand these documents superficially, in addition to any future reST docs people want to write to leverage more Sphinx features.

Note that Github understands and renders both Markdown and reStructuredText nicely, so *if you can do so, prefer reST for making such documents*. Markdown *is* dead-easy to write, though, so if learning reST syntax is keeping you from writing documentation, simply do it in Markdown and we'll add reST directives later!

# Testing
If you add a feature: ***add a test that proves it works***. If you find a bug: ***add a test that it's fixed***. Not only does a test ensure that the functionality continues to work / bug continues to be squashed, through future iterations of the code, but it also is another way of communicating the assumptions that went into the code that might not be obvious from the code itself.

## Low-level testing: Unit tests
Write simple functions named `test_<FUNCNAME>` or `test_<CLASSNAME>` within your modules. These get called automatically from [pytest](http://pytest.org/). Pytest picks up exceptions that occur, so sprinkle `assert` statements liberally in your `test_*` functions, and you should test both things you expect to succeed *and* test things that you expect to fail (wrap the latter in `try:/except:/else:` blocks, where the `assert` occurs in the `else:` clause).

We chose [pytest](http://pytest.org/): We'd rather have more unit tests written because they're dead-easy to write than to use the full-featured (and built-in) framework that comes with Python.

Expect pytest to eventually be configured to run with [doctest](https://pytest.org/latest/doctest.html) functionality enabled, so it's also acceptable to put simple tests in docstrings, Markdown documentation files, or reST documentation files. (Just these won't be exercised until we get pytest fully configured to do look for these tests.)

Finally, until we get pytest configured, the `test_*` functions are called in the `__main__` section of many of the `pisa/core` modules (e.g., `map.py`, `binning.py`, `transform.py`, `param.py`). For now, these must be invoked by calling `python <filename>`.

### How to test a service
Look at the `pisa.core.pipeline.stage_test()` function to implement tests for your service. It can run a stage in isolation (either with maps you supply or with dummy `numpy.ones()` maps) or will run a pipeline from beginning up until the stage you're testing. To keep tests synchronized with the code they are testing, it is recommended that configurations be generated within the test code rather than read from an external file. This is not a hard-and-fast rule, but usually results in better long-term outcomes.

## High-level testing
Here we will supply whatever basic configuration files and example data is necessary to fully test all high-level functionality. Outputs should be able to be compared against known-outputs, either exactly (for deterministic processes or pseudo-random processes) or as a ratio with a know result (in the case of a "new" physics implementation).

There should be both quick tests that just show an analysis can make it through from beginning to end (and produce a known output), as well as more thorough tests that check the correctness of a result.

For comparing non-deterministic results, plotting is essential for understanding differences. Difference plots, ratio plots, fractional-difference plots, side-by-side plots, overlays, correlations, ... 

# Physical Quantities

## Units

All physical quantities with units should have units attached. Parameters with units specified also require that their prior specifications adhere to those same units.

This is essential so that, transparently, the user can input units are most comprehensible while computations can be carried out in whatever units have been defined for computation. Furthermore, it is made explicit in the user's configuration files what units were chosen for input and the code for performing computation makes it explicit what units it uses as well.

## Uncertainties

Likewise, measured quantities with uncertainties should have these specified as well. Since there is a performance penalty for computing with uncertainties, this is a feature that can be enabled or disabled according to the task the user is working on. Certain parts of PISA (e.g., the convolutional likelihood metric) will only work if errors are propagated through the analysis.

## Further documentation
For more information on using units and uncertainties, see the [Units and uncertainties](units_and_uncertainties) page.
