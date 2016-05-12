## Scripts

The processing package of PISA was developed in order to aggregate
information relevant to the LLR analysis which had been run on
multiple CPU cores on High Performance Computing Clusters
(HPCC). Additionally, analysis tools were developed to analyze the
complex output from the LLH runs as well as the log files, so that
mining for information related to the minimizer and posterior
distributions can be performed.

The `llr/` directory contains the scripts for aggretating llr analysis
and log files after an LLR run on an HPCC, as well as the plotting
scripts to display standard information, and modules to easily extend
the type of data to be displayed.

### Additionally required dependencies:
* `seaborn` (Python visualization library)
* `pandas` (mandatory dependency of seaborn, should be pulled in
automatically if installed via `pip install seaborn`)
* `tabulate` (Python pretty-printing of tabular data,
e.g. `pip install tabulate`)
