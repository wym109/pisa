# Statistics functions

The `stats.py` file contains a collection of function to calculate statistical measures from PISA MapSets

## Log-Likelihoods

These all assume that our data is poisson distributed with the usual definition of its pdf as:

![poisson_pdf](images/poisson_pdf.png)
<!---
P(k,\lambda) = \frac{\lambda^k e^{-\lambda}}{\Gamma(k+1)}
--->

Note that k-factorial was replaced by the gamma function to generalize it to non-integer values (used in Asimov calculations)

The logarithem of the actual likelihood is used for numerical reasons, which also means the total likelihood becomes a sum instead of a product, just sayin'... 


### llh

The log-likelihood calculates the total of the bin-by bin log-likelihood given two maps representing `k` (= observed values) and `lambda` (= expected values).

### conv_llh

This likelihood takes into account any uncertainties on the expected values (from e.g. finite MC statistics).

### barlow_llh

This likelihood takes into account the finite MC statistics uncertainties on the expected values

## Chi-Square Values

Bla

### chi2

more bla

### mod_chi2
