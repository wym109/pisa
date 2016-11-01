# Oscillations Parameter Settings

This directory should contain standard settings for the oscillations parameters. Typically we have used [Nu-Fit](http://www.nu-fit.org/) to define our fiducial model, and so you should find those here. The naming convention adopted is that we neglect the decimal point, so nufitv20 here is Nu-Fit v2.0 on the website. In these files, the priors included on &#952;<sub>23</sub> are the "shifted" ones, as to be consistent with what was done in the LoI V2. For an explanation of the different &#952;<sub>23</sub> priors please check the README file in that directory.

An important note on the atmospheric mass splitting values included in the files - We always use &#916;m<sup>2</sup><sub>31</sub> regardless of ordering, whereas Nu-Fit report &#916;m<sup>2</sup><sub>3l</sub> (that is an l instead of a 1), which is always the bigger of the two mass splittings. Thus, in order to have the correct value in our configuration files we must add &#916;m<sup>2</sup><sub>21</sub> to the inverted ordering &#916;m<sup>2</sup><sub>31</sub> value from Nu-Fit. That is, the _absolute_ value will decrease.

The files included here are:

* nufitv20.cfg - Containing the fiducial model from [Nu-Fit v2.0](http://www.nu-fit.org/?q=node/92)
* nufitv22.cfg - Containing the fiducial model from [Nu-Fit v2.2](http://www.nu-fit.org/?q=node/123)
* loiv2.cfg - Contains the fiducial model used in the PINGU LoI version 2.

Also included here is the slightly tangential file `earth.cfg` which contains standard values for the electron densities in the Earth. It also contains a standard choice for the propagation height, which is the assumed injection height in the atmosphere of the neutrinos when calculating their baseline. Lastly, it also contains a value for the detector depth which is appropriate for IceCube.