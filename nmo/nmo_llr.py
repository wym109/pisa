
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy

from pisa.core.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.parse_config import parse_config

def fiducial_maker(truth, hypo, minimizer_settings):
    """
    Performs a fiducial fit to the truth to find the best fit parameters
    under the hypothesis set up in hypo

    Parameters
    ----------
    truth : DistributionMaker
        The DistributionMaker object from which the truth is drawn.

    hypo : OrderedDict
        The parsed config file with which to define the hypothesis.
        We pass the config so that we can make a copy of it and replace the
        free parameter values with the best fit ones found in this function.

    minimizer_settings : string
        A string pointing to the location of the file containing the settings
        that define how the minimizer operates.

    Returns
    -------
    A DistributionMaker object set up to the best fit parameters under the 
    hypothesis set in hypo.

    """

    # Make a deecopy that we will modify later for return
    fid_config = deepcopy(hypo)
    hypo_maker = DistributionMaker(hypo)
    # Create the analysis object and then do the fiducial fit
    fid_analysis = Analysis(data_maker=truth,
                            template_maker=hypo_maker,
                            metric='llh',
                            minimizer_settings = minimizer_settings)
    fid_analysis.generate_pseudodata('asimov')
    best_fit = fid_analysis.find_best_fit()
    # Set the parameters in fid_maker to those from the best fit
    fid_maker = DistributionMaker(fid_config)
    for pname in hypo_maker.params.free.names:
        fid_maker.params[pname].value = best_fit[pname]
        
    return fid_maker


def nmo_analysis(truth,template_settings,minimizer_settings,
                 trials = 1,injected=None):
    """
    Runs the NMO analysis assuming the truth provided. This can be either
    from a simulation or from truth, but will be an instance of 
    DistributionMaker. The template_settings is used to fit to pseudo data
    constructed from the truth.

    Parameters
    ----------
    truth : DistributionMaker
        The DistributionMaker object from which the truth is drawn.

    template_settings : string
        A string pointing to the location of a pipeline settings ini file from
        which to construct all of the fit templates.

    minimizer_settings : string
        A string pointing to the location of the file containing the settings
        that define how the minimizer operates.

    trials : integer
        The number of pseudo-experiments to be created.

    injected : string
        A string with the injected truth if known. Expected to be 'NO' or 'IO'

    Returns
    -------
    A dictionary object containing the output of the NMO analysis
    
    """

    # First we need to make the DisitrubitonMaker objects from which to
    # construct the NO and IO hypotheses from.
    NO_hypo_config = from_file(template_settings)
    NO_hypo_config.set('stage:osc', 'param_selector', 'nh')
    NO_hypo_parsed = parse_config(NO_hypo_config)
    NO_hypo_maker = DistributionMaker(NO_hypo_parsed)
    IO_hypo_config = from_file(template_settings)
    IO_hypo_config.set('stage:osc', 'param_selector', 'ih')
    IO_hypo_parsed = parse_config(IO_hypo_config)
    IO_hypo_maker = DistributionMaker(IO_hypo_parsed)

    # Then we do the initial fiducial fits to get the best fit NO and IO
    if injected == 'NO':
        # By construction, NO best fit will be the injected
        print ">> NO_fiducial is injected"
        NO_fid_maker = DistributionMaker(NO_hypo_parsed)
        # Perform fit to truth to find best fit to IO
        print ">> IO_fiducial will be fit for"
        IO_fid_maker = fiducial_maker(truth=truth,
                                      hypo=IO_hypo_parsed,
                                      minimizer_settings=minimizer_settings)
    elif injected == 'IO':
        # By construction, IO best fit will be the injected
        print ">> IO_fiducial is injected"
        IO_fid_maker = DistributionMaker(IO_hypo_parsed)
        # Perform fit to truth to find best fit to NO
        print ">> NO_fiducial will be fit for"
        NO_fid_maker = fiducial_maker(truth=truth,
                                      hypo=NO_hypo_parsed,
                                      minimizer_settings=minimizer_settings)
    else:
        # Both must be fit
        print ">> NO_fiducial will be fit for"
        NO_fid_maker = fiducial_maker(truth=truth,
                                      hypo=NO_hypo_parsed,
                                      minimizer_settings=minimizer_settings)
        print ">> IO_fiducial will be fit for"
        IO_fid_maker = fiducial_maker(truth=truth,
                                      hypo=IO_hypo_parsed,
                                      minimizer_settings=minimizer_settings)

    all_results = {}
    fid_keys = ['NO_fiducial','IO_fiducial']
    fid_makers = [NO_fid_maker,IO_fid_maker]
    for fid_key, fid_maker in zip(fid_keys,fid_makers):
        print ">> Testing fiducial model %s"%fid_key
        # Set up the analysis object for the trials
        # The template_maker is overwritten in the llr method so we can
        # set it arbitrarily here. NO is chosen just because.
        analysis = Analysis(data_maker=fid_maker,
                            template_maker=NO_hypo_maker,
                            metric='llh',
                            minimizer_settings = minimizer_settings)
        analysis.minimizer_settings = from_file(args.minimizer_settings)

        results = []
        for trial in range(1,trials+1):
            print ">>> Running trial %i"%trial
            analysis.generate_pseudodata('poisson')
            results.append(analysis.llr(template_maker0 = NO_hypo_maker,
                                        template_maker1 = IO_hypo_maker,
                                        hypo0 = 'NO',
                                        hypo1 = 'IO'))
        all_results[fid_key] = results

    return all_results

        
if __name__ == '__main__':

    parser = ArgumentParser(
        description='''Performs the LLR analysis for calculating the NMO 
        sensitivity of the distribution made from data-settings compared with
        hypotheses generated from template-settings.

        Currently the output should be a json file containing the dictionary 
        of best fit and likelihood values.'''
    )
    parser.add_argument(
        '-d', '--data-settings', type=str,
        metavar='configfile', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    parser.add_argument(
        '-t', '--template-settings',
        metavar='CONFIGFILE', required=True,
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '-m', '--minimizer-settings', type=str,
        metavar='JSONFILE', required=True,
        help='''Settings related to the optimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '-n', '--num_trials', type=int, default=1,
        help='''number of trials'''
    )
    parser.add_argument(
        '-o', '--outfile', metavar='FILE',
        type=str, action='store', default='out.json',
        help='file to store the output'
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.v)

    if args.data_settings is None:
        logging.warn('No data_settings provided. It is therefore assumed '
                     'that you want to do a full MC study, therefore results '
                     'for both true_NO and true_IO will be produced. If you do '
                     'not want this please provide data_settings for your '
                     'preferred truth or for real data.')
        all_results = {}
        print "> Running NO as truth"
        NO_data_config = from_file(args.template_settings)
        NO_data_config.set('stage:osc', 'param_selector', 'nh')
        NO_data_parsed = parse_config(NO_data_config)
        NO_data_maker = DistributionMaker(NO_data_parsed)
        all_results['true_NO'] = \
            nmo_analysis(truth = NO_data_maker,
                         template_settings = args.template_settings,
                         minimizer_settings = args.minimizer_settings,
                         trials = args.num_trials,
                         injected = 'NO')
                           
        print "> Running IO as truth"
        IO_data_config = from_file(args.template_settings)
        IO_data_config.set('stage:osc', 'param_selector', 'ih')
        IO_data_parsed = parse_config(IO_data_config)
        IO_data_maker = DistributionMaker(IO_data_parsed)
        all_results['true_IO'] = \
            nmo_analysis(truth = IO_data_maker,
                         template_settings = args.template_settings,
                         minimizer_settings = args.minimizer_settings,
                         trials = args.num_trials,
                         injected = 'IO')
        
    else:
        all_results = {}
        print "> Running user-specified as truth"
        data_maker = DistributionMaker(args.data_settings)
        all_results['true_data'] = \
            nmo_analysis(truth = data_maker,
                         template_settings = args.template_settings,
                         minimizer_settings = args.minimizer_settings,
                         trials = args.num_trials)


    logging.info("Output will be saved to %s"%args.outfile)
    to_file(all_results,args.outfile)
    
